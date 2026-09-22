# -*- coding: utf-8 -*-
"""
Fused gate-blend kernel: sigmoid(gate_logits) * enhanced + (1 - sigmoid(gate_logits)) * img_a

Replaces 4 separate kernel launches (sigmoid, mul, sub+mul, add) with a single Triton kernel.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# ---------------------------------------------------------------------------
# Pure-PyTorch fallback (used when Triton is unavailable)
# ---------------------------------------------------------------------------
def _torch_fused_gate_blend(gate_logits: torch.Tensor, enhanced: torch.Tensor, img_a: torch.Tensor) -> torch.Tensor:
    gate = torch.sigmoid(gate_logits)
    return gate * enhanced + (1.0 - gate) * img_a


if HAS_TRITON:

    # -----------------------------------------------------------------------
    # Forward kernel
    # -----------------------------------------------------------------------
    @triton.jit
    def _gate_blend_fwd_kernel(
        gate_logits_ptr, enhanced_ptr, img_a_ptr, out_ptr,
        n_elements,
        stride_gl0, stride_gl1, stride_gl2, stride_gl3,
        stride_en0, stride_en1, stride_en2, stride_en3,
        stride_ia0, stride_ia1, stride_ia2, stride_ia3,
        stride_ou0, stride_ou1, stride_ou2, stride_ou3,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        # Compute 4D indices from flat offset for stride-based access.
        # For contiguous tensors (the common case) stride tricks let us just use
        # flat pointers, but we pass strides to handle non-contiguous inputs.
        # Since all inputs share the same logical shape (B, C, H, W) we only
        # need to decompose once.
        #
        # However, for a simple 1D elementwise kernel we can rely on the fact
        # that .contiguous() was called before entering the kernel, making all
        # strides dense.  We keep the stride parameters for correctness but
        # use flat indexing which is valid after contiguous().

        gl = tl.load(gate_logits_ptr + offsets, mask=mask, other=0.0)
        en = tl.load(enhanced_ptr + offsets, mask=mask, other=0.0)
        ia = tl.load(img_a_ptr + offsets, mask=mask, other=0.0)

        s = tl.sigmoid(gl.to(tl.float32))
        out = s * en.to(tl.float32) + (1.0 - s) * ia.to(tl.float32)

        tl.store(out_ptr + offsets, out, mask=mask)

    # -----------------------------------------------------------------------
    # Backward kernel
    # -----------------------------------------------------------------------
    @triton.jit
    def _gate_blend_bwd_kernel(
        grad_out_ptr, gate_logits_ptr, enhanced_ptr, img_a_ptr,
        grad_gl_ptr, grad_en_ptr, grad_ia_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        go = tl.load(grad_out_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        gl = tl.load(gate_logits_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        en = tl.load(enhanced_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        ia = tl.load(img_a_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        s = tl.sigmoid(gl)
        one_minus_s = 1.0 - s

        # d_out/d_gate_logits = s * (1 - s) * (enhanced - img_a)
        grad_gl = go * (s * one_minus_s) * (en - ia)
        # d_out/d_enhanced = s
        grad_en = go * s
        # d_out/d_img_a = 1 - s
        grad_ia = go * one_minus_s

        tl.store(grad_gl_ptr + offsets, grad_gl, mask=mask)
        tl.store(grad_en_ptr + offsets, grad_en, mask=mask)
        tl.store(grad_ia_ptr + offsets, grad_ia, mask=mask)

    # -----------------------------------------------------------------------
    # Autograd Function
    # -----------------------------------------------------------------------
    class _FusedGateBlendFunction(torch.autograd.Function):

        @staticmethod
        def forward(ctx, gate_logits: torch.Tensor, enhanced: torch.Tensor, img_a: torch.Tensor) -> torch.Tensor:
            # Ensure contiguous for flat-index kernel
            gate_logits = gate_logits.contiguous()
            enhanced = enhanced.contiguous()
            img_a = img_a.contiguous()

            n_elements = gate_logits.numel()
            out = torch.empty_like(gate_logits)

            BLOCK_SIZE = 1024
            grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)

            _gate_blend_fwd_kernel[grid](
                gate_logits, enhanced, img_a, out,
                n_elements,
                *gate_logits.stride(),
                *enhanced.stride(),
                *img_a.stride(),
                *out.stride(),
                BLOCK_SIZE=BLOCK_SIZE,
            )

            ctx.save_for_backward(gate_logits, enhanced, img_a)
            return out

        @staticmethod
        def backward(ctx, grad_output: torch.Tensor):
            gate_logits, enhanced, img_a = ctx.saved_tensors
            grad_output = grad_output.contiguous()

            n_elements = gate_logits.numel()
            grad_gl = torch.empty_like(gate_logits)
            grad_en = torch.empty_like(enhanced)
            grad_ia = torch.empty_like(img_a)

            BLOCK_SIZE = 1024
            grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)

            _gate_blend_bwd_kernel[grid](
                grad_output, gate_logits, enhanced, img_a,
                grad_gl, grad_en, grad_ia,
                n_elements,
                BLOCK_SIZE=BLOCK_SIZE,
            )

            return grad_gl, grad_en, grad_ia

    def fused_gate_blend(gate_logits: torch.Tensor, enhanced: torch.Tensor, img_a: torch.Tensor) -> torch.Tensor:
        """Fused sigmoid-gate blend: sigmoid(gate_logits) * enhanced + (1 - sigmoid(gate_logits)) * img_a.

        Args:
            gate_logits: Raw gate logits (before sigmoid), shape (B, C, H, W).
            enhanced: Enhanced feature map, shape (B, C, H, W).
            img_a: Original image feature map, shape (B, C, H, W).

        Returns:
            Blended output with same shape as inputs.
        """
        return _FusedGateBlendFunction.apply(gate_logits, enhanced, img_a)

else:
    fused_gate_blend = _torch_fused_gate_blend
