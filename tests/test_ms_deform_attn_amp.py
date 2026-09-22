from __future__ import annotations

import warnings

import pytest
import torch

from magformer.models.ops.functions import ms_deform_attn_func as function_module
from magformer.models.ops.modules import ms_deform_attn as module_impl


def test_cpu_dispatch_uses_differentiable_pytorch_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_cuda_apply(*args, **kwargs):
        del args, kwargs
        raise AssertionError("CPU dispatch must not call the CUDA extension")

    monkeypatch.setattr(
        module_impl.MSDeformAttnFunction, "apply", unexpected_cuda_apply
    )
    value = torch.randn(1, 2, 1, 2, requires_grad=True)
    spatial_shapes = torch.tensor([[1, 2]], dtype=torch.long)
    level_start_index = torch.tensor([0], dtype=torch.long)
    sampling_locations = torch.rand(1, 1, 1, 1, 1, 2, requires_grad=True)
    attention_weights = torch.rand(1, 1, 1, 1, 1, requires_grad=True)

    output = module_impl._apply_ms_deform_attn(
        value,
        spatial_shapes,
        level_start_index,
        sampling_locations,
        attention_weights,
        1,
    )
    output.sum().backward()

    assert value.grad is not None
    assert sampling_locations.grad is not None
    assert attention_weights.grad is not None
    assert torch.isfinite(value.grad).all()
    assert torch.isfinite(sampling_locations.grad).all()
    assert torch.isfinite(attention_weights.grad).all()


def test_cuda_dispatch_propagates_extension_failure_without_core_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _CudaValue:
        is_cuda = True

    def extension_failure(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("synthetic extension failure")

    def unexpected_core(*args, **kwargs):
        del args, kwargs
        raise AssertionError("CUDA errors must not fall back to the CPU core")

    monkeypatch.setattr(module_impl.MSDeformAttnFunction, "apply", extension_failure)
    monkeypatch.setattr(module_impl, "ms_deform_attn_core_pytorch", unexpected_core)

    with pytest.raises(RuntimeError, match="synthetic extension failure"):
        module_impl._apply_ms_deform_attn(
            _CudaValue(),
            object(),
            object(),
            object(),
            object(),
            1,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_ms_deform_attn_amp_cuda_forward_backward_uses_fp32_boundary() -> None:
    assert function_module.ms_deform_attn_cuda_available(), (
        "CUDA is available but the MSDeformAttn extension could not be imported: "
        f"{function_module.ms_deform_attn_import_error()}"
    )

    device = torch.device("cuda:0")
    module = module_impl.MSDeformAttn(
        d_model=8,
        n_levels=2,
        n_heads=2,
        n_points=2,
    ).to(device)
    query = torch.randn(1, 3, 8, device=device, requires_grad=True)
    input_flatten = torch.randn(1, 5, 8, device=device, requires_grad=True)
    reference_points = torch.rand(1, 3, 2, 2, device=device)
    spatial_shapes = torch.tensor([[2, 2], [1, 1]], device=device, dtype=torch.long)
    level_start_index = torch.tensor([0, 4], device=device, dtype=torch.long)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            output = module(
                query,
                reference_points,
                input_flatten,
                spatial_shapes,
                level_start_index,
            )
            loss = output.float().square().mean()
        loss.backward()

    assert output.dtype == torch.float16
    assert torch.isfinite(output).all()
    assert query.grad is not None and torch.isfinite(query.grad).all()
    assert input_flatten.grad is not None and torch.isfinite(input_flatten.grad).all()
