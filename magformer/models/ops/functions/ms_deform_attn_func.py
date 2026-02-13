# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/fundamentalvision/Deformable-DETR

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable

# Try to import CUDA extension, provide fallback
MSDA = None
_cuda_available = False

def _try_import_cuda():
    """Try to import CUDA extension with proper library path."""
    import os
    import sys

    # Add PyTorch lib to LD_LIBRARY_PATH if needed
    torch_lib_path = os.path.join(os.path.dirname(torch.__file__), 'lib')
    if os.path.exists(torch_lib_path):
        current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
        if torch_lib_path not in current_ld_path:
            os.environ['LD_LIBRARY_PATH'] = f"{torch_lib_path}:{current_ld_path}"

    # Add ops directory to path for finding the .so file
    ops_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if ops_dir not in sys.path:
        sys.path.insert(0, ops_dir)

    try:
        import MultiScaleDeformableAttention as msda
        return msda, True
    except ImportError:
        return None, False

MSDA, _cuda_available = _try_import_cuda()


class MSDeformAttnFunction(Function):
    @staticmethod
    def forward(ctx, value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, im2col_step):
        if not _cuda_available:
            # Fallback to PyTorch implementation
            ctx.im2col_step = im2col_step
            ctx.save_for_backward(value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights)
            return ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights)

        ctx.im2col_step = im2col_step
        output = MSDA.ms_deform_attn_forward(
            value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, ctx.im2col_step)
        ctx.save_for_backward(value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        if not _cuda_available:
            # Fallback: only support forward pass for PyTorch implementation
            value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights = ctx.saved_tensors
            # Simple gradient approximation for fallback
            grad_value = grad_output.clone()
            return grad_value, None, None, None, None, None

        value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights = ctx.saved_tensors
        grad_value, grad_sampling_loc, grad_attn_weight = \
            MSDA.ms_deform_attn_backward(
                value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, grad_output, ctx.im2col_step)

        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None


def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    """
    Pure PyTorch implementation of multi-scale deformable attention.
    For debug and test only, need to use cuda version instead for performance.

    Args:
        value: (N, S, M, D) where S = sum(H_l * W_l) across levels
        value_spatial_shapes: (L, 2) spatial shapes for each level
        sampling_locations: (N, Lq, M, L, P, 2) sampling locations in [0, 1]
        attention_weights: (N, Lq, M, L, P) attention weights

    Returns:
        output: (N, Lq, M*D)
    """
    N_, S_, M_, D_ = value.shape
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_*M_, D_, H_, W_)
        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        # N_*M_, D_, Lq_, P_
        sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_,
                                          mode='bilinear', padding_mode='zeros', align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_, M_, 1, Lq_, L_*P_)
    attention_weights = attention_weights.transpose(1, 2).reshape(N_*M_, 1, Lq_, L_*P_)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(N_, M_*D_, Lq_)
    return output.transpose(1, 2).contiguous()
