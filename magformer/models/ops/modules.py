# -*- coding: utf-8 -*-
"""Pure PyTorch substitute for MSDeformAttn interface."""

import torch
from torch import nn


class MSDeformAttn(nn.Module):
    """
    Compatibility implementation of the MSDeformAttn API.

    This implementation preserves the call signature used by Mask2Former-style
    pixel decoders, while using regular multi-head attention on flattened memory.
    """

    def __init__(self, d_model: int, n_levels: int, n_heads: int, n_points: int):
        super().__init__()
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.query_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        query: torch.Tensor,
        reference_points: torch.Tensor,
        value: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        del reference_points, spatial_shapes, level_start_index
        q = self.query_proj(query)
        v = self.value_proj(value)

        if padding_mask is not None:
            valid = (~padding_mask).float().unsqueeze(-1)
            denom = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
            global_context = (v * valid).sum(dim=1, keepdim=True) / denom
        else:
            global_context = v.mean(dim=1, keepdim=True)

        fused = q + global_context
        return self.out_proj(fused)
