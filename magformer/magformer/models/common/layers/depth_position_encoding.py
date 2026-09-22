# -*- coding: utf-8 -*-
"""Learnable depth-guided positional encoding."""

from typing import Optional

import torch
from torch import nn


class DepthPosEncoding(nn.Module):
    """Project a normalized depth map into the decoder hidden dimension."""

    def __init__(self, hidden_dim: int = 256, beta: float = 10.0):
        super().__init__()
        if hidden_dim % 2 != 0:
            raise ValueError("hidden_dim must be even")

        groups = min(32, hidden_dim)
        if hidden_dim % groups != 0:
            raise ValueError(
                f"hidden_dim {hidden_dim} must be divisible by GroupNorm groups {groups}"
            )

        self.hidden_dim = hidden_dim
        self.beta = beta
        self.mlp = nn.Sequential(
            nn.Conv2d(1, hidden_dim, kernel_size=1, bias=False),
            nn.GroupNorm(groups, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, bias=False),
            nn.GroupNorm(groups, hidden_dim),
        )
        self.alpha = nn.Parameter(torch.full((hidden_dim, 1, 1), 0.3))

    def forward(
        self,
        depth_raw: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if depth_raw.ndim != 4 or depth_raw.shape[1] != 1:
            raise ValueError(
                f"depth_raw must have shape (B,1,H,W), got {tuple(depth_raw.shape)}"
            )

        depth_float = depth_raw.float()
        if mask is not None:
            if mask.dtype != torch.bool:
                raise TypeError(f"depth position mask must be bool, got {mask.dtype}")
            if mask.device != depth_raw.device:
                raise ValueError("depth position mask and depth input must share a device")
            expected_shape = (
                depth_raw.shape[0],
                depth_raw.shape[2],
                depth_raw.shape[3],
            )
            if mask.shape != expected_shape:
                raise ValueError(
                    f"depth position mask shape {tuple(mask.shape)} must be "
                    f"{expected_shape}"
                )
            depth_float = depth_float.masked_fill(mask.unsqueeze(1), 0)

        depth_transformed = torch.log1p(self.beta * depth_float)
        depth_embedding = self.mlp(depth_transformed) * self.alpha
        if mask is not None:
            # GroupNorm and projection can reintroduce non-zero padded values.
            depth_embedding = depth_embedding.masked_fill(mask.unsqueeze(1), 0)
        return depth_embedding.to(depth_raw.dtype)
