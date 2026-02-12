# -*- coding: utf-8 -*-
"""Depth-guided positional encoding."""

import torch
from torch import nn
import torch.nn.functional as F


class DepthPosEncoding(nn.Module):
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(1, hidden_dim // 2, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, hidden_dim // 2),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=1),
        )

    def forward(self, depth_raw: torch.Tensor, size) -> torch.Tensor:
        if depth_raw.dim() == 3:
            depth_raw = depth_raw.unsqueeze(1)
        depth = F.interpolate(depth_raw, size=size, mode="bilinear", align_corners=False)
        depth = depth - depth.amin(dim=(-2, -1), keepdim=True)
        denom = depth.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        depth = depth / denom
        return self.proj(depth)
