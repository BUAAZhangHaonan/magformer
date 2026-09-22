# -*- coding: utf-8 -*-
"""
Depth geometry prior for attention bias.

Inspired by DFormerv2 (CVPR 2025). Uses depth distances to create
decomposed attention biases with multi-head decay rates.
"""

import torch
from torch import nn
import torch.nn.functional as F


class DepthGeometryPrior(nn.Module):
    """Decomposed depth geometry prior for attention bias.

    Inspired by DFormerv2 (CVPR 2025). Uses depth distances to create
    attention biases per row/column (decomposed for efficiency).
    Multi-head decay rates capture multi-scale geometry.
    """

    def __init__(self, num_heads, decay_range=(0.75, 1.0)):
        super().__init__()
        self.num_heads = num_heads
        betas = torch.linspace(decay_range[0], decay_range[1], num_heads)
        self.register_buffer('log_betas', torch.log(betas))

    def forward(self, depth_map, feature_stride, num_patches_h, num_patches_w):
        """
        Args:
            depth_map: (B, 1, H, W) raw depth
            feature_stride: int, stride of current feature level
            num_patches_h, num_patches_w: spatial dims of feature map
        Returns:
            row_bias: (B, num_heads, num_patches_h*num_patches_w) for cross-attn
        """
        B = depth_map.shape[0]
        kh = max(1, depth_map.shape[2] // num_patches_h)
        kw = max(1, depth_map.shape[3] // num_patches_w)
        depth_pooled = F.avg_pool2d(depth_map, (kh, kw))  # (B, 1, H', W')
        H_p, W_p = depth_pooled.shape[2], depth_pooled.shape[3]

        # Mean depth per spatial position
        depth_flat = depth_pooled.view(B, 1, H_p * W_p)  # (B, 1, N)

        # Compute depth distance from mean depth per position
        depth_mean = depth_flat.mean(dim=2, keepdim=True)  # (B, 1, 1)
        depth_dist = (depth_flat - depth_mean).abs()  # (B, 1, N)

        # Normalize depth distance to [0, 1] per sample
        depth_dist = depth_dist / (depth_dist.max(dim=2, keepdim=True)[0].clamp(min=1e-6) + 1e-6)

        # Apply per-head decay: beta^d where d is normalized depth distance
        # log_betas: (num_heads,)
        log_betas = self.log_betas.view(1, self.num_heads, 1)  # (1, H, 1)
        # bias = exp(log_beta * depth_dist) for each head
        # depth_dist: (B, 1, N) -> expand to (B, H, N)
        bias = torch.exp(log_betas * depth_dist.expand(B, self.num_heads, -1))  # (B, H, N)

        # Reshape to match feature map spatial dims
        # bias shape: (B, num_heads, num_patches_h * num_patches_w)
        return bias
