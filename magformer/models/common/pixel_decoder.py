# -*- coding: utf-8 -*-
"""
Simple Pixel Decoder

Produces mask features and transformer memory from multi-scale features.
"""

from typing import Dict, List
import torch
import torch.nn as nn


class SimplePixelDecoder(nn.Module):
    """Minimal pixel decoder using 1x1 projections."""

    def __init__(
        self,
        in_features: List[str],
        in_channels: Dict[str, int],
        hidden_dim: int = 256,
        mask_dim: int = 256,
    ) -> None:
        super().__init__()
        self.in_features = list(in_features)
        self.mask_feature_key = self.in_features[0]
        self.memory_feature_key = self.in_features[-1]

        self.mask_proj = nn.Conv2d(
            in_channels[self.mask_feature_key], mask_dim, kernel_size=1)
        self.memory_proj = nn.Conv2d(
            in_channels[self.memory_feature_key], hidden_dim, kernel_size=1)

    def forward(self, features: Dict[str, torch.Tensor], confidence_maps=None, depth_raw=None, **kwargs) -> Dict[str, torch.Tensor]:
        del confidence_maps, depth_raw, kwargs
        mask_features = self.mask_proj(features[self.mask_feature_key])
        memory = self.memory_proj(features[self.memory_feature_key])
        return {"mask_features": mask_features, "memory": memory}
