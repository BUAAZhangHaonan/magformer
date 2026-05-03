from __future__ import annotations

import torch
from torch import nn


def forward_fused_features(
    *,
    rgb_backbone: nn.Module,
    depth_backbone: nn.Module,
    mgm: nn.Module,
    images_tensor: torch.Tensor,
    depths_tensor: torch.Tensor,
    depth_raw: torch.Tensor,
    rgb_image: torch.Tensor,
    depth_noise_mask: torch.Tensor | None,
    use_depth_path: bool,
):
    rgb_features = rgb_backbone(images_tensor)
    if not use_depth_path:
        return rgb_features, None, {}

    depth_features = depth_backbone(depths_tensor)
    fused_features, confidence_maps, mgm_losses = mgm(
        image_features=rgb_features,
        depth_features=depth_features,
        depth_raw=depth_raw,
        rgb_image=rgb_image,
        depth_noise_mask=depth_noise_mask,
    )
    return fused_features, confidence_maps, mgm_losses

