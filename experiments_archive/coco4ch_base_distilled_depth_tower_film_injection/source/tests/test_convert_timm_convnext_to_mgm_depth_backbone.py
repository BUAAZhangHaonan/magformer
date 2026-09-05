from __future__ import annotations

import torch

from scripts.analysis.convert_timm_convnext_to_mgm_depth_backbone import (
    map_timm_convnext_state_dict_to_mgm_depth_backbone,
)


def test_map_timm_convnext_state_dict_to_mgm_depth_backbone_maps_expected_keys() -> None:
    timm_state = {
        "stem.0.weight": torch.randn(96, 1, 4, 4),
        "stem.0.bias": torch.randn(96),
        "stem.1.weight": torch.randn(96),
        "stem.1.bias": torch.randn(96),
        "stages.1.downsample.0.weight": torch.randn(96),
        "stages.1.downsample.0.bias": torch.randn(96),
        "stages.1.downsample.1.weight": torch.randn(192, 96, 2, 2),
        "stages.1.downsample.1.bias": torch.randn(192),
        "stages.0.blocks.0.gamma": torch.randn(96),
        "stages.0.blocks.0.conv_dw.weight": torch.randn(96, 1, 7, 7),
        "stages.0.blocks.0.conv_dw.bias": torch.randn(96),
        "stages.0.blocks.0.norm.weight": torch.randn(96),
        "stages.0.blocks.0.norm.bias": torch.randn(96),
        "stages.0.blocks.0.mlp.fc1.weight": torch.randn(384, 96),
        "stages.0.blocks.0.mlp.fc1.bias": torch.randn(384),
        "stages.0.blocks.0.mlp.fc2.weight": torch.randn(96, 384),
        "stages.0.blocks.0.mlp.fc2.bias": torch.randn(96),
        # Unrelated key should be ignored
        "head.fc.weight": torch.randn(1000, 768),
    }

    mapped = map_timm_convnext_state_dict_to_mgm_depth_backbone(timm_state)

    assert mapped["downsample_layers.0.0.weight"].shape == (96, 1, 4, 4)
    assert mapped["downsample_layers.0.1.weight"].shape == (96,)
    assert mapped["downsample_layers.1.0.weight"].shape == (96,)
    assert mapped["downsample_layers.1.1.weight"].shape == (192, 96, 2, 2)
    assert mapped["stages.0.0.gamma"].shape == (96,)
    assert mapped["stages.0.0.dwconv.weight"].shape == (96, 1, 7, 7)
    assert mapped["stages.0.0.norm.weight"].shape == (96,)
    assert mapped["stages.0.0.pwconv1.weight"].shape == (384, 96)
    assert mapped["stages.0.0.pwconv2.weight"].shape == (96, 384)
    assert "head.fc.weight" not in mapped

