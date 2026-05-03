from __future__ import annotations

import torch

from magformer.models.magformer.fusion import ModalityFusionModule


def _features(fill: float) -> dict[str, torch.Tensor]:
    return {
        "res2": torch.full((1, 8, 8, 8), fill, dtype=torch.float32),
        "res3": torch.full((1, 8, 4, 4), fill, dtype=torch.float32),
    }


def test_cross_attn_only_updates_selected_scale_and_preserves_shape() -> None:
    fusion = ModalityFusionModule(
        image_feature_dims=[8, 8],
        depth_feature_dims=[8, 8],
        scale_keys=["res2", "res3"],
        mode="cross_attn",
        fuse_scales=["res3"],
        prior_enabled=True,
        prior_use_grad=True,
        prior_use_var=False,
        prior_use_valid_hole=False,
        post_fuse_norm=False,
        cross_attn_heads=2,
        cross_attn_downsample=1,
    )

    fused, confidence_maps, losses = fusion(
        image_features=_features(0.0),
        depth_features=_features(1.0),
        depth_raw=torch.ones((1, 1, 32, 32)),
    )

    assert set(fused.keys()) == {"res2", "res3"}
    assert torch.equal(fused["res2"], torch.zeros((1, 8, 8, 8)))
    assert fused["res3"].shape == (1, 8, 4, 4)
    assert confidence_maps == {}
    assert losses == {}
