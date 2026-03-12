from __future__ import annotations

import torch

from magformer.models.magformer.fusion import ModalityFusionModule


def _features(fill: float) -> dict[str, torch.Tensor]:
    return {
        "res2": torch.full((1, 8, 8, 8), fill, dtype=torch.float32),
        "res3": torch.full((1, 8, 4, 4), fill, dtype=torch.float32),
    }


def test_direct_add_only_updates_selected_scale() -> None:
    fusion = ModalityFusionModule(
        image_feature_dims=[8, 8],
        depth_feature_dims=[8, 8],
        scale_keys=["res2", "res3"],
        mode="direct_add",
        fuse_scales=["res3"],
        prior_enabled=False,
        post_fuse_norm=False,
    )
    fusion.align_image[0] = torch.nn.Identity()
    fusion.align_image[1] = torch.nn.Identity()
    fusion.align_depth[0] = torch.nn.Identity()
    fusion.align_depth[1] = torch.nn.Identity()

    fused, confidence_maps, losses = fusion(
        image_features=_features(0.0),
        depth_features={"res2": torch.ones((1, 8, 8, 8)), "res3": torch.full((1, 8, 4, 4), 2.0)},
        depth_raw=torch.ones((1, 1, 32, 32)),
    )

    assert torch.equal(fused["res2"], torch.zeros((1, 8, 8, 8)))
    assert torch.equal(fused["res3"], torch.full((1, 8, 4, 4), 2.0))
    assert confidence_maps == {}
    assert losses == {}


def test_gated_add_only_predicts_confidence_for_selected_scales() -> None:
    fusion = ModalityFusionModule(
        image_feature_dims=[8, 8],
        depth_feature_dims=[8, 8],
        scale_keys=["res2", "res3"],
        mode="gated_add",
        fuse_scales=["res3"],
        prior_enabled=False,
        loss_entropy_weight=0.0,
        post_fuse_norm=False,
    )

    fused, confidence_maps, losses = fusion(
        image_features=_features(0.0),
        depth_features=_features(1.0),
        depth_raw=torch.ones((1, 1, 32, 32)),
    )

    assert set(fused.keys()) == {"res2", "res3"}
    assert torch.equal(fused["res2"], torch.zeros((1, 8, 8, 8)))
    assert set(confidence_maps.keys()) == {"res3"}
    assert losses == {}


def test_channel_attn_only_updates_selected_scale() -> None:
    fusion = ModalityFusionModule(
        image_feature_dims=[8, 8],
        depth_feature_dims=[8, 8],
        scale_keys=["res2", "res3"],
        mode="channel_attn",
        fuse_scales=["res3"],
        prior_enabled=False,
        post_fuse_norm=False,
    )

    fused, confidence_maps, losses = fusion(
        image_features=_features(0.0),
        depth_features=_features(1.0),
        depth_raw=torch.ones((1, 1, 32, 32)),
    )

    assert torch.equal(fused["res2"], torch.zeros((1, 8, 8, 8)))
    assert fused["res3"].shape == (1, 8, 4, 4)
    assert confidence_maps == {}
    assert losses == {}


def test_spatial_gate_only_updates_selected_scale() -> None:
    fusion = ModalityFusionModule(
        image_feature_dims=[8, 8],
        depth_feature_dims=[8, 8],
        scale_keys=["res2", "res3"],
        mode="spatial_gate",
        fuse_scales=["res3"],
        prior_enabled=False,
        post_fuse_norm=False,
    )

    fused, confidence_maps, losses = fusion(
        image_features=_features(0.0),
        depth_features=_features(1.0),
        depth_raw=torch.ones((1, 1, 32, 32)),
    )

    assert torch.equal(fused["res2"], torch.zeros((1, 8, 8, 8)))
    assert fused["res3"].shape == (1, 8, 4, 4)
    assert confidence_maps == {}
    assert losses == {}


def test_cross_attn_runs_on_selected_scale_only() -> None:
    fusion = ModalityFusionModule(
        image_feature_dims=[8, 8],
        depth_feature_dims=[8, 8],
        scale_keys=["res2", "res3"],
        mode="cross_attn",
        fuse_scales=["res3"],
        prior_enabled=False,
        cross_attn_heads=2,
        cross_attn_downsample=2,
        post_fuse_norm=False,
    )
    fusion.align_image[0] = torch.nn.Identity()
    fusion.align_image[1] = torch.nn.Identity()
    fusion.align_depth[0] = torch.nn.Identity()
    fusion.align_depth[1] = torch.nn.Identity()

    fused, confidence_maps, losses = fusion(
        image_features=_features(0.25),
        depth_features=_features(0.75),
        depth_raw=torch.ones((1, 1, 32, 32)),
    )

    assert set(fused.keys()) == {"res2", "res3"}
    assert torch.equal(fused["res2"], torch.full((1, 8, 8, 8), 0.25))
    assert fused["res3"].shape == (1, 8, 4, 4)
    assert torch.isfinite(fused["res3"]).all()
    assert confidence_maps == {}
    assert losses == {}
