from __future__ import annotations

import pytest
import torch

from magformer.models.magformer.fusion import ModalityFusionModule


def _features(fill: float) -> dict[str, torch.Tensor]:
    return {
        "res2": torch.full((1, 8, 8, 8), fill, dtype=torch.float32),
        "res3": torch.full((1, 8, 4, 4), fill, dtype=torch.float32),
    }


@pytest.mark.parametrize(
    "mode",
    [
        "legacy_gated",
        "direct_add",
        "gated_add",
        "film",
        "cross_attn",
        "prior_guided_cross_attn",
        "channel_attn",
        "spatial_gate",
        "esanet_ctx",
        "cross_gate",
    ],
)
def test_fusion_rejects_unimplemented_legacy_modes(mode: str) -> None:
    with pytest.raises(ValueError, match=f"Unsupported fusion mode: {mode}"):
        ModalityFusionModule(
            image_feature_dims=[8, 8],
            depth_feature_dims=[8, 8],
            scale_keys=["res2", "res3"],
            mode=mode,
            fuse_scales=["res3"],
            prior_enabled=False,
            post_fuse_norm=False,
        )


def test_sa_gate_only_updates_selected_scale() -> None:
    fusion = ModalityFusionModule(
        image_feature_dims=[8, 8],
        depth_feature_dims=[8, 8],
        scale_keys=["res2", "res3"],
        mode="sa_gate",
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


def test_dccg_without_confidence_has_no_confidence_state_or_entropy() -> None:
    fusion = ModalityFusionModule(
        image_feature_dims=[8],
        depth_feature_dims=[8],
        scale_keys=["res3"],
        mode="dccg",
        fuse_scales=["res3"],
        prior_enabled=False,
        post_fuse_norm=False,
        loss_entropy_weight=1.0,
        dccg_use_confidence=False,
    )
    fusion.train()

    assert not hasattr(fusion, "depth_confidence")
    step_before = fusion._dccg_step.detach().clone()

    fused, confidence_maps, losses = fusion(
        image_features={"res3": torch.full((1, 8, 4, 4), 0.25)},
        depth_features={"res3": torch.full((1, 8, 4, 4), 0.75)},
        depth_raw=torch.ones((1, 1, 32, 32)),
    )

    assert fused["res3"].shape == (1, 8, 4, 4)
    assert torch.isfinite(fused["res3"]).all()
    assert confidence_maps == {}
    assert losses == {}
    assert torch.equal(fusion._dccg_step, step_before)


def test_dccg_returns_learned_confidence_with_gradient() -> None:
    fusion = ModalityFusionModule(
        image_feature_dims=[8],
        depth_feature_dims=[8],
        scale_keys=["res3"],
        mode="dccg",
        fuse_scales=["res3"],
        prior_enabled=False,
        post_fuse_norm=False,
        loss_entropy_weight=0.1,
        dccg_use_confidence=True,
    )
    fusion.train()

    step_before = fusion._dccg_step.detach().clone()
    fused, confidence_maps, losses = fusion(
        image_features={"res3": torch.randn((1, 8, 4, 4))},
        depth_features={"res3": torch.randn((1, 8, 4, 4))},
        depth_raw=torch.rand((1, 1, 16, 16)),
    )

    assert set(confidence_maps) == {"res3"}
    assert confidence_maps["res3"].requires_grad
    total = fused["res3"].square().mean() + sum(losses.values())
    total.backward()
    grad = fusion.depth_confidence.net[-1].weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert torch.count_nonzero(grad) > 0
    assert torch.equal(fusion._dccg_step, step_before)

    fusion.advance_optimizer_step()
    assert torch.equal(fusion._dccg_step, step_before + 1)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"residual_alpha": 0.05}, "residual_alpha"),
        ({"noise_mask_weight": 0.1}, "noise_mask_weight"),
        ({"mode": "dccg", "prior_enabled": True}, "does not consume depth priors"),
    ],
)
def test_fusion_rejects_silent_noop_options(kwargs, message: str) -> None:
    base = {
        "image_feature_dims": [8],
        "depth_feature_dims": [8],
        "scale_keys": ["res3"],
        "mode": "dccg",
        "fuse_scales": ["res3"],
        "prior_enabled": False,
    }
    base.update(kwargs)
    with pytest.raises(ValueError, match=message):
        ModalityFusionModule(**base)
