from __future__ import annotations

import pytest
import torch

from magformer.models.magformer.fusion import ModalityFusionModule


def _features(fill: float) -> dict[str, torch.Tensor]:
    return {
        "res2": torch.full((1, 8, 8, 8), fill, dtype=torch.float32),
        "res3": torch.full((1, 8, 4, 4), fill, dtype=torch.float32),
    }


class _RecordingAttention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict[str, torch.Tensor]] = []

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, None]:
        del need_weights
        self.calls.append(
            {
                "query": query.detach().clone(),
                "key": key.detach().clone(),
                "value": value.detach().clone(),
            }
        )
        return query, None


def _make_prior_guided_fusion(
    *,
    prior_enabled: bool = True,
    prior_use_grad: bool = True,
    prior_use_var: bool = False,
    prior_use_valid_hole: bool = False,
) -> ModalityFusionModule:
    fusion = ModalityFusionModule(
        image_feature_dims=[8, 8],
        depth_feature_dims=[8, 8],
        scale_keys=["res2", "res3"],
        mode="prior_guided_cross_attn",
        fuse_scales=["res3"],
        prior_enabled=prior_enabled,
        prior_use_grad=prior_use_grad,
        prior_use_var=prior_use_var,
        prior_use_valid_hole=prior_use_valid_hole,
        cross_attn_heads=2,
        cross_attn_downsample=1,
        post_fuse_norm=False,
    )
    fusion.align_image[0] = torch.nn.Identity()
    fusion.align_image[1] = torch.nn.Identity()
    fusion.align_depth[0] = torch.nn.Identity()
    fusion.align_depth[1] = torch.nn.Identity()
    return fusion


def test_prior_guided_cross_attn_only_updates_selected_scale_and_preserves_shape() -> None:
    fusion = _make_prior_guided_fusion()

    fused, confidence_maps, losses = fusion(
        image_features=_features(0.25),
        depth_features=_features(0.75),
        depth_raw=torch.linspace(0.1, 0.9, steps=32 * 32, dtype=torch.float32).reshape(1, 1, 32, 32),
    )

    assert set(fused.keys()) == {"res2", "res3"}
    assert torch.equal(fused["res2"], torch.full((1, 8, 8, 8), 0.25))
    assert fused["res3"].shape == (1, 8, 4, 4)
    assert torch.isfinite(fused["res3"]).all()
    assert confidence_maps == {}
    assert losses == {}


def test_prior_guided_cross_attn_requires_prior_channels() -> None:
    with pytest.raises(ValueError, match="prior"):
        _make_prior_guided_fusion(
            prior_enabled=False,
            prior_use_grad=False,
            prior_use_var=False,
            prior_use_valid_hole=False,
        )


def test_prior_guided_cross_attn_changes_kv_when_prior_changes() -> None:
    fusion = _make_prior_guided_fusion()
    recorder = _RecordingAttention()
    fusion.prior_guided_cross_attn_layers["res3"] = recorder
    fusion.prior_guided_cross_attn_norms["res3"] = torch.nn.Identity()
    torch.nn.init.ones_(fusion.prior_guided_cross_attn_prior_proj["res3"].weight)
    torch.nn.init.zeros_(fusion.prior_guided_cross_attn_prior_proj["res3"].bias)
    torch.nn.init.zeros_(fusion.prior_guided_cross_attn_gate["res3"].weight)
    torch.nn.init.zeros_(fusion.prior_guided_cross_attn_gate["res3"].bias)

    depth_raw_a = torch.linspace(0.1, 0.9, steps=32 * 32, dtype=torch.float32).reshape(1, 1, 32, 32)
    depth_raw_b = torch.zeros_like(depth_raw_a)

    fusion(
        image_features=_features(0.25),
        depth_features=_features(0.75),
        depth_raw=depth_raw_a,
    )
    key_a = recorder.calls[-1]["key"]

    fusion(
        image_features=_features(0.25),
        depth_features=_features(0.75),
        depth_raw=depth_raw_b,
    )
    key_b = recorder.calls[-1]["key"]

    assert not torch.allclose(key_a, key_b)


def test_prior_guided_cross_attn_zero_init_matches_plain_cross_attn() -> None:
    plain = ModalityFusionModule(
        image_feature_dims=[8, 8],
        depth_feature_dims=[8, 8],
        scale_keys=["res2", "res3"],
        mode="cross_attn",
        fuse_scales=["res3"],
        prior_enabled=True,
        prior_use_grad=True,
        prior_use_var=False,
        prior_use_valid_hole=False,
        cross_attn_heads=2,
        cross_attn_downsample=1,
        post_fuse_norm=False,
    )
    guided = _make_prior_guided_fusion()
    plain.align_image[0] = torch.nn.Identity()
    plain.align_image[1] = torch.nn.Identity()
    plain.align_depth[0] = torch.nn.Identity()
    plain.align_depth[1] = torch.nn.Identity()

    guided.cross_attn_layers = getattr(guided, "cross_attn_layers", torch.nn.ModuleDict())
    guided.align_image.load_state_dict(plain.align_image.state_dict())
    guided.align_depth.load_state_dict(plain.align_depth.state_dict())
    guided.prior_guided_cross_attn_layers["res3"].load_state_dict(plain.cross_attn_layers["res3"].state_dict())
    guided.prior_guided_cross_attn_norms["res3"].load_state_dict(plain.cross_attn_norms["res3"].state_dict())

    image_features = _features(0.25)
    depth_features = _features(0.75)
    depth_raw = torch.linspace(0.1, 0.9, steps=32 * 32, dtype=torch.float32).reshape(1, 1, 32, 32)

    plain_fused, _, _ = plain(
        image_features=image_features,
        depth_features=depth_features,
        depth_raw=depth_raw,
    )
    guided_fused, _, _ = guided(
        image_features=image_features,
        depth_features=depth_features,
        depth_raw=depth_raw,
    )

    torch.testing.assert_close(guided_fused["res3"], plain_fused["res3"], atol=1e-6, rtol=1e-6)


def test_prior_guided_cross_attn_keeps_invalid_depth_suppressed() -> None:
    fusion = _make_prior_guided_fusion(
        prior_enabled=True,
        prior_use_grad=False,
        prior_use_var=False,
        prior_use_valid_hole=True,
    )
    recorder = _RecordingAttention()
    fusion.prior_guided_cross_attn_layers["res3"] = recorder
    fusion.prior_guided_cross_attn_norms["res3"] = torch.nn.Identity()
    torch.nn.init.zeros_(fusion.prior_guided_cross_attn_prior_proj["res3"].weight)
    torch.nn.init.zeros_(fusion.prior_guided_cross_attn_prior_proj["res3"].bias)
    torch.nn.init.zeros_(fusion.prior_guided_cross_attn_gate["res3"].weight)
    torch.nn.init.constant_(fusion.prior_guided_cross_attn_gate["res3"].bias, 3.0)

    depth_raw = torch.full((1, 1, 4, 4), 0.5, dtype=torch.float32)
    depth_raw[:, :, 0, 0] = 0.0

    fusion(
        image_features=_features(0.25),
        depth_features=_features(1.0),
        depth_raw=depth_raw,
    )

    key = recorder.calls[-1]["key"].reshape(1, 4, 4, 8)
    assert torch.count_nonzero(key[:, 0, 0, :]) == 0
