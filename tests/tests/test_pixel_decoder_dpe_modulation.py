from __future__ import annotations

import pytest
import torch

from magformer.models.common.pixel_decoder_msdeformattn import MSDeformAttnPixelDecoder


def _build_dpe_topology_probe(dpe_enabled: bool) -> MSDeformAttnPixelDecoder:
    level_names = ["res2", "res3", "res4", "res5"]
    return MSDeformAttnPixelDecoder(
        in_features=level_names,
        in_channels={name: 32 for name in level_names},
        transformer_in_features=["res3", "res4", "res5"],
        hidden_dim=32,
        mask_dim=16,
        transformer_nheads=8,
        transformer_dim_feedforward=64,
        transformer_enc_layers=1,
        dpe_enabled=dpe_enabled,
        maskformer_num_feature_levels=3,
    )


def test_disabling_dpe_only_removes_depth_position_encoder_parameters() -> None:
    enabled = _build_dpe_topology_probe(dpe_enabled=True)
    disabled = _build_dpe_topology_probe(dpe_enabled=False)

    assert enabled.dpe_enabled is True
    assert enabled.depth_pe is not None
    assert disabled.dpe_enabled is False
    assert disabled.depth_pe is None
    assert enabled.transformer_in_features == disabled.transformer_in_features
    assert enabled.decoder_level_names == disabled.decoder_level_names
    assert (
        enabled.transformer_num_feature_levels
        == disabled.transformer_num_feature_levels
        == 3
    )
    assert enabled.maskformer_num_feature_levels == disabled.maskformer_num_feature_levels == 3
    assert len(enabled.input_proj) == len(disabled.input_proj) == 3
    assert len(enabled.transformer.encoder.layers) == len(
        disabled.transformer.encoder.layers
    )

    enabled_state = {
        key: tuple(value.shape) for key, value in enabled.state_dict().items()
    }
    disabled_state = {
        key: tuple(value.shape) for key, value in disabled.state_dict().items()
    }
    dpe_keys = {
        "depth_pe.alpha",
        "depth_pe.mlp.0.weight",
        "depth_pe.mlp.1.weight",
        "depth_pe.mlp.1.bias",
        "depth_pe.mlp.3.weight",
        "depth_pe.mlp.4.weight",
        "depth_pe.mlp.4.bias",
    }
    assert set(enabled_state) - set(disabled_state) == dpe_keys
    assert set(disabled_state) - set(enabled_state) == set()
    assert all(
        enabled_state[key] == disabled_state[key] for key in disabled_state
    )


def test_resolve_dpe_modulation_maps_prefers_generic_maps() -> None:
    generic = {"res3": torch.ones((1, 1, 4, 4), dtype=torch.float32)}
    confidence = {"res3": torch.zeros((1, 1, 4, 4), dtype=torch.float32)}
    resolved = MSDeformAttnPixelDecoder._resolve_dpe_modulation_maps(
        depth_raw=torch.ones((1, 1, 8, 8), dtype=torch.float32),
        confidence_maps=confidence,
        depth_modulation_maps=generic,
    )
    assert resolved is generic


def test_resolve_dpe_modulation_maps_falls_back_to_valid_depth_mask() -> None:
    depth = torch.tensor([[[[0.0, 0.2], [0.0, 0.8]]]], dtype=torch.float32)
    resolved = MSDeformAttnPixelDecoder._resolve_dpe_modulation_maps(
        depth_raw=depth,
        confidence_maps=None,
        depth_modulation_maps=None,
    )
    assert resolved is not None
    assert "depth_valid" in resolved
    mask = resolved["depth_valid"]
    assert mask.shape == depth.shape
    assert float(mask.max()) == 1.0


def test_resolve_dpe_modulation_maps_keeps_all_invalid_depth_disabled() -> None:
    depth = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
    resolved = MSDeformAttnPixelDecoder._resolve_dpe_modulation_maps(
        depth_raw=depth,
        confidence_maps=None,
        depth_modulation_maps=None,
    )
    assert resolved is not None
    assert torch.count_nonzero(resolved["depth_valid"]) == 0


def test_select_dpe_modulation_map_requires_exact_learned_scale() -> None:
    maps = {"res3": torch.ones((1, 1, 4, 4), dtype=torch.float32)}
    selected = MSDeformAttnPixelDecoder._select_dpe_modulation_map(
        feature_name="res3",
        modulation_maps=maps,
    )
    assert selected is maps["res3"]
    with pytest.raises(KeyError, match="res4"):
        MSDeformAttnPixelDecoder._select_dpe_modulation_map(
            feature_name="res4",
            modulation_maps=maps,
        )


def test_select_dpe_modulation_map_broadcasts_explicit_validity_map() -> None:
    valid = torch.ones((1, 1, 8, 8), dtype=torch.float32)
    selected = MSDeformAttnPixelDecoder._select_dpe_modulation_map(
        feature_name="res5",
        modulation_maps={"depth_valid": valid},
    )
    assert selected is valid


def test_four_level_decoder_uses_exact_high_resolution_dpe_order() -> None:
    level_names = ["res2", "res3", "res4", "res5"]
    decoder = MSDeformAttnPixelDecoder(
        in_features=level_names,
        in_channels={name: 32 for name in level_names},
        transformer_in_features=level_names,
        hidden_dim=32,
        mask_dim=16,
        transformer_nheads=8,
        transformer_dim_feedforward=64,
        transformer_enc_layers=1,
        dpe_enabled=True,
        maskformer_num_feature_levels=4,
    )

    assert decoder.transformer_num_feature_levels == 4
    assert decoder.maskformer_num_feature_levels == 4
    assert decoder.decoder_level_names == ["res5", "res4", "res3", "res2"]
    assert len(decoder.input_proj) == 4

    maps = {
        name: torch.full((1, 1, 2, 2), float(index))
        for index, name in enumerate(level_names)
    }
    for name in decoder.decoder_level_names:
        selected = decoder._select_dpe_modulation_map(
            feature_name=name,
            modulation_maps=maps,
        )
        assert selected is maps[name]

    with pytest.raises(KeyError, match="res2"):
        decoder._select_dpe_modulation_map(
            feature_name="res2",
            modulation_maps={name: value for name, value in maps.items() if name != "res2"},
        )
