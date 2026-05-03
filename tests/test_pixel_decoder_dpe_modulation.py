from __future__ import annotations

import torch

from magformer.models.common.pixel_decoder_msdeformattn import MSDeformAttnPixelDecoder


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
