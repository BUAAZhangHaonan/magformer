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
        depth_valid_masks=None,
    )
    assert resolved is generic


def test_resolve_dpe_modulation_maps_uses_depth_valid_masks_for_fallback() -> None:
    depth = torch.tensor([[[[0.0, 0.2], [0.0, 0.8]]]], dtype=torch.float32)
    depth_valid_masks = torch.tensor([[[[True, True], [False, True]]]])
    resolved = MSDeformAttnPixelDecoder._resolve_dpe_modulation_maps(
        depth_raw=depth,
        confidence_maps=None,
        depth_modulation_maps=None,
        depth_valid_masks=depth_valid_masks,
    )
    assert resolved is not None
    assert "depth_valid" in resolved
    mask = resolved["depth_valid"]
    assert mask.shape == depth.shape
    assert torch.equal(mask, depth_valid_masks.float())


def test_resolve_dpe_modulation_maps_does_not_guess_from_depth_when_validity_missing() -> None:
    depth = torch.tensor([[[[0.0, 0.2], [1.0, 0.8]]]], dtype=torch.float32)
    resolved = MSDeformAttnPixelDecoder._resolve_dpe_modulation_maps(
        depth_raw=depth,
        confidence_maps=None,
        depth_modulation_maps=None,
        depth_valid_masks=None,
    )
    assert resolved is None
