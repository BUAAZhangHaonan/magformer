from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from magformer.models.common.backbones import d2_swin
from magformer.models.common.backbones.d2_swin import D2SwinBackbone


def _toy_backbone() -> D2SwinBackbone:
    backbone = D2SwinBackbone.__new__(D2SwinBackbone)
    nn.Module.__init__(backbone)
    backbone.model = nn.Linear(2, 2, bias=False)
    return backbone


def test_model_prefix_is_canonicalized_and_reports_full_coverage(monkeypatch) -> None:
    backbone = _toy_backbone()
    expected = torch.full_like(backbone.model.weight, 3.0)
    monkeypatch.setattr(
        d2_swin,
        "load_torch_checkpoint",
        lambda *args, **kwargs: {"model.weight": expected.clone()},
    )

    report = backbone._load_weights("prefixed.pth")

    torch.testing.assert_close(backbone.model.weight, expected)
    assert report == {
        "loaded": 1,
        "model_tensors": 1,
        "checkpoint_tensors": 1,
        "missing": 0,
        "shape_mismatches": 0,
        "unexpected": 0,
    }


def test_prefix_canonicalization_rejects_colliding_source_keys(monkeypatch) -> None:
    backbone = _toy_backbone()
    value = torch.ones_like(backbone.model.weight)
    monkeypatch.setattr(
        d2_swin,
        "load_torch_checkpoint",
        lambda *args, **kwargs: {
            "weight": value.clone(),
            "model.weight": value.clone(),
        },
    )

    with pytest.raises(RuntimeError, match="key collision"):
        backbone._load_weights("collision.pth")


def test_unknown_wrapper_prefix_fails_with_zero_coverage(monkeypatch) -> None:
    backbone = _toy_backbone()
    monkeypatch.setattr(
        d2_swin,
        "load_torch_checkpoint",
        lambda *args, **kwargs: {
            "module.weight": torch.ones_like(backbone.model.weight),
        },
    )

    with pytest.raises(RuntimeError, match="zero compatible tensors"):
        backbone._load_weights("unsupported-wrapper.pth")
