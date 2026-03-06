from __future__ import annotations

import pytest
import torch

from magformer.depth_sanity import compute_depth_sanity_report, should_abort_for_depth_sanity


def test_compute_depth_sanity_report_summarizes_depth_and_confidence_ranges() -> None:
    depths = torch.tensor(
        [[[[0.0, 0.25], [0.75, 1.0]]]],
        dtype=torch.float32,
    )
    confidence_maps = {
        "res3": torch.tensor([[[[0.2, 0.8], [0.4, 0.6]]]], dtype=torch.float32)
    }
    pred_masks = torch.tensor(
        [[[[10.0, -10.0], [10.0, -10.0]]]],
        dtype=torch.float32,
    )

    report = compute_depth_sanity_report(
        depths=depths,
        confidence_maps=confidence_maps,
        pred_masks=pred_masks,
    )

    assert report["depth"]["min"] == 0.0
    assert report["depth"]["max"] == 1.0
    assert report["confidence"]["res3"]["min"] == pytest.approx(0.2)
    assert report["confidence"]["res3"]["max"] == pytest.approx(0.8)
    assert 0.0 < report["masks"]["foreground_ratio"] < 1.0


def test_should_abort_for_depth_sanity_flags_collapsed_confidence() -> None:
    report = {
        "depth": {"min": 0.0, "max": 1.0, "mean": 0.5, "std": 0.25},
        "confidence": {"res3": {"min": 0.5, "max": 0.5, "mean": 0.5, "std": 0.0}},
        "masks": {"foreground_ratio": 0.5},
    }

    should_abort, reasons = should_abort_for_depth_sanity(report)

    assert should_abort is True
    assert any("confidence" in reason for reason in reasons)
