from __future__ import annotations

import pytest
import torch

from tools.diagnose_vc_suda_pseudo_labels import (
    PseudoLabelDiagnosticsError,
    build_arg_parser,
    summarize_pseudo_label_diagnostics,
    summarize_threshold_sweep,
    summarize_pseudo_label_scores,
)


def _result(scores):
    return {
        "scores": torch.tensor(scores, dtype=torch.float32),
        "labels": torch.zeros(len(scores), dtype=torch.long),
        "masks": torch.zeros(len(scores), 2, 2),
        "logits": torch.zeros(len(scores), 2),
    }


def test_summarize_pseudo_label_scores_reports_keep_rate_and_empty_ratio():
    scored = [_result([0.9, 0.6, 0.2]), _result([0.1])]
    kept = [_result([0.9, 0.6]), _result([])]

    summary = summarize_pseudo_label_scores(
        scored,
        kept,
        threshold=0.5,
        threshold_source="vc_suda.pseudo_label.quality_threshold",
    )

    assert summary["images"] == 2
    assert summary["predictions"] == 4
    assert summary["kept"] == 2
    assert summary["keep_rate"] == pytest.approx(0.5)
    assert summary["kept_per_image"] == [2, 0]
    assert summary["empty_images"] == 1
    assert summary["empty_ratio"] == pytest.approx(0.5)
    assert summary["score_distribution"]["min"] == pytest.approx(0.1)
    assert summary["score_distribution"]["max"] == pytest.approx(0.9)
    assert summary["threshold"]["value"] == pytest.approx(0.5)


def test_summarize_threshold_sweep_reports_keep_rate_empty_ratio_and_mean():
    scored = [_result([0.9, 0.6, 0.2]), _result([0.1])]

    sweep = summarize_threshold_sweep(scored, thresholds=[0.15, 0.5, 0.95])

    assert len(sweep) == 3
    assert sweep[0]["threshold"] == pytest.approx(0.15)
    assert sweep[0]["kept"] == 3
    assert sweep[0]["keep_rate"] == pytest.approx(0.75)
    assert sweep[0]["empty_images"] == 1
    assert sweep[0]["empty_ratio"] == pytest.approx(0.5)
    assert sweep[0]["kept_per_image_mean"] == pytest.approx(1.5)
    assert sweep[1]["threshold"] == pytest.approx(0.5)
    assert sweep[1]["kept"] == 2
    assert sweep[1]["keep_rate"] == pytest.approx(0.5)
    assert sweep[1]["empty_images"] == 1
    assert sweep[1]["empty_ratio"] == pytest.approx(0.5)
    assert sweep[1]["kept_per_image_mean"] == pytest.approx(1.0)
    assert sweep[2]["threshold"] == pytest.approx(0.95)
    assert sweep[2]["kept"] == 0
    assert sweep[2]["keep_rate"] == pytest.approx(0.0)
    assert sweep[2]["empty_images"] == 2
    assert sweep[2]["empty_ratio"] == pytest.approx(1.0)
    assert sweep[2]["kept_per_image_mean"] == pytest.approx(0.0)


def test_summarize_pseudo_label_scores_fails_when_predictions_are_empty():
    with pytest.raises(PseudoLabelDiagnosticsError, match="predictions=0"):
        summarize_pseudo_label_scores(
            [_result([])],
            [_result([])],
            threshold=0.7,
            threshold_source="test",
        )


def test_summarize_pseudo_label_scores_fails_when_keep_rate_is_zero():
    with pytest.raises(PseudoLabelDiagnosticsError, match="keep_rate=0") as exc_info:
        summarize_pseudo_label_scores(
            [_result([0.2, 0.3])],
            [_result([])],
            threshold=0.7,
            threshold_source="test",
        )
    assert exc_info.value.summary["predictions"] == 2
    assert exc_info.value.summary["keep_rate"] == 0.0


def test_summarize_pseudo_label_diagnostics_includes_sweep_when_gate_fails():
    with pytest.raises(PseudoLabelDiagnosticsError, match="keep_rate=0") as exc_info:
        summarize_pseudo_label_diagnostics(
            [_result([0.2, 0.3])],
            [_result([])],
            threshold=0.7,
            threshold_source="test",
            threshold_sweep=[0.1, 0.25],
        )

    assert exc_info.value.summary["threshold"]["value"] == pytest.approx(0.7)
    assert exc_info.value.summary["threshold_sweep"][0]["threshold"] == pytest.approx(0.1)
    assert exc_info.value.summary["threshold_sweep"][0]["keep_rate"] == pytest.approx(1.0)
    assert exc_info.value.summary["threshold_sweep"][1]["keep_rate"] == pytest.approx(0.5)


def test_diagnostic_cli_defaults_to_cpu_and_small_smoke_batch():
    args = build_arg_parser().parse_args([])

    assert args.config == "configs/vc_suda_stage_c_1024_teacher8499.yaml"
    assert args.device == "cpu"
    assert args.max_images == 2
    assert args.num_workers == 0


def test_diagnostic_cli_accepts_threshold_sweep_values():
    args = build_arg_parser().parse_args(["--threshold-sweep", "0.05", "0.1", "0.25"])

    assert args.threshold_sweep == [0.05, 0.1, 0.25]
