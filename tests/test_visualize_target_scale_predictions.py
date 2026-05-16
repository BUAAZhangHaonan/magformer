from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.visualize_target_scale_predictions import (
    compute_image_stats,
    discover_prediction_file,
    select_images_for_contact_sheet,
)


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_discover_prediction_file_prefers_coco_instances_results(tmp_path: Path) -> None:
    _write_json(tmp_path / "metrics.cocoeval.json", {"bbox": {}})
    expected = _write_json(tmp_path / "coco_instances_results.json", [])
    _write_json(tmp_path / "other_results.json", [])

    assert discover_prediction_file(tmp_path) == expected


def test_discover_prediction_file_rejects_ambiguous_results(tmp_path: Path) -> None:
    _write_json(tmp_path / "a_results.json", [])
    _write_json(tmp_path / "b_results.json", [])

    with pytest.raises(ValueError, match="ambiguous prediction JSON"):
        discover_prediction_file(tmp_path)


def test_compute_image_stats_reports_gt_and_prediction_scale() -> None:
    gt_boxes = [[0, 0, 10, 10], [30, 30, 20, 20]]
    model_predictions = {
        "Teacher": [
            {"bbox": [0, 0, 20, 20], "score": 0.9},
            {"bbox": [100, 100, 30, 30], "score": 0.8},
        ],
        "R46": [
            {"bbox": [0, 0, 10, 10], "score": 0.9},
            {"bbox": [30, 30, 20, 20], "score": 0.8},
        ],
    }

    stats = compute_image_stats(1, "sample.png", gt_boxes, model_predictions)

    assert stats["gt_bbox_area_p50"] == pytest.approx(250.0)
    assert stats["models"]["Teacher"]["pred_count"] == 2
    assert stats["models"]["Teacher"]["pred_bbox_area_p50"] == pytest.approx(650.0)
    assert stats["models"]["Teacher"]["best_iou_p50"] == pytest.approx(0.125)
    assert stats["models"]["R46"]["pred_bbox_area_p50"] == pytest.approx(250.0)
    assert stats["models"]["R46"]["best_iou_p50"] == pytest.approx(1.0)


def test_select_images_prioritizes_r46_matches_and_oversized_teacher_r59() -> None:
    rows = [
        {
            "image_id": 1,
            "gt_bbox_area_p50": 100.0,
            "models": {
                "Teacher": {"pred_bbox_area_p50": 1000.0, "best_iou_p50": 0.1},
                "R59": {"pred_bbox_area_p50": 900.0, "best_iou_p50": 0.1},
                "R46": {"pred_bbox_area_p50": 110.0, "best_iou_p50": 0.9},
            },
        },
        {
            "image_id": 2,
            "gt_bbox_area_p50": 100.0,
            "models": {
                "Teacher": {"pred_bbox_area_p50": 2000.0, "best_iou_p50": 0.1},
                "R59": {"pred_bbox_area_p50": 1800.0, "best_iou_p50": 0.1},
                "R46": {"pred_bbox_area_p50": 100.0, "best_iou_p50": 0.5},
            },
        },
        {
            "image_id": 3,
            "gt_bbox_area_p50": 100.0,
            "models": {
                "Teacher": {"pred_bbox_area_p50": 120.0, "best_iou_p50": 0.5},
                "R59": {"pred_bbox_area_p50": 110.0, "best_iou_p50": 0.5},
                "R46": {"pred_bbox_area_p50": 100.0, "best_iou_p50": 0.95},
            },
        },
    ]

    selected = select_images_for_contact_sheet(rows, limit=2)

    assert [row["image_id"] for row in selected] == [1, 2]
