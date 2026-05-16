from __future__ import annotations

import json
from pathlib import Path

import pytest
from pycocotools import mask as mask_utils

from tools.analyze_vc_suda_dataset_protocol import ProtocolError, run_analysis


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _rle(width: int, height: int, x0: int, y0: int, size: int) -> dict[str, object]:
    import numpy as np

    mask = np.zeros((height, width), dtype="uint8")
    mask[y0 : y0 + size, x0 : x0 + size] = 1
    encoded = mask_utils.encode(np.asfortranarray(mask))
    encoded["counts"] = encoded["counts"].decode("ascii")
    return encoded


def test_run_analysis_reports_gt_and_prediction_scale_without_matching(tmp_path: Path) -> None:
    ann_path = _write_json(
        tmp_path / "ann.json",
        {
            "images": [
                {"id": 1, "file_name": "one.png", "width": 32, "height": 32},
                {"id": 2, "file_name": "two.png", "width": 32, "height": 32},
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [0, 0, 10, 10],
                    "area": 100,
                    "segmentation": _rle(32, 32, 0, 0, 10),
                    "iscrowd": 0,
                },
                {
                    "id": 2,
                    "image_id": 2,
                    "category_id": 1,
                    "bbox": [2, 2, 4, 4],
                    "area": 16,
                    "segmentation": _rle(32, 32, 2, 2, 4),
                    "iscrowd": 0,
                },
                {
                    "id": 3,
                    "image_id": 2,
                    "category_id": 1,
                    "bbox": [10, 10, 8, 8],
                    "area": 64,
                    "segmentation": _rle(32, 32, 10, 10, 8),
                    "iscrowd": 0,
                },
            ],
            "categories": [{"id": 1, "name": "component"}],
        },
    )
    pred_path = _write_json(
        tmp_path / "pred.json",
        [
            {
                "image_id": 1,
                "category_id": 1,
                "score": 0.9,
                "bbox": [0, 0, 5, 5],
                "segmentation": _rle(32, 32, 0, 0, 5),
            },
            {
                "image_id": 2,
                "category_id": 1,
                "score": 0.8,
                "bbox": [2, 2, 4, 4],
                "segmentation": _rle(32, 32, 2, 2, 4),
            },
            {
                "image_id": 2,
                "category_id": 1,
                "score": 0.7,
                "bbox": [10, 10, 2, 2],
                "segmentation": _rle(32, 32, 10, 10, 2),
            },
        ],
    )

    summary = run_analysis(
        annotations={"toy": ann_path},
        predictions={"toy_pred": ("toy", pred_path)},
    )

    dataset = summary["datasets"]["toy"]
    assert dataset["images"] == 2
    assert dataset["instances"] == 3
    assert dataset["instances_per_image"]["p50"] == 1.5
    assert dataset["density_buckets"]["1-24"]["images"] == 2
    assert dataset["mask_area"]["p50"] == 64.0
    assert dataset["area_le_256_ratio"] == 1.0

    pred = summary["predictions"]["toy_pred"]
    assert pred["annotation_name"] == "toy"
    assert pred["predictions"] == 3
    assert pred["prediction_count_per_image"]["p50"] == 1.5
    assert pred["predictions_per_gt_instance"]["p50"] == 1.0
    assert pred["scale_vs_gt"]["mask_area_p50_ratio"] == pytest.approx(16.0 / 64.0)
    assert pred["scale_vs_gt"]["bbox_width_p50_ratio"] == pytest.approx(4.0 / 8.0)


def test_run_analysis_fails_when_required_coco_field_is_missing(tmp_path: Path) -> None:
    ann_path = _write_json(
        tmp_path / "bad.json",
        {
            "images": [{"id": 1, "file_name": "one.png", "width": 32, "height": 32}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 1, 1]}],
            "categories": [{"id": 1, "name": "component"}],
        },
    )

    with pytest.raises(ProtocolError, match="annotation 1 missing required field: area"):
        run_analysis(annotations={"bad": ann_path}, predictions={})
