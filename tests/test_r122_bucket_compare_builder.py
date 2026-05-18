from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest


def _rect(x0: int, y0: int, x1: int, y1: int) -> list[list[float]]:
    return [[float(x0), float(y0), float(x1), float(y0), float(x1), float(y1), float(x0), float(y1)]]


def _annotation(ann_id: int, image_id: int, category_id: int, rect: tuple[int, int, int, int]) -> dict[str, object]:
    x0, y0, x1, y1 = rect
    return {
        "id": ann_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": [x0, y0, x1 - x0, y1 - y0],
        "area": float((x1 - x0) * (y1 - y0)),
        "iscrowd": 0,
        "segmentation": _rect(x0, y0, x1, y1),
    }


def _prediction(image_id: int, category_id: int, score: float, rect: tuple[int, int, int, int]) -> dict[str, object]:
    x0, y0, x1, y1 = rect
    return {
        "image_id": image_id,
        "category_id": category_id,
        "score": score,
        "bbox": [x0, y0, x1 - x0, y1 - y0],
        "segmentation": _rect(x0, y0, x1, y1),
    }


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _fixture(tmp_path: Path, *, missing_gt_segmentation: bool = False) -> tuple[Path, Path]:
    annotations = [
        _annotation(1, 1, 1, (0, 0, 8, 8)),
        _annotation(2, 1, 1, (12, 0, 32, 20)),
        _annotation(3, 2, 1, (0, 0, 40, 40)),
    ]
    if missing_gt_segmentation:
        annotations[0].pop("segmentation")
    coco = {
        "images": [
            {"id": 1, "file_name": "one.png", "height": 64, "width": 64},
            {"id": 2, "file_name": "two.png", "height": 64, "width": 64},
        ],
        "annotations": annotations,
        "categories": [{"id": 1, "name": "object"}],
    }
    predictions = [
        _prediction(1, 1, 0.99, (0, 0, 8, 8)),
        _prediction(1, 1, 0.91, (12, 0, 32, 20)),
        _prediction(2, 1, 0.88, (2, 2, 42, 42)),
        _prediction(2, 1, 0.10, (48, 48, 62, 62)),
    ]
    return _write_json(tmp_path / "gt.json", coco), _write_json(tmp_path / "pred.json", predictions)


def _run_builder(tmp_path: Path, gt_json: Path, pred_json: Path) -> subprocess.CompletedProcess[str]:
    repo_root = Path(__file__).resolve().parents[1]
    return subprocess.run(
        [
            sys.executable,
            str(repo_root / "tools" / "build_r122_bucket_compare.py"),
            "--gt-json",
            str(gt_json),
            "--pred-json",
            str(pred_json),
            "--run-name",
            "r122_remaining75",
            "--output-csv",
            str(tmp_path / "bucket_compare.csv"),
            "--output-json",
            str(tmp_path / "summary.json"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def _read_rows(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {(row["bucket_type"], row["bucket"]): row for row in rows}


def test_bucket_compare_builder_outputs_comparator_columns_and_metrics(tmp_path: Path) -> None:
    gt_json, pred_json = _fixture(tmp_path)

    result = _run_builder(tmp_path, gt_json, pred_json)

    assert result.returncode == 0, result.stderr
    rows = _read_rows(tmp_path / "bucket_compare.csv")
    required = {"run", "bucket_type", "bucket", "boundary_f_mean", "best_iou_mean", "recall75", "fp75"}
    assert required <= set(next(iter(rows.values())))
    assert {"tiny_area_le_256", "small_257_1024", "large_gt1024"} <= {
        bucket for bucket_type, bucket in rows if bucket_type == "area_bucket"
    }
    assert {"low25"} <= {bucket for bucket_type, bucket in rows if bucket_type == "density_bucket"}
    assert rows[("area_bucket", "tiny_area_le_256")]["run"] == "r122_remaining75"
    assert float(rows[("area_bucket", "tiny_area_le_256")]["best_iou_mean"]) == pytest.approx(1.0)
    assert float(rows[("area_bucket", "small_257_1024")]["recall75"]) == pytest.approx(1.0)
    assert int(rows[("area_bucket", "large_gt1024")]["fp75"]) == 1
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["run"] == "r122_remaining75"
    assert summary["gt"] == 3
    assert summary["pred"] == 4


def test_bucket_compare_builder_errors_when_gt_segmentation_is_missing(tmp_path: Path) -> None:
    gt_json, pred_json = _fixture(tmp_path, missing_gt_segmentation=True)

    result = _run_builder(tmp_path, gt_json, pred_json)

    assert result.returncode == 2
    assert "annotations[0] is missing segmentation" in result.stderr


def test_bucket_compare_builder_errors_when_prediction_references_unknown_image(tmp_path: Path) -> None:
    gt_json, pred_json = _fixture(tmp_path)
    predictions = json.loads(pred_json.read_text(encoding="utf-8"))
    predictions[0]["image_id"] = 999
    pred_json.write_text(json.dumps(predictions), encoding="utf-8")

    result = _run_builder(tmp_path, gt_json, pred_json)

    assert result.returncode == 2
    assert "prediction references unknown image_id=999" in result.stderr


def test_bucket_compare_builder_boundary_f_is_bounded_for_shifted_mask(tmp_path: Path) -> None:
    gt_json, pred_json = _fixture(tmp_path)

    result = _run_builder(tmp_path, gt_json, pred_json)
    assert result.returncode == 0, result.stderr
    rows = _read_rows(tmp_path / "bucket_compare.csv")
    boundary_f = float(rows[("area_bucket", "large_gt1024")]["boundary_f_mean"])
    assert 0.0 < boundary_f < 1.0


def test_bucket_compare_builder_errors_on_duplicate_image_ids(tmp_path: Path) -> None:
    gt_json, pred_json = _fixture(tmp_path)
    coco = json.loads(gt_json.read_text(encoding="utf-8"))
    coco["images"].append({"id": 1, "file_name": "duplicate.png", "height": 64, "width": 64})
    gt_json.write_text(json.dumps(coco), encoding="utf-8")

    result = _run_builder(tmp_path, gt_json, pred_json)

    assert result.returncode == 2
    assert "duplicate image id=1" in result.stderr
