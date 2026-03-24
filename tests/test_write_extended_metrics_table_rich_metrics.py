from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


def _write_min_val_annotations(dataset_root: Path) -> None:
    ann_dir = dataset_root / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "images": [
            {
                "id": 1,
                "file_name": "val_000001.png",
                "width": 10,
                "height": 10,
            }
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "iscrowd": 0,
                "area": 16.0,
                "bbox": [2.0, 2.0, 4.0, 4.0],
                "segmentation": [[2.0, 2.0, 6.0, 2.0, 6.0, 6.0, 2.0, 6.0]],
            }
        ],
        "categories": [{"id": 1, "name": "component"}],
    }
    (ann_dir / "instances_val.json").write_text(json.dumps(payload), encoding="utf-8")


def test_extended_metrics_table_includes_bbox_detail_and_prf50_columns(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "analysis" / "write_extended_metrics_table.py"

    dataset_root = tmp_path / "dataset"
    _write_min_val_annotations(dataset_root)

    model_dir = tmp_path / "demo_model"
    model_dir.mkdir(parents=True)
    (model_dir / "metadata.json").write_text(
        json.dumps({"dataset_root": str(dataset_root)}),
        encoding="utf-8",
    )
    (model_dir / "coco_instances_results.json").write_text(
        json.dumps(
            [
                {
                    "image_id": 1,
                    "category_id": 1,
                    "score": 0.95,
                    "bbox": [2.0, 2.0, 4.0, 4.0],
                    "segmentation": [[2.0, 2.0, 6.0, 2.0, 6.0, 6.0, 2.0, 6.0]],
                }
            ]
        ),
        encoding="utf-8",
    )

    summary = {
        "experiment": "demo",
        "output_root": str(tmp_path),
        "demo_model": {
            "status": "ok",
            "best": {
                "segm": {"AP": 50.0, "AP50": 75.0, "AP75": 55.0},
                "bbox": {"AP": 40.0, "AP50": 60.0, "AP75": 35.0},
            },
            "last": {
                "segm": {"AP": 49.0},
                "bbox": {"AP": 39.0},
            },
            "params_trainable": 1234,
            "wall_time_sec": 12.0,
            "peak_memory_mb": 345.0,
            "inference": {"latency_ms_mean": 8.0, "throughput_fps": 125.0, "status": "ok"},
        },
    }
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    out_json = tmp_path / "extended.json"
    out_csv = tmp_path / "extended.csv"
    out_md = tmp_path / "extended.md"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--summary",
            str(summary_path),
            "--out-json",
            str(out_json),
            "--out-csv",
            str(out_csv),
            "--out-md",
            str(out_md),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    rows = json.loads(out_json.read_text(encoding="utf-8"))
    row = rows[0]
    assert float(row["best_bbox_AP50"]) == 60.0
    assert float(row["best_bbox_AP75"]) == 35.0
    assert float(row["segm_precision_at_50"]) == pytest.approx(100.0)
    assert float(row["segm_recall_at_50"]) == pytest.approx(100.0)
    assert float(row["segm_f1_at_50"]) == pytest.approx(100.0)

    markdown = out_md.read_text(encoding="utf-8")
    assert "Best bbox AP50" in markdown
    assert "Best bbox AP75" in markdown
    assert "P@50" in markdown
    assert "R@50" in markdown
    assert "F1@50" in markdown
