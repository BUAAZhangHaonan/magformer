from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_visualize_suite_generates_overlays_and_triptych(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "visualize_suite.py"
    assert script.exists()

    # Minimal COCO dataset (1 image, 1 annotation) under tmp_path/ds
    ds_root = tmp_path / "ds"
    (ds_root / "images" / "val").mkdir(parents=True)
    (ds_root / "annotations").mkdir(parents=True)

    # Create a tiny image
    import numpy as np
    import cv2

    img = np.zeros((32, 32, 3), dtype=np.uint8)
    img[:, :] = (10, 20, 30)
    cv2.imwrite(str(ds_root / "images" / "val" / "0001.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    ann = {
        "images": [{"id": 1, "file_name": "0001.png", "width": 32, "height": 32}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [8, 8, 16, 16],
                "area": 256,
                "iscrowd": 0,
                "segmentation": [[8, 8, 24, 8, 24, 24, 8, 24]],
            }
        ],
        "categories": [{"id": 1, "name": "component"}],
    }
    (ds_root / "annotations" / "instances_val.json").write_text(json.dumps(ann) + "\n", encoding="utf-8")

    # Two dummy model result jsons (same as GT for simplicity)
    out_root = tmp_path / "out"
    (out_root / "magformer").mkdir(parents=True)
    (out_root / "mgm_mask2former").mkdir(parents=True)

    pred_rows = [
        {
            "image_id": 1,
            "category_id": 1,
            "score": 1.0,
            "bbox": [8, 8, 16, 16],
            "segmentation": [[8, 8, 24, 8, 24, 24, 8, 24]],
        }
    ]
    (out_root / "magformer" / "coco_instances_results.json").write_text(json.dumps(pred_rows) + "\n", encoding="utf-8")
    (out_root / "mgm_mask2former" / "coco_instances_results.json").write_text(
        json.dumps(pred_rows) + "\n", encoding="utf-8"
    )

    summary = {
        "experiment": "dummy",
        "output_root": str(out_root),
        "magformer": {"status": "ok", "artifacts": {"coco_instances_results": str(out_root / "magformer" / "coco_instances_results.json")}},
        "mgm_mask2former": {
            "status": "ok",
            "artifacts": {"coco_instances_results": str(out_root / "mgm_mask2former" / "coco_instances_results.json")},
        },
    }
    summary_path = out_root / f"summary_{out_root.name}.json"
    summary_path.write_text(json.dumps(summary) + "\n", encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            str(script),
            "--output-root",
            str(out_root),
            "--dataset-root",
            str(ds_root),
            "--summary",
            str(summary_path),
            "--num-images",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert (out_root / "visualizations" / "magformer" / "overlay" / "overlay_0000_id1.png").exists()
    assert (out_root / "visualizations" / "mgm_mask2former" / "overlay" / "overlay_0000_id1.png").exists()
    assert (out_root / "visualizations" / "triptych_gt_magformer_mgm" / "triptych_0000_id1.png").exists()


def test_visualize_suite_writes_gt_and_new_unet_baseline_overlays_without_triptych_models(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "experiments" / "visualize_suite.py"

    ds_root = tmp_path / "ds"
    (ds_root / "images" / "val").mkdir(parents=True)
    (ds_root / "annotations").mkdir(parents=True)

    import cv2
    import numpy as np

    img = np.zeros((32, 32, 3), dtype=np.uint8)
    img[:, :] = (10, 20, 30)
    cv2.imwrite(str(ds_root / "images" / "val" / "0001.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    ann = {
        "images": [{"id": 1, "file_name": "0001.png", "width": 32, "height": 32}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [8, 8, 16, 16],
                "area": 256,
                "iscrowd": 0,
                "segmentation": [[8, 8, 24, 8, 24, 24, 8, 24]],
            }
        ],
        "categories": [{"id": 1, "name": "component"}],
    }
    (ds_root / "annotations" / "instances_val.json").write_text(json.dumps(ann) + "\n", encoding="utf-8")

    out_root = tmp_path / "out"
    pred_rows = [
        {
            "image_id": 1,
            "category_id": 1,
            "score": 1.0,
            "bbox": [8, 8, 16, 16],
            "segmentation": [[8, 8, 24, 8, 24, 24, 8, 24]],
        }
    ]
    summary = {"experiment": "dummy", "output_root": str(out_root)}
    for model_id in ["cellpose", "stardist", "iaunet"]:
        model_dir = out_root / model_id
        model_dir.mkdir(parents=True)
        results_path = model_dir / "coco_instances_results.json"
        results_path.write_text(json.dumps(pred_rows) + "\n", encoding="utf-8")
        summary[model_id] = {"status": "ok", "artifacts": {"coco_instances_results": str(results_path)}}

    summary_path = out_root / f"summary_{out_root.name}.json"
    summary_path.write_text(json.dumps(summary) + "\n", encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            str(script),
            "--output-root",
            str(out_root),
            "--dataset-root",
            str(ds_root),
            "--summary",
            str(summary_path),
            "--num-images",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    for model_id in ["cellpose", "stardist", "iaunet"]:
        overlay = out_root / "visualizations" / model_id / "overlay" / "overlay_0000_id1.png"
        mirrored = out_root / model_id / "visualizations" / "overlay" / "overlay_0000_id1.png"
        assert overlay.exists() and overlay.stat().st_size > 0
        assert mirrored.exists() and mirrored.stat().st_size > 0
    gt_overlay = out_root / "visualizations" / "_gt" / "overlay" / "overlay_0000_id1.png"
    assert gt_overlay.exists() and gt_overlay.stat().st_size > 0
