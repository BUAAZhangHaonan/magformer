from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_visualize_suite_also_writes_into_model_local_visualizations(tmp_path: Path) -> None:
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
    (out_root / "magformer").mkdir(parents=True)
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

    summary = {
        "experiment": "dummy",
        "output_root": str(out_root),
        "magformer": {"status": "ok", "artifacts": {"coco_instances_results": str(out_root / "magformer" / "coco_instances_results.json")}},
    }
    summary_path = out_root / "summary_dummy.json"
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
    assert (out_root / "magformer" / "visualizations" / "overlay" / "overlay_0000_id1.png").exists()
