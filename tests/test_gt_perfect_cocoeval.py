from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

def test_gt_as_predictions_yields_perfect_ap(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    gen_script = repo_root / "scripts" / "analysis" / "generate_coco_instances_results_from_gt.py"
    assert gen_script.exists()

    # Minimal COCO dataset (1 image, 1 annotation)
    ds_root = tmp_path / "ds"
    (ds_root / "images" / "val").mkdir(parents=True)
    (ds_root / "annotations").mkdir(parents=True)

    import numpy as np
    import cv2

    img = np.zeros((32, 32, 3), dtype=np.uint8)
    cv2.imwrite(str(ds_root / "images" / "val" / "0001.png"), img)

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
    ann_path = ds_root / "annotations" / "instances_val.json"
    ann_path.write_text(json.dumps(ann) + "\n", encoding="utf-8")

    results_json = tmp_path / "results.json"
    subprocess.run(
        [
            sys.executable,
            str(gen_script),
            "--dataset-root",
            str(ds_root),
            "--ann-file",
            str(ann_path),
            "--output-json",
            str(results_json),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert results_json.exists()

    # Evaluate with the same helper as postprocess_cocoeval.py
    from baselines.coco_eval_results import evaluate_coco_results

    metrics = evaluate_coco_results(ann_file=ann_path, results_json=results_json, iteration=-1)
    assert metrics["segm/AP"] == pytest.approx(100.0, abs=1e-6)
    assert metrics["bbox/AP"] == pytest.approx(100.0, abs=1e-6)
