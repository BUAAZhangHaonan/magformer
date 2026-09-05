from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_convert_coco_to_yolo_seg_handles_relative_dataset_root_symlinks(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "baselines" / "ultralytics_tools" / "convert_coco_to_yolo_seg.py"
    assert script.exists()

    dataset_root = tmp_path / "ecc"
    (dataset_root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (dataset_root / "annotations").mkdir(parents=True, exist_ok=True)

    # The converter does not decode images, only symlinks/copies them.
    (dataset_root / "images" / "train" / "a.png").write_bytes(b"not-a-real-png")

    coco = {
        "images": [{"id": 1, "file_name": "a.png", "width": 10, "height": 10}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "iscrowd": 0,
                "segmentation": [[0, 0, 9, 0, 9, 9, 0, 9]],
            }
        ],
        "categories": [{"id": 1, "name": "obj"}],
    }
    (dataset_root / "annotations" / "instances_train.json").write_text(
        json.dumps(coco), encoding="utf-8"
    )

    output_root = tmp_path / "out"
    rel_dataset_root = os.path.relpath(dataset_root, repo_root)

    subprocess.run(
        [
            sys.executable,
            str(script),
            "--dataset-root",
            rel_dataset_root,
            "--output-root",
            str(output_root),
            "--splits",
            "train",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    out_img = output_root / "images" / "train" / "a.png"
    assert out_img.exists()
