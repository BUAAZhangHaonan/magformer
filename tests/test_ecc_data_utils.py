from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


def _write_min_ecc_rgb_dataset(root: Path) -> None:
    (root / "annotations").mkdir(parents=True, exist_ok=True)
    (root / "images" / "train").mkdir(parents=True, exist_ok=True)

    image_name = "train_000001.png"
    Image.new("RGB", (8, 8), color=(12, 34, 56)).save(root / "images" / "train" / image_name)
    payload = {
        "images": [{"id": 1, "file_name": image_name, "width": 8, "height": 8}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "segmentation": [[1, 1, 4, 1, 4, 4, 1, 4]],
                "area": 9,
                "bbox": [1, 1, 3, 3],
                "iscrowd": 0,
            }
        ],
        "categories": [{"id": 1, "name": "component"}],
    }
    (root / "annotations" / "instances_train.json").write_text(json.dumps(payload), encoding="utf-8")


def test_load_ecc_coco_rgb_records_and_image(tmp_path: Path) -> None:
    from baselines.ecc_data_utils import load_ecc_coco_rgb_image, load_ecc_coco_rgb_records

    dataset_root = tmp_path / "ecc"
    _write_min_ecc_rgb_dataset(dataset_root)

    records = load_ecc_coco_rgb_records(dataset_root, "train")
    assert len(records) == 1
    assert records[0]["image_id"] == 1
    assert Path(records[0]["image_path"]).exists()

    image = load_ecc_coco_rgb_image(records[0]["image_path"], image_size=4)
    assert image.shape == (4, 4, 3)
    assert image.dtype == np.uint8

