from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
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
    assert "annotation_targets" in records[0]

    light_records = load_ecc_coco_rgb_records(dataset_root, "train", include_targets=False)
    assert "annotations" in light_records[0]
    assert "annotation_targets" not in light_records[0]

    image = load_ecc_coco_rgb_image(records[0]["image_path"], image_size=4)
    assert image.shape == (4, 4, 3)
    assert image.dtype == np.uint8


def test_load_stardist_ecc_split_honors_max_images_without_rescanning_annotations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from baselines.stardist_instance_utils import load_stardist_ecc_split

    dataset_root = tmp_path / "ecc"
    (dataset_root / "annotations").mkdir(parents=True, exist_ok=True)
    (dataset_root / "images" / "train").mkdir(parents=True, exist_ok=True)
    for idx in range(3):
        image_name = f"train_{idx:06d}.png"
        Image.new("RGB", (8, 8), color=(12, 34, 56)).save(dataset_root / "images" / "train" / image_name)
    (dataset_root / "annotations" / "instances_train.json").write_text("{}", encoding="utf-8")

    class CountingAnnotations:
        def __init__(self, annotations: list[dict[str, object]]) -> None:
            self.annotations = annotations
            self.iterations = 0

        def __iter__(self):
            self.iterations += 1
            if self.iterations > 1:
                raise AssertionError("annotations were rescanned instead of being indexed once")
            return iter(self.annotations)

    annotations = CountingAnnotations(
        [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "segmentation": [[1, 1, 4, 1, 4, 4, 1, 4]],
                "area": 9,
                "bbox": [1, 1, 3, 3],
                "iscrowd": 0,
            },
            {
                "id": 2,
                "image_id": 2,
                "category_id": 1,
                "segmentation": [[2, 2, 5, 2, 5, 5, 2, 5]],
                "area": 9,
                "bbox": [2, 2, 3, 3],
                "iscrowd": 0,
            },
            {
                "id": 3,
                "image_id": 3,
                "category_id": 1,
                "segmentation": [[3, 3, 6, 3, 6, 6, 3, 6]],
                "area": 9,
                "bbox": [3, 3, 3, 3],
                "iscrowd": 0,
            },
        ]
    )

    def fake_json_loads(_text: str):
        return {
            "images": [
                {"id": 1, "file_name": "train_000000.png", "width": 8, "height": 8},
                {"id": 2, "file_name": "train_000001.png", "width": 8, "height": 8},
                {"id": 3, "file_name": "train_000002.png", "width": 8, "height": 8},
            ],
            "annotations": annotations,
        }

    monkeypatch.setattr("baselines.ecc_data_utils.json.loads", fake_json_loads)

    images, labels, records = load_stardist_ecc_split(dataset_root, "train", image_size=4, max_images=1)

    assert len(images) == 1
    assert len(labels) == 1
    assert len(records) == 1
    assert records[0]["image_id"] == 1
    assert annotations.iterations == 1


def test_load_stardist_ecc_split_uses_light_records_and_returns_light_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from baselines import stardist_instance_utils as utils

    dataset_root = tmp_path / "ecc"
    (dataset_root / "images" / "train").mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=(12, 34, 56)).save(dataset_root / "images" / "train" / "train.png")

    calls: list[bool] = []

    def fake_records(dataset_root_arg, split, max_images=None, include_targets=True):
        calls.append(include_targets)
        return [
            {
                "image_id": 1,
                "file_name": "train.png",
                "image_path": str(dataset_root / "images" / "train" / "train.png"),
                "height": 8,
                "width": 8,
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
            }
        ]

    monkeypatch.setattr(utils, "load_ecc_coco_rgb_records", fake_records)

    images, labels, records = utils.load_stardist_ecc_split(dataset_root, "train", image_size=4)

    assert calls == [False]
    assert len(images) == 1
    assert len(labels) == 1
    assert "annotations" not in records[0]
    assert "annotation_targets" not in records[0]
    assert records[0]["image_id"] == 1
    assert labels[0].dtype == np.uint16
