import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from pycocotools import mask as coco_mask
from torch.utils.data import DataLoader

from magformer.data.dataset import CocoRgbdDataset
from tools.build_coco_loader_cache import build_coco_loader_cache


def _write_tiny_coco_dataset(root: Path) -> Path:
    image_dir = root / "images" / "train"
    depth_dir = root / "depth" / "depth_npy" / "train"
    ann_dir = root / "annotations"
    image_dir.mkdir(parents=True)
    depth_dir.mkdir(parents=True)
    ann_dir.mkdir(parents=True)

    for name, value in [("img_b.png", 40), ("img_a.png", 120)]:
        image = np.full((6, 7, 3), value, dtype=np.uint8)
        assert cv2.imwrite(str(image_dir / name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        np.save(depth_dir / f"{Path(name).stem}.npy", np.full((6, 7), value / 255, dtype=np.float32))

    rle_mask = np.zeros((6, 7), dtype=np.uint8)
    rle_mask[1:5, 2:6] = 1
    rle = coco_mask.encode(np.asfortranarray(rle_mask))
    rle["counts"] = rle["counts"].decode("ascii")

    coco = {
        "info": {"description": "tiny cache parity fixture"},
        "licenses": [{"id": 1, "name": "fixture"}],
        "categories": [{"id": 5, "name": "component", "supercategory": "part"}],
        "images": [
            {"id": 42, "file_name": "img_b.png", "height": 6, "width": 7, "extra": "keep"},
            {"id": 7, "file_name": "img_a.png", "height": 6, "width": 7},
        ],
        "annotations": [
            {
                "id": 100,
                "image_id": 42,
                "category_id": 5,
                "bbox": [1, 1, 3, 3],
                "area": 9,
                "segmentation": [[1, 1, 4, 1, 4, 4, 1, 4]],
                "iscrowd": 0,
                "extra_ann_field": {"kept": True},
            },
            {
                "id": 101,
                "image_id": 42,
                "category_id": 5,
                "bbox": [0, 0, 1, 1],
                "area": 1,
                "segmentation": [[0, 0, 1, 0, 1, 1, 0, 1]],
                "iscrowd": 1,
            },
            {
                "id": 200,
                "image_id": 7,
                "category_id": 5,
                "bbox": [2, 1, 4, 4],
                "area": 16,
                "segmentation": rle,
                "iscrowd": 0,
            },
        ],
    }
    ann_path = ann_dir / "instances_train.json"
    ann_path.write_text(json.dumps(coco), encoding="utf-8")
    return ann_path


def _dataset(root: Path, ann_file: str | Path) -> CocoRgbdDataset:
    return CocoRgbdDataset(
        dataset_root=str(root),
        ann_file=str(ann_file),
        split="train",
        transform=None,
        is_train=True,
        has_annotations=True,
    )


def _assert_sample_equal(left, right) -> None:
    assert left["image_id"] == right["image_id"]
    assert left["height"] == right["height"]
    assert left["width"] == right["width"]
    np.testing.assert_array_equal(left["labels"], right["labels"])
    np.testing.assert_allclose(left["boxes"], right["boxes"])
    assert left["masks"].shape == right["masks"].shape
    np.testing.assert_array_equal(left["masks"], right["masks"])


def test_cache_backend_matches_json_backend_for_tiny_coco(tmp_path):
    ann_path = _write_tiny_coco_dataset(tmp_path)
    cache_path = tmp_path / "annotations" / "instances_train.sqlite"

    build_coco_loader_cache(ann_path, cache_path)

    json_dataset = _dataset(tmp_path, "annotations/instances_train.json")
    cache_dataset = _dataset(tmp_path, "annotations/instances_train.sqlite")

    assert cache_dataset.image_ids == [7, 42]
    assert cache_dataset.class_names == json_dataset.class_names == ["component"]
    assert len(cache_dataset) == len(json_dataset) == 2

    for idx in range(len(json_dataset)):
        _assert_sample_equal(json_dataset[idx], cache_dataset[idx])

    assert cache_dataset.coco.loadImgs(42)[0]["extra"] == "keep"
    ann = cache_dataset.coco.loadAnns([100])[0]
    assert ann["extra_ann_field"] == {"kept": True}


def test_cache_backend_fails_loud_when_source_hash_changes(tmp_path):
    ann_path = _write_tiny_coco_dataset(tmp_path)
    cache_path = tmp_path / "annotations" / "instances_train.sqlite"
    build_coco_loader_cache(ann_path, cache_path)

    payload = json.loads(ann_path.read_text(encoding="utf-8"))
    payload["images"][0]["file_name"] = "changed.png"
    ann_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="source hash mismatch"):
        _dataset(tmp_path, "annotations/instances_train.sqlite")


@pytest.mark.parametrize("num_workers", [0, 2])
def test_cache_backend_is_readable_from_dataloader_workers(tmp_path, num_workers):
    ann_path = _write_tiny_coco_dataset(tmp_path)
    cache_path = tmp_path / "annotations" / "instances_train.sqlite"
    build_coco_loader_cache(ann_path, cache_path)

    dataset = _dataset(tmp_path, "annotations/instances_train.sqlite")
    loader = DataLoader(dataset, batch_size=1, num_workers=num_workers, shuffle=False)

    image_ids = []
    for batch in loader:
        image_ids.append(int(batch["image_id"].item()))

    assert image_ids == [7, 42]
