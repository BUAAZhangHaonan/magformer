from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


def _write_min_dataset(root: Path, *, with_depth: bool) -> None:
    (root / "annotations").mkdir(parents=True, exist_ok=True)
    (root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (root / "images" / "val").mkdir(parents=True, exist_ok=True)
    if with_depth:
        (root / "depth" / "depth_npy" / "train").mkdir(parents=True, exist_ok=True)
        (root / "depth" / "depth_npy" / "val").mkdir(parents=True, exist_ok=True)

    for split in ("train", "val"):
        image_name = f"{split}_000001.png"
        Image.new("RGB", (8, 8), color=(12, 34, 56)).save(root / "images" / split / image_name)
        if with_depth:
            np.save(root / "depth" / "depth_npy" / split / f"{split}_000001.npy", np.full((8, 8), 0.5, dtype=np.float32))
        payload = {
            "images": [{"id": 1, "file_name": image_name, "width": 8, "height": 8}],
            "annotations": [],
            "categories": [{"id": 1, "name": "component"}],
        }
        (root / "annotations" / f"instances_{split}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_register_ecc_coco_0831_and_0909() -> None:
    ws_root = Path(__file__).resolve().parents[2]

    from detectron2.data import DatasetCatalog

    from baselines.ecc_datasets import register_ecc_coco

    root_0831 = ws_root / "magformer_datasets" / "0831_1K"
    if root_0831.exists():
        train_name, val_name = register_ecc_coco("0831", str(root_0831))
        assert len(DatasetCatalog.get(train_name)) > 0
        assert len(DatasetCatalog.get(val_name)) > 0

    root_0909 = ws_root / "magformer_datasets" / "0909_512_0.12K"
    if root_0909.exists():
        train_name, val_name = register_ecc_coco("0909", str(root_0909))
        assert len(DatasetCatalog.get(train_name)) > 0
        assert len(DatasetCatalog.get(val_name)) > 0


def test_register_ecc_coco_rgbd_0831_and_0909() -> None:
    ws_root = Path(__file__).resolve().parents[2]

    from detectron2.data import DatasetCatalog

    from baselines.ecc_datasets import register_ecc_coco_rgbd

    root_0831 = ws_root / "magformer_datasets" / "0831_1K"
    if root_0831.exists():
        train_name, val_name = register_ecc_coco_rgbd("0831", str(root_0831))
        sample = DatasetCatalog.get(train_name)[0]
        assert Path(sample["file_name"]).exists()
        assert Path(sample["depth_file_name"]).exists()
        assert len(DatasetCatalog.get(val_name)) > 0


def test_register_ecc_coco_supports_custom_dataset_root(tmp_path: Path) -> None:
    from detectron2.data import DatasetCatalog

    from baselines.ecc_datasets import register_ecc_coco

    dataset_root = tmp_path / "20260318_1K_1566"
    _write_min_dataset(dataset_root, with_depth=False)

    train_name, val_name = register_ecc_coco("20260318_1K_1566", str(dataset_root))
    assert train_name == "ecc20260318_1k_1566_train"
    assert val_name == "ecc20260318_1k_1566_val"
    assert len(DatasetCatalog.get(train_name)) == 1
    assert len(DatasetCatalog.get(val_name)) == 1


def test_register_ecc_coco_rgbd_supports_custom_dataset_root(tmp_path: Path) -> None:
    from detectron2.data import DatasetCatalog

    from baselines.ecc_datasets import register_ecc_coco_rgbd

    dataset_root = tmp_path / "20260318_1K_1566"
    _write_min_dataset(dataset_root, with_depth=True)

    train_name, val_name = register_ecc_coco_rgbd("20260318_1K_1566", str(dataset_root))
    assert train_name == "ecc20260318_1k_1566_rgbd_train"
    assert val_name == "ecc20260318_1k_1566_rgbd_val"
    sample = DatasetCatalog.get(train_name)[0]
    assert Path(sample["file_name"]).exists()
    assert Path(sample["depth_file_name"]).exists()
    assert len(DatasetCatalog.get(val_name)) == 1
