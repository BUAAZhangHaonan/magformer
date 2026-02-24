from __future__ import annotations

from pathlib import Path


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

    root_0909 = ws_root / "magformer_datasets" / "0909_512_0.12K"
    if root_0909.exists():
        train_name, val_name = register_ecc_coco_rgbd("0909", str(root_0909))
        sample = DatasetCatalog.get(train_name)[0]
        assert Path(sample["file_name"]).exists()
        assert Path(sample["depth_file_name"]).exists()
        assert len(DatasetCatalog.get(val_name)) > 0

