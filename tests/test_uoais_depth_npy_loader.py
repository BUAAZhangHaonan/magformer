from __future__ import annotations

from pathlib import Path


def _add_uoais_to_syspath() -> None:
    # Workspace layout:
    #   <ws>/magformer/tests/...
    #   <ws>/magformer/baselines/icra_2026/uoais/...
    repo_root = Path(__file__).resolve().parents[1]
    uoais_root = repo_root / "baselines" / "icra_2026" / "uoais"

    import sys

    sys.path.insert(0, str(uoais_root))


def test_uoais_dataset_mapper_supports_npy_depth_rgbd_late_fusion() -> None:
    # Skip if datasets are not present locally.
    ws_root = Path(__file__).resolve().parents[2]
    dataset_root = ws_root / "magformer_datasets" / "0831_1K"
    if not dataset_root.exists():
        return

    from detectron2.data import DatasetCatalog

    from baselines.register_0831_1k_coco_rgbd import (
        DATASET_NAME_TRAIN,
        DATASET_NAME_VAL,
        register_0831_1k_coco_rgbd,
    )

    register_0831_1k_coco_rgbd(str(dataset_root))
    ds = DatasetCatalog.get(DATASET_NAME_TRAIN)
    assert isinstance(ds, list) and len(ds) > 0

    sample = ds[0]
    assert Path(sample["file_name"]).exists()
    assert Path(sample["depth_file_name"]).suffix.lower() == ".npy"
    assert Path(sample["depth_file_name"]).exists()

    _add_uoais_to_syspath()

    import torch

    from adet.config import get_cfg
    from adet.data.dataset_mapper import DatasetMapperWithBasis

    cfg = get_cfg()
    cfg.INPUT.AMODAL = False
    cfg.INPUT.DEPTH = True
    cfg.INPUT.DEPTH_ONLY = False
    cfg.MODEL.RGBD_FUSION = "late"
    cfg.INPUT.IMG_SIZE = (512, 512)  # (width, height) in this codebase
    cfg.INPUT.DEPTH_RANGE = (0.0, 1.0)
    cfg.INPUT.COLOR_AUGMENTATION = False
    cfg.INPUT.PERLIN_DISTORTION = False
    cfg.INPUT.CROP_RATIO = 0.5
    cfg.DATASETS.TRAIN = (DATASET_NAME_TRAIN,)
    cfg.DATASETS.TEST = (DATASET_NAME_VAL,)

    mapper = DatasetMapperWithBasis(cfg, is_train=False)
    out = mapper(sample)

    assert "image" in out
    assert isinstance(out["image"], torch.Tensor)
    # Late fusion concatenates RGB(3) + depth(3) => 6 channels.
    assert tuple(out["image"].shape) == (6, 512, 512)
