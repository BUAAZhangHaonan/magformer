from __future__ import annotations

from pathlib import Path


def test_register_0831_1k_coco_rgbd_has_depth_file_name() -> None:
    # Workspace layout:
    #   <ws>/magformer/tests/...
    #   <ws>/magformer_datasets/0831_1K/...
    ws_root = Path(__file__).resolve().parents[2]
    dataset_root = ws_root / "magformer_datasets" / "0831_1K"
    if not dataset_root.exists():
        # Allow running unit tests without datasets present.
        return

    from detectron2.data import DatasetCatalog

    from baselines.register_0831_1k_coco_rgbd import (
        DATASET_NAME_TRAIN,
        DATASET_NAME_VAL,
        register_0831_1k_coco_rgbd,
    )

    register_0831_1k_coco_rgbd(str(dataset_root))

    for name in [DATASET_NAME_TRAIN, DATASET_NAME_VAL]:
        ds = DatasetCatalog.get(name)
        assert isinstance(ds, list)
        assert len(ds) > 0
        # Spot-check a few entries.
        for d in ds[:5]:
            assert "file_name" in d
            assert "depth_file_name" in d
            assert Path(d["file_name"]).exists()
            assert Path(d["depth_file_name"]).exists()

