from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def test_ucn_dataset_converts_scalar_depth_to_xyz_like_geometry(tmp_path: Path) -> None:
    from baselines.run_ucn_ecc import ECCUCNDataset

    root = tmp_path / "0831_1K"
    (root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (root / "depth" / "train").mkdir(parents=True, exist_ok=True)
    (root / "annotations").mkdir(parents=True, exist_ok=True)

    image_path = root / "images" / "train" / "sample.png"
    depth_path = root / "depth" / "train" / "sample.npy"
    ann_path = root / "annotations" / "instances_train.json"

    image = np.zeros((2, 2, 3), dtype=np.uint8)
    cv2.imwrite(str(image_path), image)
    np.save(
        depth_path,
        np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ],
            dtype=np.float32,
        ),
    )
    ann_path.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "sample.png", "width": 2, "height": 2}],
                "annotations": [],
                "categories": [{"id": 1, "name": "component"}],
            }
        ),
        encoding="utf-8",
    )

    dataset = ECCUCNDataset(
        dataset_root=str(root),
        split="train",
        img_size=2,
        train=False,
        pixel_mean_bgr_255=[0.0, 0.0, 0.0],
        depth_clip=(1.0, 4.0),
    )

    sample = dataset[0]
    depth = sample["depth"].permute(1, 2, 0).numpy()
    assert depth.shape == (2, 2, 3)
    assert depth[0, 0, 0] <= 0.0
    assert depth[0, 1, 0] > 0.0
    assert depth[1, 0, 0] < 0.0
    assert depth[1, 1, 0] > 0.0
    assert depth[0, 0, 1] <= 0.0
    assert depth[0, 1, 1] < 0.0
    assert depth[1, 0, 1] > 0.0
    assert depth[1, 1, 1] > 0.0
    assert depth[0, 0, 2] < depth[0, 1, 2] < depth[1, 0, 2] < depth[1, 1, 2]
    assert not np.allclose(depth[:, :, 0], depth[:, :, 2])
    assert not np.allclose(depth[:, :, 1], depth[:, :, 2])


def test_ucn_dataset_prefers_depth_depth_npy_layout_when_present(tmp_path: Path) -> None:
    from baselines.run_ucn_ecc import ECCUCNDataset

    root = tmp_path / "20260318_1K_1566"
    (root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (root / "depth" / "depth_npy" / "train").mkdir(parents=True, exist_ok=True)
    (root / "annotations").mkdir(parents=True, exist_ok=True)

    image_path = root / "images" / "train" / "sample.png"
    depth_path = root / "depth" / "depth_npy" / "train" / "sample.npy"
    ann_path = root / "annotations" / "instances_train.json"

    image = np.zeros((2, 2, 3), dtype=np.uint8)
    cv2.imwrite(str(image_path), image)
    np.save(depth_path, np.full((2, 2), 0.5, dtype=np.float32))
    ann_path.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "sample.png", "width": 2, "height": 2}],
                "annotations": [],
                "categories": [{"id": 1, "name": "component"}],
            }
        ),
        encoding="utf-8",
    )

    dataset = ECCUCNDataset(
        dataset_root=str(root),
        split="train",
        img_size=2,
        train=False,
        pixel_mean_bgr_255=[0.0, 0.0, 0.0],
        depth_clip=(0.0, 1.0),
    )

    sample = dataset[0]
    assert sample["depth"].shape == (3, 2, 2)
