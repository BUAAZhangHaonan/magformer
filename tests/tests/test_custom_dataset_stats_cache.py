from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


def _make_min_dataset(root: Path) -> None:
    (root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (root / "depth" / "depth_npy" / "train").mkdir(parents=True, exist_ok=True)
    (root / "annotations").mkdir(parents=True, exist_ok=True)

    image_a = np.zeros((4, 4, 3), dtype=np.uint8)
    image_a[..., 0] = 10
    image_a[..., 1] = 20
    image_a[..., 2] = 30
    Image.fromarray(image_a, mode="RGB").save(root / "images" / "train" / "a.png")

    image_b = np.zeros((4, 4, 3), dtype=np.uint8)
    image_b[..., 0] = 40
    image_b[..., 1] = 50
    image_b[..., 2] = 60
    Image.fromarray(image_b, mode="RGB").save(root / "images" / "train" / "b.png")

    np.save(root / "depth" / "depth_npy" / "train" / "a.npy", np.full((4, 4), 0.25, dtype=np.float32))
    np.save(root / "depth" / "depth_npy" / "train" / "b.npy", np.full((4, 4), 0.75, dtype=np.float32))

    annotations = {
        "images": [
            {"id": 1, "file_name": "a.png", "width": 4, "height": 4},
            {"id": 2, "file_name": "b.png", "width": 4, "height": 4},
        ],
        "annotations": [],
        "categories": [{"id": 1, "name": "component"}],
    }
    (root / "annotations" / "instances_train.json").write_text(json.dumps(annotations), encoding="utf-8")
