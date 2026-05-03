from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "baselines" / "run_unet_instance_ecc.py"
    spec = importlib.util.spec_from_file_location("unet_runner_norm", path)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_dataset(root: Path) -> None:
    (root / "annotations").mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "depth" / "depth_npy" / split).mkdir(parents=True, exist_ok=True)
        image_name = f"{split}_000001.png"
        Image.new("RGB", (16, 16), color=(24, 48, 96)).save(root / "images" / split / image_name)
        np.save(root / "depth" / "depth_npy" / split / f"{split}_000001.npy", np.full((16, 16), 0.5, dtype=np.float32))
        payload = {
            "images": [{"id": 1, "file_name": image_name, "width": 16, "height": 16}],
            "annotations": [],
            "categories": [{"id": 1, "name": "component"}],
        }
        (root / "annotations" / f"instances_{split}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_unet_dataset_uses_dataset_specific_rgbd_normalization(tmp_path: Path) -> None:
    mod = _load_module()
    dataset_root = tmp_path / "20260318_1K_1566"
    _write_dataset(dataset_root)

    dataset = mod.ECCUnetDataset(str(dataset_root), "train", 16, False, "unet_boundary_inst")
    sample = dataset[0]

    assert float(sample["image"].abs().max()) == 0.0
    assert float(sample["depth"].abs().max()) == 0.0
