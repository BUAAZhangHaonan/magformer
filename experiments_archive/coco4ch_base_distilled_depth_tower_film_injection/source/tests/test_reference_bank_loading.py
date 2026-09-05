from __future__ import annotations

import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "baselines" / "run_unet_instance_ecc.py"
    spec = importlib.util.spec_from_file_location("unet_runner", path)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_load_reference_bank_requires_rgb_depth_mask_dirs(tmp_path: Path) -> None:
    mod = _load_module()
    ref_root = tmp_path / "refs"
    (ref_root / "rgb").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        mod.load_reference_bank(str(ref_root), image_size=64)


def test_load_reference_bank_sorts_views_and_builds_shape_stats(tmp_path: Path) -> None:
    mod = _load_module()
    ref_root = tmp_path / "refs"
    for name in ["rgb", "depth", "mask"]:
        (ref_root / name).mkdir(parents=True)

    for stem in ["b", "a"]:
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        img[8:24, 8:24] = (20, 40, 60)
        cv2.imwrite(str(ref_root / "rgb" / f"{stem}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        np.save(ref_root / "depth" / f"{stem}.npy", np.full((32, 32), 0.95, dtype=np.float32))
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[8:24, 8:24] = 255
        cv2.imwrite(str(ref_root / "mask" / f"{stem}.png"), mask)

    bank = mod.load_reference_bank(str(ref_root), image_size=64)

    assert bank["view_ids"] == ["a", "b"]
    assert tuple(bank["images"].shape) == (2, 3, 64, 64)
    assert tuple(bank["depths"].shape) == (2, 1, 64, 64)
    assert tuple(bank["masks"].shape) == (2, 1, 64, 64)
    assert bank["shape_stats"]["mean_area_ratio"] > 0.0

