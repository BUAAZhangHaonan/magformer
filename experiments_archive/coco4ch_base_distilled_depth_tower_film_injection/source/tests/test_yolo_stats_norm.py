from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "baselines" / "yolo_stats_norm.py"
    spec = importlib.util.spec_from_file_location("yolo_stats_norm", path)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_normalize_images_uint8_uses_dataset_specific_mean_std() -> None:
    mod = _load_module()
    images = torch.tensor([[[[24.0]], [[48.0]], [[96.0]]]], dtype=torch.float32)
    mean = [24.0, 48.0, 96.0]
    std = [1.0, 1.0, 1.0]

    out = mod.normalize_images_uint8(images, mean, std)

    assert torch.allclose(out, torch.zeros_like(out))
