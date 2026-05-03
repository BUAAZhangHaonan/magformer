from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "baselines" / "unet_instance_models.py"
    spec = importlib.util.spec_from_file_location("unet_instance_models", path)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_build_instance_model_returns_expected_head_shapes() -> None:
    mod = _load_module()
    model = mod.build_instance_model("unet_boundary_inst", in_channels=3, base_channels=8)
    fg_logits, aux_logits = model(torch.randn(2, 3, 64, 64))

    assert fg_logits.shape == (2, 1, 64, 64)
    assert aux_logits.shape == (2, 1, 64, 64)


def test_boundary_postprocess_splits_two_components() -> None:
    mod = _load_module()
    fg_logits = np.full((64, 64), -10.0, dtype=np.float32)
    boundary_logits = np.full((64, 64), -10.0, dtype=np.float32)
    fg_logits[8:24, 8:24] = 10.0
    fg_logits[40:56, 40:56] = 10.0

    masks = mod.instances_from_boundary_logits(
        fg_logits=fg_logits,
        boundary_logits=boundary_logits,
        threshold=0.5,
        min_area=10,
    )

    assert len(masks) == 2
    assert all(mask.dtype == np.uint8 for mask in masks)


def test_distance_postprocess_can_seed_multiple_instances() -> None:
    mod = _load_module()
    fg_logits = np.full((64, 64), -10.0, dtype=np.float32)
    dist_logits = np.zeros((64, 64), dtype=np.float32)
    fg_logits[8:56, 8:56] = 10.0
    dist_logits[20, 20] = 5.0
    dist_logits[44, 44] = 5.0

    masks = mod.instances_from_distance_logits(
        fg_logits=fg_logits,
        distance_logits=dist_logits,
        threshold=0.5,
        min_area=10,
    )

    assert len(masks) >= 2


def test_semantic_postprocess_uses_connected_components() -> None:
    mod = _load_module()
    fg_logits = np.full((64, 64), -10.0, dtype=np.float32)
    fg_logits[8:24, 8:24] = 10.0
    fg_logits[40:56, 40:56] = 10.0

    masks = mod.instances_from_semantic_logits(
        fg_logits=fg_logits,
        threshold=0.5,
        min_area=10,
    )

    assert len(masks) == 2


def test_build_instance_model_supports_modern_smp_variant() -> None:
    mod = _load_module()
    model = mod.build_instance_model("smp_unet_mobilenetv2_boundary_inst", in_channels=3, base_channels=8)
    fg_logits, aux_logits = model(torch.randn(2, 3, 64, 64))

    assert fg_logits.shape == (2, 1, 64, 64)
    assert aux_logits.shape == (2, 1, 64, 64)
