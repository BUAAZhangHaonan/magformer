from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_select_timm_out_indices_maps_resnet_style_reductions() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    depth_mod = _load_module(
        "light_depth_backbone",
        repo_root / "magformer" / "models" / "common" / "backbones" / "depth.py",
    )
    reductions = [2, 4, 8, 16, 32]
    assert depth_mod.select_timm_out_indices(reductions, ["res2", "res3", "res4", "res5"]) == (1, 2, 3, 4)
    assert depth_mod.select_timm_out_indices(reductions, ["res3"]) == (2,)


def test_select_timm_out_indices_maps_convnext_style_reductions() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    depth_mod = _load_module(
        "light_depth_backbone",
        repo_root / "magformer" / "models" / "common" / "backbones" / "depth.py",
    )
    reductions = [4, 8, 16, 32]
    assert depth_mod.select_timm_out_indices(reductions, ["res2", "res3", "res4", "res5"]) == (0, 1, 2, 3)
    assert depth_mod.select_timm_out_indices(reductions, ["res3", "res5"]) == (1, 3)
