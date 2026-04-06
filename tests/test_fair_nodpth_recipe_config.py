from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _flatten(payload: Any, prefix: tuple[str, ...] = ()) -> dict[tuple[str, ...], Any]:
    if isinstance(payload, dict):
        flattened: dict[tuple[str, ...], Any] = {}
        for key, value in payload.items():
            flattened.update(_flatten(value, prefix + (str(key),)))
        return flattened
    return {prefix: payload}


def test_fair_nodpth_config_matches_depth_recipe_except_depth_switches() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    depth_cfg_path = (
        repo_root
        / "output"
        / "experiments"
        / "20260318_1k_1566_20ep_1024_full19"
        / "magformer_depthnorm_on"
        / "magformer_runtime_config.yaml"
    )
    fair_cfg_path = repo_root / "configs" / "magformer_0831_1k_20ep_1024_nodpth_ref_fair.yaml"

    assert depth_cfg_path.exists()
    depth_cfg = yaml.safe_load(depth_cfg_path.read_text(encoding="utf-8"))
    fair_cfg = yaml.safe_load(fair_cfg_path.read_text(encoding="utf-8"))

    assert fair_cfg["model"]["finetune_weights"] == depth_cfg["model"]["finetune_weights"]
    assert fair_cfg["solver"] == depth_cfg["solver"]
    assert fair_cfg["data"] == depth_cfg["data"]
    assert fair_cfg["model"]["magformer"]["rgb_backbone"] == depth_cfg["model"]["magformer"]["rgb_backbone"]
    assert fair_cfg["model"]["magformer"]["swin"] == depth_cfg["model"]["magformer"]["swin"]
    assert fair_cfg["model"]["magformer"]["mask_former"] == depth_cfg["model"]["magformer"]["mask_former"]
    assert fair_cfg["model"]["magformer"]["sem_seg_head"] == depth_cfg["model"]["magformer"]["sem_seg_head"]
    assert fair_cfg["model"]["magformer"]["pixel_mean"] == depth_cfg["model"]["magformer"]["pixel_mean"]
    assert fair_cfg["model"]["magformer"]["pixel_std"] == depth_cfg["model"]["magformer"]["pixel_std"]

    depth_flat = _flatten(depth_cfg)
    fair_flat = _flatten(fair_cfg)
    diff_keys = {
        key
        for key in sorted(set(depth_flat) | set(fair_flat))
        if depth_flat.get(key) != fair_flat.get(key)
    }

    assert diff_keys <= {
        ("dpe_enabled",),
        ("name",),
        ("model", "magformer", "depth_backbone", "enabled"),
        ("model", "magformer", "modality_fusion", "enabled"),
        ("runtime", "ddp_enabled"),
        ("runtime", "find_unused_parameters"),
        ("runtime", "gpus"),
        ("runtime", "logger", "log_dir"),
        ("runtime", "logger", "run_name"),
        ("runtime", "output_dir"),
    }
    assert {
        ("dpe_enabled",),
        ("model", "magformer", "depth_backbone", "enabled"),
        ("model", "magformer", "modality_fusion", "enabled"),
    } <= diff_keys
    assert fair_cfg["model"]["magformer"]["depth_backbone"]["enabled"] is False
    assert fair_cfg["model"]["magformer"]["modality_fusion"]["enabled"] is False
    assert fair_cfg["dpe_enabled"] is False
