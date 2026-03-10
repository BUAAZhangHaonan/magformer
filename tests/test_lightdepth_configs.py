from __future__ import annotations

from pathlib import Path

import yaml


def test_lightdepth_configs_use_res3_only_and_disable_dpe() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    configs = [
        repo_root / "configs" / "magformer_0831_1k_20ep_1024_lightdepth_mobilenetv3.yaml",
        repo_root / "configs" / "magformer_0831_1k_20ep_1024_lightdepth_resnet18.yaml",
        repo_root / "configs" / "magformer_0831_1k_20ep_1024_lightdepth_film.yaml",
    ]

    for path in configs:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        magformer_cfg = payload["model"]["magformer"]
        assert magformer_cfg["depth_mode"] == "light"
        assert magformer_cfg["modality_fusion"]["fuse_scales"] == ["res3"]
        assert magformer_cfg["modality_fusion"]["prior"]["compute_on"] == "res3"
        assert magformer_cfg["dpe"]["enabled"] is False
