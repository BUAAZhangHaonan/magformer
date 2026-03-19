from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

from tests.test_custom_dataset_stats_cache import _make_min_dataset


def test_render_magformer_runtime_config_applies_dataset_stats(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "analysis" / "render_magformer_runtime_config.py"
    dataset_root = tmp_path / "20260318_1K_1566"
    _make_min_dataset(dataset_root)

    out_config = tmp_path / "runtime.yaml"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--base-config",
            str(repo_root / "configs" / "magformer_0831_1k_20ep_1024_depthnorm_on.yaml"),
            "--out-config",
            str(out_config),
            "--output-dir",
            str(tmp_path / "out"),
            "--run-name",
            "test_runtime",
            "--base-lr",
            "0.0001",
            "--max-iter",
            "20",
            "--steps",
            "16,18",
            "--warmup-iters",
            "1",
            "--ims-per-batch",
            "2",
            "--eval-period",
            "1",
            "--checkpoint-period",
            "1",
            "--dataset-root",
            str(dataset_root),
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    cfg = yaml.safe_load(out_config.read_text(encoding="utf-8"))
    manifest_path = tmp_path / "out" / ".."
    del manifest_path

    stats_cache = repo_root / "output" / "cache" / "dataset_stats" / "20260318_1k_1566"
    manifest = json.loads((stats_cache / "manifest.json").read_text(encoding="utf-8"))
    rgb_stats = manifest["rgb_stats"]
    depth_stats = manifest["depth_stats"]

    assert cfg["model"]["magformer"]["pixel_mean"] == rgb_stats["mean_rgb"]
    assert cfg["model"]["magformer"]["pixel_std"] == rgb_stats["std_rgb"]
    assert cfg["data"]["depth"]["clip_min"] == depth_stats["p1"]
    assert cfg["data"]["depth"]["clip_max"] == depth_stats["p99"]
