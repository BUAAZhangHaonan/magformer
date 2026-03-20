from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image


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


def test_render_runtime_config_injects_dataset_specific_stats_for_20260318(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    dataset_root = tmp_path / "20260318_1K_1566"
    _write_dataset(dataset_root)

    out_config = tmp_path / "runtime.yaml"
    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "analysis" / "render_magformer_runtime_config.py"),
            "--base-config",
            str(repo_root / "configs" / "magformer_0831_1k_20ep_1024_nodpth_ref.yaml"),
            "--out-config",
            str(out_config),
            "--output-dir",
            str(tmp_path / "out"),
            "--run-name",
            "test_run",
            "--base-lr",
            "0.00005",
            "--max-iter",
            "10",
            "--steps",
            "8,9",
            "--warmup-iters",
            "1",
            "--ims-per-batch",
            "1",
            "--eval-period",
            "1",
            "--checkpoint-period",
            "1",
            "--dataset-root",
            str(dataset_root),
        ],
        check=True,
        cwd=repo_root,
        text=True,
    )

    payload = yaml.safe_load(out_config.read_text(encoding="utf-8"))
    assert payload["data"]["dataset_root"] == str(dataset_root.resolve())
    assert payload["model"]["magformer"]["pixel_mean"] == [24.0, 48.0, 96.0]
    assert payload["model"]["magformer"]["pixel_std"] == [0.0, 0.0, 0.0]
    assert payload["data"]["depth"]["clip_min"] == 0.5
    assert payload["data"]["depth"]["clip_max"] == 0.5
