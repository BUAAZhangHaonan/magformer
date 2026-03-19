from __future__ import annotations

import json
import subprocess
import sys
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


def test_ensure_dataset_stats_creates_cache_for_arbitrary_dataset_root(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "analysis" / "ensure_dataset_stats.py"
    dataset_root = tmp_path / "custom_ds"
    _make_min_dataset(dataset_root)

    res = subprocess.run(
        [sys.executable, str(script), "--dataset-root", str(dataset_root)],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(res.stdout)
    rgb_stats_path = Path(payload["rgb_stats_path"])
    depth_stats_path = Path(payload["depth_stats_path"])

    assert rgb_stats_path.exists()
    assert depth_stats_path.exists()
    assert rgb_stats_path.parent == depth_stats_path.parent

    rgb_stats = json.loads(rgb_stats_path.read_text(encoding="utf-8"))
    depth_stats = json.loads(depth_stats_path.read_text(encoding="utf-8"))
    assert rgb_stats["num_images"] == 2
    assert len(rgb_stats["mean_bgr"]) == 3
    assert depth_stats["p1"] <= depth_stats["p99"]


def test_ecc_common_can_read_stats_for_arbitrary_dataset_root(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    dataset_root = tmp_path / "custom_ds"
    _make_min_dataset(dataset_root)

    shell = f"""
set -euo pipefail
REPO_ROOT='{repo_root}'
PROJECT_ROOT='{repo_root.parent}'
source '{repo_root}/scripts/experiments/ecc_common.sh'
ecc_read_rgb_stats_bgr_for_dataset_root '{dataset_root}'
echo '---'
ecc_read_depth_clip_for_dataset_root '{dataset_root}'
"""
    res = subprocess.run(
        ["bash", "-lc", shell],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    rgb_line, _, depth_line = [line.strip() for line in res.stdout.splitlines() if line.strip()]
    assert rgb_line.startswith("[")
    low, high = depth_line.split()
    assert float(low) <= float(high)
