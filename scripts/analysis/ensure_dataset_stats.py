#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.compute_depth_stats_0831_1k import compute_stats as compute_depth_stats
from scripts.analysis.compute_rgb_stats_coco import compute_rgb_stats


def _dataset_id(dataset_root: Path) -> str:
    return re.sub(r"[^a-z0-9]+", "_", dataset_root.name.strip().lower()).strip("_")


def _depth_train_dir(dataset_root: Path) -> Path:
    candidates = [
        dataset_root / "depth" / "depth_npy" / "train",
        dataset_root / "depth" / "train",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find train depth directory under {dataset_root}")


def ensure_dataset_stats(dataset_root: Path, cache_root: Path, force: bool = False) -> Dict[str, Any]:
    dataset_root = dataset_root.resolve()
    dataset_id = _dataset_id(dataset_root)
    cache_dir = cache_root.resolve() / dataset_id
    cache_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = cache_dir / "manifest.json"
    rgb_stats_path = cache_dir / "rgb_stats.json"
    depth_stats_path = cache_dir / "depth_stats.json"

    if force or not rgb_stats_path.exists():
        rgb_stats = compute_rgb_stats(dataset_root / "images" / "train")
        rgb_stats_path.write_text(json.dumps(rgb_stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    else:
        rgb_stats = json.loads(rgb_stats_path.read_text(encoding="utf-8"))

    if force or not depth_stats_path.exists():
        depth_stats = compute_depth_stats(_depth_train_dir(dataset_root), stride=4096)
        depth_stats_path.write_text(json.dumps(depth_stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    else:
        depth_stats = json.loads(depth_stats_path.read_text(encoding="utf-8"))

    payload = {
        "dataset_id": dataset_id,
        "dataset_root": str(dataset_root),
        "cache_dir": str(cache_dir),
        "manifest_path": str(manifest_path),
        "rgb_stats_path": str(rgb_stats_path),
        "depth_stats_path": str(depth_stats_path),
        "rgb_stats": rgb_stats,
        "depth_stats": depth_stats,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Ensure cached RGB/depth stats exist for a dataset root.")
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--cache-root", type=str, default=str(REPO_ROOT / "output" / "cache" / "dataset_stats"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--field", type=str, default=None)
    args = parser.parse_args()

    payload = ensure_dataset_stats(
        dataset_root=Path(args.dataset_root),
        cache_root=Path(args.cache_root),
        force=bool(args.force),
    )
    if args.field:
        print(payload[args.field])
    else:
        print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
