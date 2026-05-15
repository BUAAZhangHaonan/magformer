#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np


def compute_stats(depth_dir: Path, stride: int) -> Dict[str, float | int | str]:
    paths = sorted(depth_dir.glob("*.npy"))
    if not paths:
        raise FileNotFoundError(f"No .npy files found under: {depth_dir}")

    samples = []
    mins = []
    maxs = []
    for p in paths:
        a = np.load(p).astype(np.float32)
        mins.append(float(a.min()))
        maxs.append(float(a.max()))
        samples.append(a.ravel()[::stride])

    s = np.concatenate(samples)
    return {
        "p1": float(np.percentile(s, 1.0)),
        "p99": float(np.percentile(s, 99.0)),
        "min": float(min(mins)),
        "max": float(max(maxs)),
        "sampled_pixels": int(s.size),
        "stride": int(stride),
        "script_version": "v1",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help="Path to magformer_datasets/0831_1K (default: workspace-relative).",
    )
    ap.add_argument("--stride", type=int, default=4096, help="Deterministic sampling stride.")
    ap.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: <repo>/configs/stats/0831_1k_depth_stats.json).",
    )
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    ws_root = repo_root.parent
    dataset_root = Path(args.dataset_root) if args.dataset_root else (ws_root / "magformer_datasets" / "0831_1K")

    depth_dir = dataset_root / "depth" / "train"
    stats = compute_stats(depth_dir=depth_dir, stride=int(args.stride))

    out = Path(args.output) if args.output else (repo_root / "configs" / "stats" / "0831_1k_depth_stats.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"[depth-stats] wrote: {out}")


if __name__ == "__main__":
    main()
