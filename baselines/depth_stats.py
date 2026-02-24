from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DepthStats:
    p1: float
    p99: float
    min: float
    max: float


def load_0831_1k_depth_stats() -> DepthStats:
    """
    Load global depth normalization stats for ECC 0831_1K.

    Source of truth: `configs/stats/0831_1k_depth_stats.json`
    """
    # File: <repo>/baselines/depth_stats.py
    repo_root = Path(__file__).resolve().parents[1]
    p = repo_root / "configs" / "stats" / "0831_1k_depth_stats.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    return DepthStats(
        p1=float(d["p1"]),
        p99=float(d["p99"]),
        min=float(d["min"]),
        max=float(d["max"]),
    )


def load_0909_512_depth_stats() -> DepthStats:
    """
    Load global depth normalization stats for ECC 0909_512_0.12K.

    Source of truth: `configs/stats/0909_512_depth_stats.json`
    """
    # File: <repo>/baselines/depth_stats.py
    repo_root = Path(__file__).resolve().parents[1]
    p = repo_root / "configs" / "stats" / "0909_512_depth_stats.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    return DepthStats(
        p1=float(d["p1"]),
        p99=float(d["p99"]),
        min=float(d["min"]),
        max=float(d["max"]),
    )
