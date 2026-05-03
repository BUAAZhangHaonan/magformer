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


def _load_depth_stats_json(path: Path) -> DepthStats:
    d = json.loads(path.read_text(encoding="utf-8"))
    return DepthStats(
        p1=float(d["p1"]),
        p99=float(d["p99"]),
        min=float(d["min"]),
        max=float(d["max"]),
    )


def load_0831_1k_depth_stats() -> DepthStats:
    """
    Load global depth normalization stats for ECC 0831_1K.

    Source of truth: `configs/stats/0831_1k_depth_stats.json`
    """
    # File: <repo>/baselines/depth_stats.py
    repo_root = Path(__file__).resolve().parents[1]
    p = repo_root / "configs" / "stats" / "0831_1k_depth_stats.json"
    return _load_depth_stats_json(p)


def load_0909_512_depth_stats() -> DepthStats:
    """
    Load global depth normalization stats for ECC 0909_512_0.12K.

    Source of truth: `configs/stats/0909_512_depth_stats.json`
    """
    # File: <repo>/baselines/depth_stats.py
    repo_root = Path(__file__).resolve().parents[1]
    p = repo_root / "configs" / "stats" / "0909_512_depth_stats.json"
    return _load_depth_stats_json(p)


def load_depth_stats_for_dataset_root(dataset_root: str) -> DepthStats:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in __import__("sys").path:
        __import__("sys").path.insert(0, str(repo_root))
    from scripts.analysis.ensure_dataset_stats import ensure_dataset_stats

    payload = ensure_dataset_stats(
        dataset_root=Path(dataset_root),
        cache_root=repo_root / "output" / "cache" / "dataset_stats",
    )
    return _load_depth_stats_json(Path(payload["depth_stats_path"]))
