from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "baselines" / "unet_instance_models.py"
    spec = importlib.util.spec_from_file_location("unet_instance_models", path)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_affinity_graph_merge_merges_split_parts_of_same_object() -> None:
    mod = _load_module()
    fragments = np.zeros((32, 32), dtype=np.int32)
    fragments[8:16, 8:12] = 1
    fragments[8:16, 12:16] = 2
    boundary = np.zeros((32, 32), dtype=np.float32)
    affinity = np.ones((2, 32, 32), dtype=np.float32) * 0.9

    merged = mod.merge_fragment_graph(
        fragments=fragments,
        boundary_prob=boundary,
        affinity_prob=affinity,
        shape_consistency={(1, 2): 1.0},
        merge_threshold=0.6,
    )

    labels = sorted(x for x in np.unique(merged).tolist() if x > 0)
    assert labels == [1]


def test_affinity_graph_merge_keeps_adjacent_objects_separate_when_boundary_is_strong() -> None:
    mod = _load_module()
    fragments = np.zeros((32, 32), dtype=np.int32)
    fragments[8:16, 8:12] = 1
    fragments[8:16, 12:16] = 2
    boundary = np.zeros((32, 32), dtype=np.float32)
    boundary[:, 11:13] = 1.0
    affinity = np.ones((2, 32, 32), dtype=np.float32) * 0.9

    merged = mod.merge_fragment_graph(
        fragments=fragments,
        boundary_prob=boundary,
        affinity_prob=affinity,
        shape_consistency={(1, 2): 0.0},
        merge_threshold=0.6,
    )

    labels = sorted(x for x in np.unique(merged).tolist() if x > 0)
    assert labels == [1, 2]

