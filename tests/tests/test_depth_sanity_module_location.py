from __future__ import annotations

from pathlib import Path


def test_depth_sanity_module_lives_under_utils() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    assert not (repo_root / "magformer" / "depth_sanity.py").exists()
    assert (repo_root / "magformer" / "utils" / "depth_sanity.py").exists()
