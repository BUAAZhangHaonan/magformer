#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable, List, Set


def _repo_root() -> Path:
    # File: <repo>/scripts/analysis/cleanup_temp_artifacts.py
    return Path(__file__).resolve().parents[2]


def _iter_static_targets(repo_root: Path) -> Iterable[Path]:
    patterns = [
        "output/experiments/0831_1k_5k_scratch8",
        "output/experiments/tmp_eval_full40k",
        "output/_legacy/tmp_*",
        "output/_legacy/pre_*scratch8*",
        "output/_legacy/package_tree_output_*",
        "baselines/*/runs",
    ]
    for pattern in patterns:
        for path in repo_root.glob(pattern):
            if path.exists():
                yield path


def _iter_cache_targets(repo_root: Path) -> Iterable[Path]:
    pytest_cache = repo_root / ".pytest_cache"
    if pytest_cache.exists():
        yield pytest_cache

    for path in repo_root.rglob("__pycache__"):
        if path.exists():
            yield path


def _unique_paths(paths: Iterable[Path]) -> List[Path]:
    seen: Set[Path] = set()
    unique: List[Path] = []
    for path in paths:
        rp = path.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        unique.append(path)
    return sorted(unique, key=lambda p: str(p))


def _safe_under_repo(repo_root: Path, path: Path) -> bool:
    try:
        path.resolve().relative_to(repo_root.resolve())
        return True
    except Exception:
        return False


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
        return
    shutil.rmtree(path, ignore_errors=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print matched paths without deleting (default behavior).",
    )
    ap.add_argument(
        "--write",
        action="store_true",
        help="Actually remove matched paths. Default is dry-run.",
    )
    args = ap.parse_args()

    if args.dry_run and args.write:
        raise SystemExit("Use either --dry-run or --write, not both.")

    repo_root = _repo_root()
    targets = _unique_paths(list(_iter_static_targets(repo_root)) + list(_iter_cache_targets(repo_root)))

    if not targets:
        print("[cleanup] nothing to remove")
        return

    print(f"[cleanup] repo_root={repo_root}")
    print(f"[cleanup] mode={'write' if args.write else 'dry-run'}")
    print(f"[cleanup] targets={len(targets)}")
    for path in targets:
        rel = path.resolve().relative_to(repo_root.resolve())
        print(f"[cleanup] REMOVE {rel}")

    if not args.write:
        return

    for path in targets:
        if not path.exists():
            continue
        if not _safe_under_repo(repo_root, path):
            raise RuntimeError(f"Refusing to remove path outside repo: {path}")
        _remove_path(path)

    print("[cleanup] done")


if __name__ == "__main__":
    main()
