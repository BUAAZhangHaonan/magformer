#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path
from typing import Iterable


CANONICAL_TOP_LEVEL = {
    "experiments",
    "baselines",
    "_legacy",
    "README.md",
    "INDEX.md",
}


def _repo_root() -> Path:
    # File: <repo>/scripts/analysis/archive_output_top_level.py
    return Path(__file__).resolve().parents[2]


def _unique_dest(dest: Path) -> Path:
    if not dest.exists():
        return dest
    for i in range(1, 1000):
        cand = dest.with_name(f"{dest.name}_dup{i}")
        if not cand.exists():
            return cand
    raise RuntimeError(f"Failed to pick a unique destination for: {dest}")


def _iter_targets(output_dir: Path) -> Iterable[Path]:
    if not output_dir.exists():
        return []
    for p in sorted(output_dir.iterdir()):
        if p.name in CANONICAL_TOP_LEVEL:
            continue
        yield p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory to canonicalize (default: <repo>/output).",
    )
    ap.add_argument(
        "--tag",
        type=str,
        default="pre_legacy_cleanup",
        help="Archive tag for _legacy folder name.",
    )
    ap.add_argument(
        "--write",
        action="store_true",
        help="Actually move directories/files. If not set, only prints the plan.",
    )
    args = ap.parse_args()

    repo_root = _repo_root()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else (repo_root / "output")

    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    legacy_root = output_dir / "_legacy" / f"{args.tag}_{ts}"

    targets = list(_iter_targets(output_dir))
    if not targets:
        print(f"[archive] nothing to archive under: {output_dir}")
        return

    print(f"[archive] output_dir={output_dir}")
    print(f"[archive] legacy_root={legacy_root}")
    print(f"[archive] mode={'write' if args.write else 'dry-run'}")

    for p in targets:
        dest = legacy_root / p.name
        dest = _unique_dest(dest)
        print(f"[archive] MOVE {p} -> {dest}")

    if not args.write:
        return

    legacy_root.mkdir(parents=True, exist_ok=True)
    for p in targets:
        dest = legacy_root / p.name
        dest = _unique_dest(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(p), str(dest))

    print("[archive] done")


if __name__ == "__main__":
    main()
