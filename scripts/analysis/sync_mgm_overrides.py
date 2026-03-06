#!/usr/bin/env python3
from __future__ import annotations

import argparse
import filecmp
import shutil
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--source-root",
        type=str,
        default="third_party_overrides/mgm_mask2former",
    )
    ap.add_argument(
        "--target-root",
        type=str,
        default="../mask2former/MGM_Mask2Former",
    )
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    source_root = (repo_root / args.source_root).resolve()
    target_root = (repo_root / args.target_root).resolve()

    if not source_root.exists():
        raise FileNotFoundError(f"source root not found: {source_root}")
    if not target_root.exists():
        raise FileNotFoundError(f"target root not found: {target_root}")

    copied = 0
    for src in sorted(source_root.rglob("*")):
        if src.is_dir():
            continue
        rel = src.relative_to(source_root)
        dst = target_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and filecmp.cmp(src, dst, shallow=False):
            continue
        shutil.copy2(src, dst)
        copied += 1
        print(f"[sync-mgm-overrides] copied {rel}")

    print(f"[sync-mgm-overrides] total_copied={copied}")


if __name__ == "__main__":
    main()
