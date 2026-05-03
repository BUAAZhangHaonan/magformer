#!/usr/bin/env python3
"""Generate SHA256 sidecar files for model weights used by integrity checks."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


DEFAULT_SCAN_ROOTS = ("baselines", "output/pretrained")
WEIGHT_SUFFIXES = (".pth", ".bin")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write <weight-file>.sha256 sidecars for .pth and .bin files."
    )
    parser.add_argument(
        "roots",
        nargs="*",
        default=list(DEFAULT_SCAN_ROOTS),
        help="Directories to scan recursively for weight files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rewrite sidecar files even when an existing hash already matches.",
    )
    return parser.parse_args()


def iter_weight_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in WEIGHT_SUFFIXES
    )


def compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_sidecar(sidecar_path: Path) -> str | None:
    if not sidecar_path.exists():
        return None
    return sidecar_path.read_text(encoding="utf-8").strip()


def write_sidecar(sidecar_path: Path, digest: str) -> None:
    sidecar_path.write_text(f"{digest}\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    scanned = 0
    written = 0
    skipped = 0

    for root_arg in args.roots:
        root = Path(root_arg)
        if not root.exists():
            print(f"skip missing root: {root}")
            continue
        if not root.is_dir():
            print(f"skip non-directory root: {root}")
            continue

        for weight_path in iter_weight_files(root):
            scanned += 1
            digest = compute_sha256(weight_path)
            sidecar_path = weight_path.with_name(f"{weight_path.name}.sha256")
            existing_digest = read_sidecar(sidecar_path)

            if existing_digest == digest and not args.force:
                skipped += 1
                continue

            write_sidecar(sidecar_path, digest)
            written += 1

    print(
        f"scanned={scanned} written={written} skipped={skipped} "
        f"roots={len(args.roots)} force={args.force}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
