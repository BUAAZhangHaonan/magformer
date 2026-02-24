#!/usr/bin/env python3
"""
Run UOAIS (vendored baseline) on ECC datasets (0831 / 0909) with COCOeval (segm/bbox AP).

This wrapper:
- registers the ECC RGBD COCO dataset in-process (adds `depth_file_name`)
- executes the vendored UOAIS `train_net.py` with passthrough args
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path
from typing import List

# Ensure sibling baseline utilities are importable when running as a file.
BASELINES_DIR = Path(__file__).resolve().parent
if str(BASELINES_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINES_DIR))

from ecc_datasets import register_ecc_coco_rgbd


def _split_args(argv: List[str]) -> tuple[list[str], list[str]]:
    if "--" in argv:
        i = argv.index("--")
        return argv[:i], argv[i + 1 :]
    return argv, []


def main() -> None:
    wrapper_argv, passthrough = _split_args(sys.argv[1:])

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--register",
        type=str,
        default="0831",
        help="ECC dataset id: 0831 | 0909",
    )
    ap.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help="Path to magformer_datasets/<dataset> root (default: env + workspace-relative).",
    )
    ap.add_argument(
        "--uoais-root",
        type=str,
        default="baselines/uoais",
        help="Path to vendored UOAIS repo checkout.",
    )
    args = ap.parse_args(wrapper_argv)

    register_ecc_coco_rgbd(args.register, args.dataset_root)

    repo_root = Path(args.uoais_root).resolve()
    train_py = repo_root / "train_net.py"
    if not train_py.exists():
        raise FileNotFoundError(f"train_net.py not found: {train_py}")

    os.chdir(repo_root)
    sys.path.insert(0, str(repo_root))

    sys.argv = [str(train_py)] + passthrough
    runpy.run_path(str(train_py), run_name="__main__")


if __name__ == "__main__":
    main()

