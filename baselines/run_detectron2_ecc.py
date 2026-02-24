#!/usr/bin/env python3
"""
Run Detectron2 official training script with ECC datasets (0831 / 0909).

We keep detectron2 as a submodule under `baselines/detectron2` and do not patch it.
This wrapper registers the dataset in-process and then executes detectron2's train_net.py.
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

from ecc_datasets import register_ecc_coco


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
        "--detectron2-root",
        type=str,
        default="baselines/detectron2",
        help="Path to official detectron2 repo checkout.",
    )
    args = ap.parse_args(wrapper_argv)

    register_ecc_coco(args.register, args.dataset_root)

    repo_root = Path(args.detectron2_root).resolve()
    train_py = repo_root / "tools" / "train_net.py"
    if not train_py.exists():
        raise FileNotFoundError(f"tools/train_net.py not found: {train_py}")

    os.chdir(repo_root)
    sys.path.insert(0, str(repo_root))

    sys.argv = [str(train_py)] + passthrough
    runpy.run_path(str(train_py), run_name="__main__")


if __name__ == "__main__":
    main()

