#!/usr/bin/env python3
"""
Run official facebookresearch/Mask2Former training with our custom 0831_1K dataset.

Why this wrapper exists:
- Detectron2 requires datasets to be registered in-process.
- We keep official baseline repos as submodules (no patching inside them).
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

from register_0831_1k_coco import register_0831_1k_coco


def _split_args(argv: List[str]) -> tuple[list[str], list[str]]:
    if "--" in argv:
        i = argv.index("--")
        return argv[:i], argv[i + 1 :]
    return argv, []


def main() -> None:
    wrapper_argv, passthrough = _split_args(sys.argv[1:])

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help="Path to magformer_datasets/0831_1K (default: workspace-relative).",
    )
    ap.add_argument(
        "--mask2former-root",
        type=str,
        default="baselines/Mask2Former",
        help="Path to official Mask2Former repo checkout.",
    )
    args = ap.parse_args(wrapper_argv)

    register_0831_1k_coco(args.dataset_root)

    repo_root = Path(args.mask2former_root).resolve()
    train_py = repo_root / "train_net.py"
    if not train_py.exists():
        raise FileNotFoundError(f"train_net.py not found: {train_py}")

    os.chdir(repo_root)

    # Ensure `import mask2former` works when running train_net.py
    sys.path.insert(0, str(repo_root))

    sys.argv = [str(train_py)] + passthrough
    runpy.run_path(str(train_py), run_name="__main__")


if __name__ == "__main__":
    main()
