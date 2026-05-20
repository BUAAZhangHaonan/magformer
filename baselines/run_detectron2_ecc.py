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
from typing import List, Optional, Tuple

# Ensure sibling baseline utilities are importable when running as a file.
BASELINES_DIR = Path(__file__).resolve().parent
if str(BASELINES_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINES_DIR))

from ecc_datasets import dataset_prefix, register_ecc_coco
from run_official_mask2former_ecc import _register_coco_split, _resolve_dataset_path


def _split_args(argv: List[str]) -> tuple[list[str], list[str]]:
    if "--" in argv:
        i = argv.index("--")
        return argv[:i], argv[i + 1 :]
    return argv, []


def _has_explicit_split_args(
    *,
    train_ann: Optional[str],
    val_ann: Optional[str],
    train_image_dir: Optional[str],
    val_image_dir: Optional[str],
    train_split: str,
    val_split: str,
) -> bool:
    return any((train_ann, val_ann, train_image_dir, val_image_dir)) or train_split != "train" or val_split != "val"


def register_detectron2_datasets(
    *,
    register: str,
    dataset_root: Optional[str],
    train_ann: Optional[str] = None,
    val_ann: Optional[str] = None,
    train_image_dir: Optional[str] = None,
    val_image_dir: Optional[str] = None,
    train_split: str = "train",
    val_split: str = "val",
    normalized_ann_dir: str = "output/diagnostics/detectron2_coco",
) -> Tuple[str, str]:
    if not _has_explicit_split_args(
        train_ann=train_ann,
        val_ann=val_ann,
        train_image_dir=train_image_dir,
        val_image_dir=val_image_dir,
        train_split=train_split,
        val_split=val_split,
    ):
        return register_ecc_coco(register, dataset_root)

    if dataset_root is None:
        raise ValueError("--dataset-root is required when using explicit Detectron2 split arguments")
    if train_ann is None or val_ann is None or train_image_dir is None or val_image_dir is None:
        raise ValueError(
            "Explicit Detectron2 split registration requires --train-ann, --val-ann, "
            "--train-image-dir, and --val-image-dir"
        )

    prefix = dataset_prefix(register, dataset_root)
    train_name = f"{prefix}_{train_split}"
    val_name = f"{prefix}_{val_split}"

    _register_coco_split(
        name=train_name,
        ann_file=_resolve_dataset_path(dataset_root, train_ann),
        image_root=_resolve_dataset_path(dataset_root, train_image_dir),
        normalized_ann_dir=normalized_ann_dir,
    )
    _register_coco_split(
        name=val_name,
        ann_file=_resolve_dataset_path(dataset_root, val_ann),
        image_root=_resolve_dataset_path(dataset_root, val_image_dir),
        normalized_ann_dir=normalized_ann_dir,
    )
    print(f"[detectron2] registered train dataset: {train_name}", flush=True)
    print(f"[detectron2] registered val/test dataset: {val_name}", flush=True)
    return train_name, val_name


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
    ap.add_argument("--train-ann", type=str, default=None, help="Train COCO annotation file, absolute or relative to --dataset-root.")
    ap.add_argument("--val-ann", type=str, default=None, help="Val/test COCO annotation file, absolute or relative to --dataset-root.")
    ap.add_argument("--train-image-dir", type=str, default=None, help="Train image directory, absolute or relative to --dataset-root.")
    ap.add_argument("--val-image-dir", type=str, default=None, help="Val/test image directory, absolute or relative to --dataset-root.")
    ap.add_argument("--train-split", type=str, default="train", help="Train split suffix used in the registered dataset name.")
    ap.add_argument("--val-split", type=str, default="val", help="Val/test split suffix used in the registered dataset name.")
    ap.add_argument(
        "--normalized-ann-dir",
        type=str,
        default="output/diagnostics/detectron2_coco",
        help="Directory for temporary COCO JSON files with required info/licenses metadata.",
    )
    args = ap.parse_args(wrapper_argv)

    register_detectron2_datasets(
        register=args.register,
        dataset_root=args.dataset_root,
        train_ann=args.train_ann,
        val_ann=args.val_ann,
        train_image_dir=args.train_image_dir,
        val_image_dir=args.val_image_dir,
        train_split=args.train_split,
        val_split=args.val_split,
        normalized_ann_dir=args.normalized_ann_dir,
    )

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
