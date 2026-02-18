#!/usr/bin/env python3
"""
Detectron2 COCO registration for magformer_datasets/0831_1K.

This is intentionally RGB-only for baseline models (Mask R-CNN / official Mask2Former).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Optional


DATASET_NAME_TRAIN = "ecc0831_1k_train"
DATASET_NAME_VAL = "ecc0831_1k_val"


def _default_dataset_root() -> Path:
    # File: <workspace>/magformer/baselines/register_0831_1k_coco.py
    # Dataset: <workspace>/magformer_datasets/0831_1K
    workspace_root = Path(__file__).resolve().parents[2]
    return workspace_root / "magformer_datasets" / "0831_1K"


def register_0831_1k_coco(dataset_root: Optional[str] = None) -> List[str]:
    """
    Register COCO instance datasets for 0831_1K (train/val).

    Args:
        dataset_root: path to `magformer_datasets/0831_1K` directory.
                      If None, try env vars then fall back to workspace-relative default.

    Returns:
        Registered dataset names.
    """
    if dataset_root is None:
        dataset_root = os.environ.get("MAGFORMER_DATASET_ROOT_0831_1K") or os.environ.get("MAGFORMER_DATASET_ROOT")
    root = Path(dataset_root) if dataset_root is not None else _default_dataset_root()

    # Delayed imports so this file can be inspected without detectron2 installed.
    from detectron2.data.datasets import register_coco_instances
    from detectron2.data import MetadataCatalog

    for name, split in [(DATASET_NAME_TRAIN, "train"), (DATASET_NAME_VAL, "val")]:
        img_dir = root / "images" / split
        ann_file = root / "annotations" / f"instances_{split}.json"
        register_coco_instances(name, {}, str(ann_file), str(img_dir))
        MetadataCatalog.get(name).set(thing_classes=["component"])

    return [DATASET_NAME_TRAIN, DATASET_NAME_VAL]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help="Path to magformer_datasets/0831_1K (overrides env + default).",
    )
    ap.add_argument("--verify", action="store_true", help="Call DatasetCatalog.get() and print sample counts.")
    args = ap.parse_args()

    names = register_0831_1k_coco(args.dataset_root)
    print("Registered datasets:", ", ".join(names))

    if args.verify:
        from detectron2.data import DatasetCatalog

        for name in names:
            d = DatasetCatalog.get(name)
            print(f"{name}: {len(d)} images")


if __name__ == "__main__":
    main()

