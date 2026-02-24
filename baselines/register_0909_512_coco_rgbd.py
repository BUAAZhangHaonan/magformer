#!/usr/bin/env python3
"""
Detectron2 COCO registration for magformer_datasets/0909_512_0.12K (RGBD).

Adds `depth_file_name` (points to `.npy`) for each dataset dict entry:
  - <dataset_root>/depth/<split>/<stem>.npy

This is used by RGBD baselines (e.g. UOAIS / MSMFormer wrappers).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Optional


DATASET_NAME_TRAIN = "ecc0909_512_rgbd_train"
DATASET_NAME_VAL = "ecc0909_512_rgbd_val"


def _default_dataset_root() -> Path:
    # File: <workspace>/magformer/baselines/register_0909_512_coco_rgbd.py
    # Dataset: <workspace>/magformer_datasets/0909_512_0.12K
    workspace_root = Path(__file__).resolve().parents[2]
    return workspace_root / "magformer_datasets" / "0909_512_0.12K"


def register_0909_512_coco_rgbd(dataset_root: Optional[str] = None) -> List[str]:
    """
    Register COCO instance datasets for 0909_512_0.12K (train/val) with RGBD file paths.

    Args:
        dataset_root: path to `magformer_datasets/0909_512_0.12K` directory.
                      If None, try env vars then fall back to workspace-relative default.

    Returns:
        Registered dataset names.
    """
    if dataset_root is None:
        dataset_root = os.environ.get("MAGFORMER_DATASET_ROOT_0909_512") or os.environ.get("MAGFORMER_DATASET_ROOT")
    root = Path(dataset_root) if dataset_root is not None else _default_dataset_root()

    # Delayed imports so this file can be inspected without detectron2 installed.
    from detectron2.data import DatasetCatalog, MetadataCatalog
    from detectron2.data.datasets.coco import load_coco_json

    for name, split in [(DATASET_NAME_TRAIN, "train"), (DATASET_NAME_VAL, "val")]:
        img_dir = root / "images" / split
        depth_dir = root / "depth" / split
        ann_file = root / "annotations" / f"instances_{split}.json"

        def _loader(ann_file=ann_file, img_dir=img_dir, depth_dir=depth_dir, name=name):
            ds = load_coco_json(str(ann_file), str(img_dir), dataset_name=name)
            for d in ds:
                stem = Path(d["file_name"]).stem
                depth_path = depth_dir / f"{stem}.npy"
                if not depth_path.exists():
                    raise FileNotFoundError(f"Missing depth file for {d['file_name']}: {depth_path}")
                d["depth_file_name"] = str(depth_path)
            return ds

        # Make registration idempotent across repeated calls (e.g. multiple unit tests).
        if name not in DatasetCatalog:
            DatasetCatalog.register(name, _loader)
        MetadataCatalog.get(name).set(
            thing_classes=["component"],
            evaluator_type="coco",
            image_root=str(img_dir),
            json_file=str(ann_file),
        )

    return [DATASET_NAME_TRAIN, DATASET_NAME_VAL]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help="Path to magformer_datasets/0909_512_0.12K (overrides env + default).",
    )
    ap.add_argument("--verify", action="store_true", help="Call DatasetCatalog.get() and print sample counts.")
    args = ap.parse_args()

    names = register_0909_512_coco_rgbd(args.dataset_root)
    print("Registered datasets:", ", ".join(names))

    if args.verify:
        from detectron2.data import DatasetCatalog

        for name in names:
            d = DatasetCatalog.get(name)
            assert d and isinstance(d, list)
            print(f"{name}: {len(d)} images")
            sample = d[0]
            print(f"  sample keys: {sorted(sample.keys())}")
            print(f"  file_name: {sample['file_name']}")
            print(f"  depth_file_name: {sample['depth_file_name']}")


if __name__ == "__main__":
    main()

