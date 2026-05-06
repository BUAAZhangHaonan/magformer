#!/usr/bin/env python3
"""Build VC-SUDA dataset from per-image AP results.

Selects bottom-N% images by AP and creates new COCO annotation files.
The resulting dataset is used as the "target" domain for VC-SUDA training.

Usage:
    python tools/build_vc_suda_dataset.py \
        --ap-results output/per_image_ap_train.json \
        --source-ann /path/to/instances_train.json \
        --output-dir /path/to/magformer_datasets/vc_suda_32k \
        --bottom-percent 10
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Build VC-SUDA dataset")
    parser.add_argument("--ap-results", required=True,
                        help="JSON file from compute_per_image_ap.py")
    parser.add_argument("--source-ann", required=True,
                        help="Original train annotation JSON")
    parser.add_argument("--val-ann", default=None,
                        help="Original val annotation JSON (optional, for splitting)")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for new dataset")
    parser.add_argument("--bottom-percent", type=float, default=10,
                        help="Select bottom N%% of images by AP")
    parser.add_argument("--labeled-count", type=int, default=20,
                        help="Number of target images to keep labeled (Stage B warm-up)")
    parser.add_argument("--copy-images", action="store_true",
                        help="Copy image files (if False, symlink)")
    parser.add_argument("--source-images-dir", default=None,
                        help="Source images directory (for symlinks/copy)")
    return parser.parse_args()


def load_coco_ann(path):
    with open(path, "r") as f:
        return json.load(f)


def save_coco_ann(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def filter_coco_by_image_ids(coco_data, image_ids_set):
    """Filter COCO annotation to only include specified image IDs."""
    filtered_images = [img for img in coco_data["images"]
                       if img["id"] in image_ids_set]
    filtered_img_ids = {img["id"] for img in filtered_images}
    filtered_anns = [ann for ann in coco_data["annotations"]
                     if ann["image_id"] in filtered_img_ids]

    result = {
        "info": coco_data.get("info", {}),
        "licenses": coco_data.get("licenses", []),
        "categories": coco_data["categories"],
        "images": filtered_images,
        "annotations": filtered_anns,
    }
    return result


def main():
    args = parse_args()

    # Load per-image AP
    with open(args.ap_results, "r") as f:
        ap_data = json.load(f)

    per_image_ap = {int(k): v for k, v in ap_data["per_image_ap"].items()}
    sorted_images = sorted(per_image_ap.items(), key=lambda x: x[1])

    total = len(sorted_images)
    n_bottom = max(1, int(total * args.bottom_percent / 100))

    bottom_images = sorted_images[:n_bottom]
    bottom_ids = {img_id for img_id, _ in bottom_images}
    remaining_ids = {img_id for img_id, _ in sorted_images[n_bottom:]}

    print(f"[DatasetBuilder] Total images: {total}")
    print(f"[DatasetBuilder] Bottom {args.bottom_percent}%: {n_bottom} images")

    # AP statistics for bottom set
    bottom_aps = [ap for _, ap in bottom_images]
    print(f"  Bottom set AP: mean={np.mean(bottom_aps):.4f}, "
          f"min={np.min(bottom_aps):.4f}, max={np.max(bottom_aps):.4f}")

    # Load source annotations
    source_coco = load_coco_ann(args.source_ann)
    print(f"[DatasetBuilder] Source annotations: {len(source_coco['images'])} images, "
          f"{len(source_coco['annotations'])} annotations")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ann_dir = output_dir / "annotations"
    ann_dir.mkdir(exist_ok=True)

    # 1. Source train (remaining images - "easy" domain)
    source_train_coco = filter_coco_by_image_ids(source_coco, remaining_ids)
    save_coco_ann(source_train_coco, ann_dir / "instances_source_train.json")
    print(f"\n[Saved] instances_source_train.json: "
          f"{len(source_train_coco['images'])} images, "
          f"{len(source_train_coco['annotations'])} annotations")

    # 2. Target labeled (top of bottom set - for Stage B warm-up)
    labeled_images = bottom_images[-args.labeled_count:]  # highest AP among bottom
    labeled_ids = {img_id for img_id, _ in labeled_images}
    target_labeled_coco = filter_coco_by_image_ids(source_coco, labeled_ids)
    save_coco_ann(target_labeled_coco, ann_dir / "instances_target_labeled.json")
    print(f"[Saved] instances_target_labeled.json: "
          f"{len(target_labeled_coco['images'])} images, "
          f"{len(target_labeled_coco['annotations'])} annotations")

    # 3. Target unlabeled (rest of bottom set - for Stage C+)
    unlabeled_ids = bottom_ids - labeled_ids
    target_unlabeled_coco = filter_coco_by_image_ids(source_coco, unlabeled_ids)

    # Remove annotations for unlabeled set (they won't have labels in semi-supervised)
    # But keep the annotations in a separate file for reference/evaluation
    target_unlabeled_noann = {
        "info": target_unlabeled_coco.get("info", {}),
        "licenses": target_unlabeled_coco.get("licenses", []),
        "categories": target_unlabeled_coco["categories"],
        "images": target_unlabeled_coco["images"],
        "annotations": [],  # No annotations for unlabeled
    }
    save_coco_ann(target_unlabeled_noann, ann_dir / "instances_target_unlabeled.json")
    save_coco_ann(target_unlabeled_coco, ann_dir / "instances_target_unlabeled_with_gt.json")
    print(f"[Saved] instances_target_unlabeled.json: "
          f"{len(target_unlabeled_noann['images'])} images (no annotations)")
    print(f"[Saved] instances_target_unlabeled_with_gt.json: "
          f"{len(target_unlabeled_coco['images'])} images, "
          f"{len(target_unlabeled_coco['annotations'])} annotations (GT reference)")

    # 4. Target val (all bottom images with annotations, for evaluation)
    target_val_coco = filter_coco_by_image_ids(source_coco, bottom_ids)
    save_coco_ann(target_val_coco, ann_dir / "instances_target_val.json")
    print(f"[Saved] instances_target_val.json: "
          f"{len(target_val_coco['images'])} images, "
          f"{len(target_val_coco['annotations'])} annotations")

    # 5. Full train (all images, for Stage A pretrain)
    save_coco_ann(source_coco, ann_dir / "instances_train_all.json")
    print(f"[Saved] instances_train_all.json: "
          f"{len(source_coco['images'])} images (full train set)")

    # Save per-image AP mapping for reference
    ap_summary = {
        "source_file": str(args.source_ann),
        "total_images": total,
        "bottom_percent": args.bottom_percent,
        "bottom_count": n_bottom,
        "labeled_count": args.labeled_count,
        "unlabeled_count": len(unlabeled_ids),
        "source_train_count": len(remaining_ids),
        "per_image_ap": ap_data["per_image_ap"],
        "bottom_image_ids": sorted([int(x) for x in bottom_ids]),
        "labeled_image_ids": sorted([int(x) for x in labeled_ids]),
        "unlabeled_image_ids": sorted([int(x) for x in unlabeled_ids]),
    }
    save_coco_ann(ap_summary, ann_dir / "dataset_split_info.json")

    # 6. Create symlinks to image/depth directories
    source_root = Path(args.source_ann).parent.parent
    for subdir in ["images", "depth"]:
        src = source_root / subdir
        dst = output_dir / subdir
        if src.exists() and not dst.exists():
            dst.symlink_to(src)
            print(f"[Symlink] {dst} -> {src}")
        elif not src.exists():
            print(f"[Warning] Source {src} does not exist, skipping")

    print(f"\n[DatasetBuilder] Dataset created at {output_dir}")
    print(f"  Annotations in: {ann_dir}")
    print(f"  Summary:")
    print(f"    Source train: {len(remaining_ids)} images")
    print(f"    Target labeled: {len(labeled_ids)} images")
    print(f"    Target unlabeled: {len(unlabeled_ids)} images")
    print(f"    Target val: {n_bottom} images")


if __name__ == "__main__":
    main()
