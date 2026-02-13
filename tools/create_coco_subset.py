#!/usr/bin/env python3
"""
Create a deterministic COCO subset annotation file.

Example:
  python tools/create_coco_subset.py \
    --input magformer_datasets/0831_1K/annotations/instances_train.json \
    --output magformer_datasets/0831_1K/annotations/instances_train_subset20.json \
    --num-images 20 \
    --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("create_coco_subset")
    p.add_argument("--input", required=True, help="Input COCO annotation json.")
    p.add_argument("--output", required=True, help="Output subset annotation json.")
    p.add_argument("--num-images", type=int, default=20, help="Number of images to keep.")
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (used when --sample=rand).",
    )
    p.add_argument(
        "--sample",
        choices=["first", "rand"],
        default="first",
        help="Sampling strategy for image selection.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with in_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    images = list(coco.get("images", []))
    num = max(1, min(int(args.num_images), len(images)))

    if args.sample == "rand":
        rng = random.Random(args.seed)
        selected = rng.sample(images, num)
    else:
        selected = images[:num]

    selected_ids = {img["id"] for img in selected}
    annotations = [
        ann for ann in coco.get("annotations", [])
        if ann.get("image_id") in selected_ids
    ]

    subset = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "images": selected,
        "annotations": annotations,
        "categories": coco.get("categories", []),
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(subset, f)

    print(
        f"[create_coco_subset] images={len(selected)} annotations={len(annotations)} -> {out_path}"
    )


if __name__ == "__main__":
    main()
