#!/usr/bin/env python3
"""
Export COCO Results

Runs inference on validation set and exports coco_instances_results.json.
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from magformer.config import load_config, setup_device
from magformer.data import CocoRgbdDataset
from magformer.data.transforms import RGBDTransform
from magformer.data.collate import collate_fn
from magformer.engine.evaluator import COCOEvaluator
from magformer.models import build_model
from magformer.engine.utils import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export COCO results")
    parser.add_argument("--config-file", required=True, help="Path to config yaml")
    parser.add_argument("--dataset-root", required=False, help="Override dataset root")
    parser.add_argument("--weights", required=True, help="Checkpoint path")
    parser.add_argument("--output", default="output/eval", help="Output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    overrides = {"runtime": {"output_dir": args.output}}
    if args.dataset_root is not None:
        overrides.setdefault("data", {})["dataset_root"] = args.dataset_root

    config = load_config(args.config_file, overrides=overrides)

    device = setup_device(config.runtime)

    dataset_root = args.dataset_root or config.data.dataset_root
    dataset = CocoRgbdDataset(
        dataset_root=dataset_root,
        ann_file=config.data.val_ann,
        split="val",
        transform=None,
        is_train=False,
    )
    dataset.transform = RGBDTransform(
        image_size=config.data.image_size,
        min_scale=config.data.min_scale,
        max_scale=config.data.max_scale,
        random_flip="none",
        rgb_brightness=0.0,
        rgb_contrast=0.0,
        rgb_saturation=0.0,
        rgb_hue=0.0,
        depth_scale=config.data.depth.scale,
        depth_shift=config.data.depth.shift,
        depth_clip_min=config.data.depth.clip_min,
        depth_clip_max=config.data.depth.clip_max,
        depth_norm=config.data.depth.norm,
        is_train=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.runtime.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    model = build_model(config)
    load_checkpoint(args.weights, model, strict=False)
    model = model.to(device)
    model.eval()

    evaluator = COCOEvaluator(dataset.coco, iou_types=["bbox", "segm"])
    results = []

    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device)
            depths = batch["depths"].to(device)
            image_ids = batch["image_ids"].tolist()

            outputs = model(images, depths)
            predictions = outputs.get("predictions", [])

            for b_idx, pred in enumerate(predictions):
                img_id = image_ids[b_idx]
                scores = pred.get("scores", [])
                masks = pred.get("masks", [])
                category_ids = pred.get("category_ids", None)

                for s_idx, score in enumerate(scores):
                    mask = masks[s_idx]
                    if isinstance(mask, torch.Tensor):
                        mask = mask.detach().cpu().numpy()
                    mask = (mask > 0.5).astype("uint8")

                    if category_ids is not None:
                        category_id = int(category_ids[s_idx]) + 1
                    else:
                        category_id = 1

                    results.append({
                        "image_id": img_id,
                        "category_id": category_id,
                        "score": float(score),
                        "mask": mask,
                    })

    coco_results = evaluator._convert_to_coco_format(results)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "coco_instances_results.json"

    import json
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(coco_results, f)

    print(f"[Export] Saved results to {output_file}")


if __name__ == "__main__":
    main()
