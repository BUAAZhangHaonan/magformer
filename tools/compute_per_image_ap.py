#!/usr/bin/env python3
"""Compute per-image AP on the train set using a trained model.

Outputs a JSON file mapping image_id -> AP, sorted by AP ascending.
Used to identify the hardest images for VC-SUDA target domain.

Usage:
    CUDA_VISIBLE_DEVICES=4 python tools/compute_per_image_ap.py \
        --config-file configs/finetune_32k_teacher.yaml \
        --weights output/experiments/20260505_32k_finetune_teacher/model_best.pth \
        --output output/per_image_ap_train.json \
        --batch-size 1 \
        --split train
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np

sys.path.insert(0, str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from magformer.config import load_config, setup_device, set_seed
from magformer.data import CocoRgbdDataset
from magformer.data.transforms import RGBDTransform
from magformer.data.collate import collate_fn
from magformer.engine.coco_export import outputs_to_coco_instances
from magformer.models import build_model
from magformer.engine.utils import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Compute per-image AP")
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--weights", required=True, help="Model checkpoint")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--output", default="output/per_image_ap.json")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--score-threshold", type=float, default=0.05)
    return parser.parse_args()


def build_loader(config, split="train", dataset_root=None, num_workers=4, batch_size=1):
    data_cfg = config.data
    dataset_root = dataset_root or data_cfg.dataset_root

    if split == "train":
        ann_file = data_cfg.train_ann
    else:
        ann_file = data_cfg.val_ann

    dataset = CocoRgbdDataset(
        dataset_root=dataset_root,
        ann_file=ann_file,
        split=split,
        transform=None,
        is_train=False,
    )

    dataset.transform = RGBDTransform(
        image_size=data_cfg.image_size,
        min_scale=data_cfg.min_scale,
        max_scale=data_cfg.max_scale,
        random_flip="none",
        rgb_brightness=0.0,
        rgb_contrast=0.0,
        rgb_saturation=0.0,
        rgb_hue=0.0,
        depth_clip_min=data_cfg.depth.clip_min,
        depth_clip_max=data_cfg.depth.clip_max,
        depth_norm=data_cfg.depth.norm,
        depth_per_sample_norm=getattr(data_cfg.depth, "per_sample_norm", True),
        is_train=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    return dataset, loader


@torch.no_grad()
def run_inference(model, loader, device, score_threshold=0.05, category_ids=None):
    """Run inference on all images, collect COCO-format predictions."""
    all_predictions = []
    total = len(loader)
    t0 = time.time()

    for i, batch in enumerate(loader):
        images = batch["images"].to(device)
        depths = batch["depths"].to(device)
        noise_masks = batch.get("noise_masks")
        if noise_masks is not None:
            noise_masks = noise_masks.to(device)
        padding_masks = batch.get("padding_masks")
        if padding_masks is not None:
            padding_masks = padding_masks.to(device)

        outputs = model.forward_inference_raw(
            images, depths,
            padding_masks=padding_masks,
            depth_noise_masks=noise_masks,
        )

        preds = outputs_to_coco_instances(
            outputs=outputs,
            image_ids=batch.get("image_ids"),
            score_threshold=score_threshold,
            mask_threshold=0.5,
            category_offset=1,
            category_ids=category_ids,
        )
        all_predictions.extend(preds)

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (total - i - 1)
            print(f"  [{i+1}/{total}] {len(all_predictions)} preds, "
                  f"elapsed {elapsed:.0f}s, ETA {eta:.0f}s")

    elapsed = time.time() - t0
    print(f"  Inference done: {len(all_predictions)} predictions in {elapsed:.1f}s")
    return all_predictions


def compute_per_image_ap(coco_gt, predictions, iou_type="segm"):
    """Compute AP per image using COCOeval with image filtering."""
    if not predictions:
        print("  WARNING: No predictions to evaluate")
        return {}

    # Load predictions into COCO format
    coco_dt = coco_gt.loadRes(predictions)

    all_img_ids = sorted(coco_gt.getImgIds())
    total = len(all_img_ids)
    per_image = {}

    print(f"  Computing per-image AP for {total} images (iou_type={iou_type})...")

    for idx, img_id in enumerate(all_img_ids):
        try:
            coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
            coco_eval.params.imgIds = [img_id]
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            ap = float(coco_eval.stats[0])  # AP@[0.50:0.95]
        except Exception:
            ap = -1.0

        per_image[img_id] = ap

        if (idx + 1) % 500 == 0:
            print(f"    [{idx+1}/{total}] per-image AP computed")

    return per_image


def main():
    args = parse_args()

    overrides = {}
    if args.dataset_root:
        overrides.setdefault("data", {})["dataset_root"] = args.dataset_root

    config = load_config(args.config_file, overrides=overrides)
    device = setup_device(config.runtime)
    set_seed(config.runtime.seed)

    print(f"[PerImageAP] Loading model from {args.weights}")
    model = build_model(config)
    load_checkpoint(args.weights, model, strict=False)
    model = model.to(device)
    model.eval()

    dataset, loader = build_loader(
        config,
        split=args.split,
        dataset_root=args.dataset_root,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
    )

    category_ids = list(getattr(dataset, "category_ids", [])) or None

    print(f"[PerImageAP] Running inference on {args.split} set ({len(dataset)} images)")
    predictions = run_inference(model, loader, device,
                                score_threshold=args.score_threshold,
                                category_ids=category_ids)

    print(f"[PerImageAP] Computing per-image AP")
    per_image_ap = compute_per_image_ap(dataset.coco, predictions, iou_type="segm")

    # Sort by AP ascending
    sorted_ap = sorted(per_image_ap.items(), key=lambda x: x[1])

    # Build result
    result = {
        "config_file": args.config_file,
        "weights": args.weights,
        "split": args.split,
        "total_images": len(sorted_ap),
        "per_image_ap": {str(k): v for k, v in sorted_ap},
        "statistics": {
            "mean_ap": float(np.mean([v for v in per_image_ap.values() if v >= 0])),
            "median_ap": float(np.median([v for v in per_image_ap.values() if v >= 0])),
            "min_ap": sorted_ap[0][1] if sorted_ap else -1,
            "max_ap": sorted_ap[-1][1] if sorted_ap else -1,
            "p10_ap": float(np.percentile([v for v in per_image_ap.values() if v >= 0], 10)),
            "p25_ap": float(np.percentile([v for v in per_image_ap.values() if v >= 0], 25)),
            "p50_ap": float(np.percentile([v for v in per_image_ap.values() if v >= 0], 50)),
        }
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n[PerImageAP] Results saved to {output_path}")
    stats = result["statistics"]
    print(f"  Total: {result['total_images']} images")
    print(f"  Mean AP: {stats['mean_ap']:.4f}")
    print(f"  Median AP: {stats['median_ap']:.4f}")
    print(f"  P10 AP: {stats['p10_ap']:.4f}")
    print(f"  Min AP: {stats['min_ap']:.4f}")
    print(f"  Max AP: {stats['max_ap']:.4f}")

    # Show bottom 10
    print(f"\n  Bottom 10 images:")
    for img_id, ap in sorted_ap[:10]:
        print(f"    image_id={img_id}, AP={ap:.4f}")


if __name__ == "__main__":
    main()
