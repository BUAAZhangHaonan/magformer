#!/usr/bin/env python3
"""
Analyze validation ranking quality for MAGFormer instance predictions.

This script bypasses COCOeval and directly measures:
- per-image best IoU (all predictions / top-10 / top-1)
- score statistics for good vs bad predictions
- average number of predictions kept by score threshold
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from pycocotools import mask as coco_mask
from torch.utils.data import DataLoader

# Add magformer/ to sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from magformer.config import load_config, setup_device, set_seed  # noqa: E402
from magformer.data import CocoRgbdDataset  # noqa: E402
from magformer.data.collate import collate_fn  # noqa: E402
from magformer.data.transforms import RGBDTransform  # noqa: E402
from magformer.engine.utils import load_checkpoint  # noqa: E402
from magformer.models import build_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Analyze validation ranking quality")
    parser.add_argument(
        "--config-file",
        required=True,
        help="Path to config yaml.",
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Dataset root path.",
    )
    parser.add_argument(
        "--weights",
        required=True,
        help="Checkpoint path.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=110,
        help="Maximum number of validation images to analyze.",
    )
    parser.add_argument(
        "--score-thr",
        type=float,
        default=0.05,
        help="Score threshold used when counting kept predictions.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Dataloader workers for analysis.",
    )
    parser.add_argument(
        "--fast-size",
        type=int,
        default=256,
        help=(
            "Downsample masks to this size for fast IoU analysis. "
            "Set <=0 to disable downsampling."
        ),
    )
    return parser.parse_args()


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    if inter == 0:
        return 0.0
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union + 1e-9)


def resize_bool_masks(masks: List[np.ndarray], size: int) -> List[np.ndarray]:
    if size <= 0:
        return masks
    import cv2

    out: List[np.ndarray] = []
    for m in masks:
        rs = cv2.resize(
            m.astype(np.uint8),
            (size, size),
            interpolation=cv2.INTER_NEAREST,
        )
        out.append(rs.astype(bool))
    return out


def decode_gt_masks(dataset: CocoRgbdDataset) -> Dict[int, List[np.ndarray]]:
    gt_by_image: Dict[int, List[np.ndarray]] = {}
    for image_id in dataset.image_ids:
        image_info = dataset.coco.imgs[image_id]
        h, w = image_info["height"], image_info["width"]
        ann_ids = dataset.coco.getAnnIds(imgIds=image_id)
        anns = dataset.coco.loadAnns(ann_ids)
        masks: List[np.ndarray] = []
        for ann in anns:
            if ann.get("iscrowd", 0) != 0:
                continue
            seg = ann.get("segmentation")
            if seg is None:
                continue
            if isinstance(seg, list):
                rles = coco_mask.frPyObjects(seg, h, w)
                m = coco_mask.decode(rles)
                if m.ndim == 3:
                    m = np.any(m, axis=2)
            else:
                m = coco_mask.decode(seg)
                if m.ndim == 3:
                    m = m[..., 0]
            masks.append(m.astype(bool))
        gt_by_image[image_id] = masks
    return gt_by_image


def build_val_loader(config, dataset_root: str, num_workers: int) -> tuple[CocoRgbdDataset, DataLoader]:
    val_split = getattr(config.data, "val_split", "val")
    dataset = CocoRgbdDataset(
        dataset_root=dataset_root,
        ann_file=config.data.val_ann,
        split=val_split,
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
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    return dataset, loader


def main() -> None:
    args = parse_args()
    config = load_config(
        args.config_file,
        overrides={"data": {"dataset_root": args.dataset_root}},
    )
    device = setup_device(config.runtime)
    set_seed(config.runtime.seed)

    dataset, loader = build_val_loader(config, args.dataset_root, args.num_workers)
    gt_by_image = decode_gt_masks(dataset)

    model = build_model(config).to(device)
    load_checkpoint(args.weights, model, strict=False)
    model.eval()

    best_all: List[float] = []
    best_top10: List[float] = []
    best_top1: List[float] = []
    scores_good: List[float] = []
    scores_bad: List[float] = []
    num_kept: List[int] = []

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            if idx >= args.max_images:
                break

            image_id = int(batch["image_ids"][0])
            gt_masks = gt_by_image.get(image_id, [])
            gt_masks = resize_bool_masks(gt_masks, args.fast_size)
            if len(gt_masks) == 0:
                continue

            images = batch["images"].to(device)
            depths = batch["depths"].to(device)
            raw_outputs = model.forward_inference_raw(images, depths)
            outputs = model._export_inference_predictions(
                raw_outputs,
                include_raw_tensors=False,
            )
            pred = outputs.get("predictions", [{}])[0]

            pred_scores = np.asarray(pred.get("scores", []), dtype=np.float32)
            pred_probs = np.asarray(pred.get("masks", []), dtype=np.float32)
            pred_masks = (pred_probs > 0.5).astype(np.uint8)
            if args.fast_size > 0 and pred_masks.size > 0:
                import cv2

                resized = []
                for pm in pred_masks:
                    rs = cv2.resize(
                        pm,
                        (args.fast_size, args.fast_size),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    resized.append(rs.astype(bool))
                pred_masks = np.stack(resized, axis=0)
            else:
                pred_masks = pred_masks.astype(bool)

            if pred_scores.size == 0 or pred_masks.size == 0:
                best_all.append(0.0)
                best_top10.append(0.0)
                best_top1.append(0.0)
                num_kept.append(0)
                continue

            order = np.argsort(-pred_scores)
            pred_scores = pred_scores[order]
            pred_masks = pred_masks[order]

            per_gt_all = []
            per_gt_top10 = []
            per_gt_top1 = []
            for gt in gt_masks:
                ious = np.asarray([mask_iou(gt, pm) for pm in pred_masks], dtype=np.float32)
                per_gt_all.append(float(ious.max()))
                per_gt_top10.append(float(ious[:10].max()))
                per_gt_top1.append(float(ious[0]))

            best_all.append(float(np.mean(per_gt_all)))
            best_top10.append(float(np.mean(per_gt_top10)))
            best_top1.append(float(np.mean(per_gt_top1)))

            max_iou_per_pred = []
            for pm in pred_masks:
                max_iou_per_pred.append(max(mask_iou(pm, gt) for gt in gt_masks))
            max_iou_per_pred = np.asarray(max_iou_per_pred, dtype=np.float32)

            scores_good.extend(pred_scores[max_iou_per_pred >= 0.5].tolist())
            scores_bad.extend(pred_scores[max_iou_per_pred < 0.1].tolist())
            num_kept.append(int((pred_scores >= args.score_thr).sum()))

    best_all_arr = np.asarray(best_all, dtype=np.float32)
    best_top10_arr = np.asarray(best_top10, dtype=np.float32)
    best_top1_arr = np.asarray(best_top1, dtype=np.float32)

    print("=" * 72)
    print("Validation Ranking Analysis")
    print("=" * 72)
    print(f"images_analyzed: {len(best_all)}")
    print(f"mean_best_iou_all:   {float(best_all_arr.mean()):.6f}")
    print(f"mean_best_iou_top10: {float(best_top10_arr.mean()):.6f}")
    print(f"mean_best_iou_top1:  {float(best_top1_arr.mean()):.6f}")
    print(f"ratio(best_all>=0.5):   {float((best_all_arr >= 0.5).mean()):.6f}")
    print(f"ratio(best_top10>=0.5): {float((best_top10_arr >= 0.5).mean()):.6f}")
    print(f"ratio(best_top1>=0.5):  {float((best_top1_arr >= 0.5).mean()):.6f}")
    if len(num_kept) > 0:
        nk = np.asarray(num_kept, dtype=np.int32)
        print(
            f"num_preds(score>={args.score_thr:.2f}) mean/min/max: "
            f"{float(nk.mean()):.3f}/{int(nk.min())}/{int(nk.max())}"
        )
    print(
        "good_scores(iou>=0.5) n/mean: "
        f"{len(scores_good)}/{(float(np.mean(scores_good)) if len(scores_good) else None)}"
    )
    print(
        "bad_scores(iou<0.1) n/mean:  "
        f"{len(scores_bad)}/{(float(np.mean(scores_bad)) if len(scores_bad) else None)}"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
