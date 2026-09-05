#!/usr/bin/env python3
"""
Slicing Aided Hyper Inference (SAHI) evaluation for MagFormer.

Slices large images into overlapping tiles, runs inference on each tile,
then merges results with NMS. This gives small objects more pixels at
inference time, improving AP_small significantly.

Usage:
    python tools/evaluate_sahi.py \
        --config configs/finetune_1k_full_1024_v14.yaml \
        --checkpoint output/.../checkpoint_best.pth \
        --tile-size 640 --overlap 200 --nms-iou 0.5
"""

import argparse
import os
import sys
import time
import gc
from pathlib import Path

sys.path.insert(0, str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader
from torchvision.ops import nms as torchvision_nms

from magformer.config import load_config
from magformer.data import CocoRgbdDataset
from magformer.data.transforms import RGBDTransform
from magformer.data.collate import collate_fn
from magformer.engine.coco_export import predictions_to_coco_instances
from magformer.engine.evaluator import COCOEvaluator


def parse_args():
    parser = argparse.ArgumentParser(
        description="SAHI (Slicing Aided Hyper Inference) Evaluation for MagFormer"
    )
    parser.add_argument("--config", required=True, help="Path to config yaml")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--tile-size", type=int, default=640,
                        help="Tile size in pixels (default: 640)")
    parser.add_argument("--overlap", type=int, default=200,
                        help="Overlap between adjacent tiles in pixels (default: 200)")
    parser.add_argument("--nms-iou", type=float, default=0.5,
                        help="NMS IoU threshold for merging tile predictions (default: 0.5)")
    parser.add_argument("--score-thresh", type=float, default=0.0,
                        help="Score threshold for pre-filtering tile predictions (default: 0.0)")
    parser.add_argument("--export-score-thresh", type=float, default=0.0,
                        help="Score threshold for final COCO export (default: 0.0)")
    parser.add_argument("--mask-thresh", type=float, default=0.5,
                        help="Mask binarization threshold (default: 0.5)")
    parser.add_argument("--max-preds", type=int, default=300,
                        help="Max predictions per image after merging (default: 300)")
    parser.add_argument("--iou-types", nargs="+", default=["bbox", "segm"],
                        choices=["bbox", "segm"],
                        help="COCO IoU types to evaluate (default: bbox segm)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory for results (default: checkpoint dir)")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device for inference (default: cuda:0)")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Limit number of images for quick testing")
    parser.add_argument("--no-full-image", action="store_true",
                        help="Skip running inference on the full image (tiles only)")
    parser.add_argument("--amp", action="store_true", default=True,
                        help="Enable AMP (default: True)")
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable AMP")
    parser.add_argument("--report-interval", type=int, default=50,
                        help="Print progress every N images")
    parser.add_argument("--empty-cache-interval", type=int, default=0,
                        help="Call torch.cuda.empty_cache() every N images. 0 disables.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model loading (same pattern as evaluate_tta.py)
# ---------------------------------------------------------------------------

def build_model_from_config(config, checkpoint_path, device):
    from magformer.models import build_model as _build_model
    from magformer.engine.utils import load_torch_checkpoint

    model = _build_model(config)
    ckpt = load_torch_checkpoint(checkpoint_path, map_location="cpu")
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[k[7:] if k.startswith("module.") else k] = v

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    print(f"[SAHI] Loaded: {checkpoint_path}")
    print(f"[SAHI] Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    model = model.to(device)
    model.eval()
    return model


def build_eval_dataset(config):
    data_cfg = config.data
    val_split = getattr(data_cfg, "val_split", "val")
    dataset = CocoRgbdDataset(
        dataset_root=data_cfg.dataset_root,
        ann_file=data_cfg.val_ann,
        split=val_split,
        transform=None,
        is_train=False,
    )
    eval_transform = RGBDTransform(
        image_size=data_cfg.image_size,
        min_scale=data_cfg.min_scale,
        max_scale=data_cfg.max_scale,
        random_flip="none",
        rgb_brightness=0.0, rgb_contrast=0.0, rgb_saturation=0.0, rgb_hue=0.0,
        depth_scale=data_cfg.depth.scale, depth_shift=data_cfg.depth.shift,
        depth_clip_min=data_cfg.depth.clip_min, depth_clip_max=data_cfg.depth.clip_max,
        depth_norm=data_cfg.depth.norm,
        depth_per_sample_norm=getattr(data_cfg.depth, "per_sample_norm", True),
        is_train=False,
    )
    dataset.transform = eval_transform
    return dataset


# ---------------------------------------------------------------------------
# Slicing utilities
# ---------------------------------------------------------------------------

def compute_slice_positions(img_h, img_w, tile_size, overlap):
    """Compute (y_start, y_end, x_start, x_end) for each tile.

    Returns a list of 4-tuples (y_start, y_end, x_start, x_end).
    The last tile in each dimension is adjusted so it covers the edge.
    """
    positions = []
    step = tile_size - overlap

    y_starts = list(range(0, img_h - overlap, step))
    if not y_starts or y_starts[-1] + tile_size < img_h:
        y_starts.append(max(0, img_h - tile_size))
    # Deduplicate while preserving order
    seen = set()
    unique_y = []
    for y in y_starts:
        if y not in seen:
            seen.add(y)
            unique_y.append(y)
    y_starts = unique_y

    x_starts = list(range(0, img_w - overlap, step))
    if not x_starts or x_starts[-1] + tile_size < img_w:
        x_starts.append(max(0, img_w - tile_size))
    seen = set()
    unique_x = []
    for x in x_starts:
        if x not in seen:
            seen.add(x)
            unique_x.append(x)
    x_starts = unique_x

    for y in y_starts:
        for x in x_starts:
            y_end = min(y + tile_size, img_h)
            x_end = min(x + tile_size, img_w)
            positions.append((y, y_end, x, x_end))

    return positions


def extract_tile(tensor, y_start, y_end, x_start, x_end):
    """Extract a spatial tile from a 4D (B,C,H,W) or 3D (C,H,W) tensor."""
    if tensor.ndim == 4:
        return tensor[:, :, y_start:y_end, x_start:x_end]
    return tensor[:, y_start:y_end, x_start:x_end]


# ---------------------------------------------------------------------------
# Inference on a single tile
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_tile_inference(model, tile_images, tile_depths, device, amp_enabled):
    """Run forward_inference_raw on one tile. Returns raw prediction dict."""
    if amp_enabled and device.type == "cuda":
        with autocast("cuda"):
            outputs = model.forward_inference_raw(
                tile_images, tile_depths,
                include_raw_tensors=True,
                move_predictions_to_cpu=False,
            )
    else:
        outputs = model.forward_inference_raw(
            tile_images, tile_depths,
            include_raw_tensors=True,
            move_predictions_to_cpu=False,
        )
    return outputs


# ---------------------------------------------------------------------------
# Remap tile predictions back to full-image coordinates
# ---------------------------------------------------------------------------

def remap_tile_predictions(pred, y_off, x_off, tile_h, tile_w,
                           full_h, full_w, score_thresh, device):
    """Take predictions from a single tile and remap masks/bboxes to full image.

    Args:
        pred: prediction dict for one image (from outputs["predictions"][0])
        y_off, x_off: tile offset in full image
        tile_h, tile_w: tile spatial dimensions
        full_h, full_w: full image spatial dimensions
        score_thresh: minimum score to keep
        device: torch device

    Returns:
        Dict with scores, masks (full image size), category_ids
    """
    scores = pred["scores"]
    cats = pred["category_ids"]

    # Use raw mask logits for better quality remapping
    masks_are_logits = "mask_logits" in pred
    if masks_are_logits:
        masks = pred["mask_logits"]
    elif "mask_probs" in pred:
        masks = pred["mask_probs"]
    else:
        masks = pred["masks"].float()

    # Filter by score early
    keep = scores > score_thresh
    if keep.sum() == 0:
        return {
            "scores": scores.new_empty((0,), device=device),
            "masks": scores.new_zeros((0, full_h, full_w), device=device),
            "category_ids": cats.new_empty((0,), dtype=torch.long, device=device),
        }

    scores = scores[keep].to(device)
    cats = cats[keep].to(device)
    masks = masks[keep].to(device)

    # masks shape: (N, tile_h, tile_w) -> resize to full image and shift
    N = masks.shape[0]
    if N == 0:
        return {
            "scores": scores,
            "masks": scores.new_zeros((0, full_h, full_w)),
            "category_ids": cats,
        }

    # Place each mask into full-image canvas
    full_masks = scores.new_zeros((N, full_h, full_w), device=device)

    # Crop masks to tile dimensions (avoids blending padded pixels)
    if masks.shape[-2:] != (tile_h, tile_w):
        masks = masks[:, :tile_h, :tile_w]

    # For logits, sigmoid first
    if masks_are_logits:
        masks = masks.sigmoid()

    # Compute valid region (clipped to full image bounds)
    y1 = y_off
    x1 = x_off
    y2 = min(y_off + tile_h, full_h)
    x2 = min(x_off + tile_w, full_w)

    # Also clip the mask if tile extends beyond image
    mask_y1 = 0
    mask_x1 = 0
    mask_y2 = tile_h
    mask_x2 = tile_w

    if y2 - y1 < tile_h:
        mask_y2 = y2 - y1
    if x2 - x1 < tile_w:
        mask_x2 = x2 - x1

    full_masks[:, y1:y2, x1:x2] = masks[:, mask_y1:mask_y2, mask_x1:mask_x2]

    return {
        "scores": scores,
        "masks": full_masks,
        "category_ids": cats,
    }


# ---------------------------------------------------------------------------
# NMS merge
# ---------------------------------------------------------------------------

def masks_to_bboxes_vectorized(masks, threshold=0.5):
    """Vectorized bbox extraction from probability masks. Returns (N, 4) xyxy."""
    binary = (masks > threshold).float()
    N = masks.shape[0]
    if N == 0:
        return masks.new_zeros((0, 4))

    bboxes = masks.new_zeros((N, 4))
    for i in range(N):
        rows = (binary[i].sum(dim=1) > 0)
        cols = (binary[i].sum(dim=0) > 0)
        if rows.any():
            row_idx = torch.where(rows)[0]
            col_idx = torch.where(cols)[0]
            bboxes[i, 0] = col_idx[0].float()
            bboxes[i, 1] = row_idx[0].float()
            bboxes[i, 2] = col_idx[-1].float() + 1.0
            bboxes[i, 3] = row_idx[-1].float() + 1.0
    return bboxes


def nms_merge_predictions(all_scores, all_masks, all_cats,
                          iou_threshold=0.5, max_preds=300,
                          mask_threshold=0.5):
    """Merge predictions from multiple tiles using per-class bbox NMS.

    Uses bounding boxes derived from masks for fast NMS via torchvision.
    This is much faster than mask IoU NMS and works well for SAHI tile
    deduplication since the same object in overlapping tiles has similar bboxes.
    """
    if all_scores.numel() == 0:
        H, W = all_masks.shape[-2], all_masks.shape[-1] if all_masks.numel() > 0 else (1, 1)
        return {
            "scores": all_scores.new_empty((0,)),
            "masks": all_scores.new_zeros((0, H, W)),
            "category_ids": all_cats.new_empty((0,), dtype=torch.long),
        }

    N = all_scores.shape[0]
    H, W = all_masks.shape[-2], all_masks.shape[-1]

    # Downscale masks for fast bbox computation (128x128 is sufficient for bbox extraction)
    DS = 128
    scale_y = DS / H
    scale_x = DS / W
    small_masks = F.interpolate(
        all_masks.float().unsqueeze(1), size=(DS, DS), mode='bilinear', align_corners=False
    ).squeeze(1)
    binary_small = (small_masks > mask_threshold).float()

    # Vectorized bbox extraction from downscaled masks
    bboxes = torch.zeros((N, 4), device=all_scores.device, dtype=torch.float32)
    for i in range(N):
        rows = binary_small[i].sum(dim=1) > 0
        cols = binary_small[i].sum(dim=0) > 0
        if rows.any():
            row_idx = torch.where(rows)[0]
            col_idx = torch.where(cols)[0]
            bboxes[i, 0] = col_idx[0].float() / scale_x
            bboxes[i, 1] = row_idx[0].float() / scale_y
            bboxes[i, 2] = (col_idx[-1].float() + 1) / scale_x
            bboxes[i, 3] = (row_idx[-1].float() + 1) / scale_y

    keep_indices = []
    unique_cats = all_cats.unique()

    for cat_id in unique_cats:
        cat_mask = (all_cats == cat_id)
        cat_indices = torch.where(cat_mask)[0]
        if len(cat_indices) == 0:
            continue

        cat_scores = all_scores[cat_indices]
        cat_bboxes = bboxes[cat_indices]

        # Filter zero-area bboxes
        areas = (cat_bboxes[:, 2] - cat_bboxes[:, 0]) * (cat_bboxes[:, 3] - cat_bboxes[:, 1])
        valid = areas > 0
        if valid.sum() == 0:
            continue

        valid_indices = cat_indices[valid]
        valid_scores = cat_scores[valid]
        valid_bboxes = cat_bboxes[valid]

        # Fast bbox NMS using torchvision (GPU-accelerated)
        keep = torchvision_nms(valid_bboxes, valid_scores, iou_threshold)
        keep_indices.extend(valid_indices[keep].tolist())

    if len(keep_indices) == 0:
        return {
            "scores": all_scores.new_empty((0,)),
            "masks": all_scores.new_zeros((0, H, W)),
            "category_ids": all_cats.new_empty((0,), dtype=torch.long),
        }

    keep_indices = torch.tensor(keep_indices, device=all_scores.device)
    sorted_idx = all_scores[keep_indices].argsort(descending=True)
    keep_indices = keep_indices[sorted_idx[:max_preds]]

    return {
        "scores": all_scores[keep_indices],
        "masks": all_masks[keep_indices],
        "category_ids": all_cats[keep_indices],
    }


# ---------------------------------------------------------------------------
# SAHI inference for a single image
# ---------------------------------------------------------------------------

@torch.no_grad()
def sahi_inference_single_image(model, images, depths, device, amp_enabled,
                                tile_size, overlap, nms_iou, score_thresh,
                                max_preds, mask_thresh, include_full_image):
    """Run SAHI inference on a single image.

    Args:
        model: MagFormer model
        images: (1, C, H, W) tensor
        depths: (1, 1, H, W) tensor
        device: torch device
        amp_enabled: whether to use AMP
        tile_size: tile size in pixels
        overlap: overlap between tiles
        nms_iou: NMS IoU threshold
        score_thresh: pre-filter score threshold
        max_preds: max predictions after merge
        mask_thresh: mask binarization threshold
        include_full_image: whether to also run inference on full image

    Returns:
        Dict with merged scores, masks, category_ids
    """
    full_h, full_w = images.shape[-2], images.shape[-1]
    all_scores_list = []
    all_masks_list = []
    all_cats_list = []

    # --- Full image pass (optional) ---
    if include_full_image:
        full_outputs = run_tile_inference(model, images, depths, device, amp_enabled)
        full_pred = full_outputs["predictions"][0]
        remapped = remap_tile_predictions(
            full_pred, 0, 0, full_h, full_w, full_h, full_w,
            score_thresh, device
        )
        if remapped["scores"].numel() > 0:
            all_scores_list.append(remapped["scores"])
            all_masks_list.append(remapped["masks"])
            all_cats_list.append(remapped["category_ids"])
        del full_outputs

    # --- Tile passes ---
    # Only slice if the image is larger than tile_size in at least one dimension
    if full_h > tile_size or full_w > tile_size:
        positions = compute_slice_positions(full_h, full_w, tile_size, overlap)
        # Pad images/depths to tile_size if needed for edge tiles
        for (y_start, y_end, x_start, x_end) in positions:
            tile_h = y_end - y_start
            tile_w = x_end - x_start

            tile_img = extract_tile(images, y_start, y_end, x_start, x_end)
            tile_dep = extract_tile(depths, y_start, y_end, x_start, x_end)

            # Pad to tile_size x tile_size if tile is smaller
            if tile_h < tile_size or tile_w < tile_size:
                pad_h = tile_size - tile_h
                pad_w = tile_size - tile_w
                tile_img = F.pad(tile_img, (0, pad_w, 0, pad_h), mode="constant", value=0)
                tile_dep = F.pad(tile_dep, (0, pad_w, 0, pad_h), mode="constant", value=0)

            tile_outputs = run_tile_inference(model, tile_img, tile_dep, device, amp_enabled)
            tile_pred = tile_outputs["predictions"][0]

            remapped = remap_tile_predictions(
                tile_pred, y_start, x_start, tile_h, tile_w,
                full_h, full_w, score_thresh, device
            )
            if remapped["scores"].numel() > 0:
                all_scores_list.append(remapped["scores"])
                all_masks_list.append(remapped["masks"])
                all_cats_list.append(remapped["category_ids"])

            del tile_outputs, tile_img, tile_dep
    else:
        # Image is smaller than tile_size, just run full image (already done above if enabled)
        if not include_full_image:
            # Single pass on the full image
            full_outputs = run_tile_inference(model, images, depths, device, amp_enabled)
            full_pred = full_outputs["predictions"][0]
            remapped = remap_tile_predictions(
                full_pred, 0, 0, full_h, full_w, full_h, full_w,
                score_thresh, device
            )
            if remapped["scores"].numel() > 0:
                all_scores_list.append(remapped["scores"])
                all_masks_list.append(remapped["masks"])
                all_cats_list.append(remapped["category_ids"])
            del full_outputs

    # --- Merge all predictions ---
    if not all_scores_list:
        return {
            "scores": torch.zeros((0,), device=device),
            "masks": torch.zeros((0, full_h, full_w), device=device),
            "category_ids": torch.empty((0,), dtype=torch.long, device=device),
        }

    # Merge on CPU to avoid GPU OOM with large masks
    all_scores = torch.cat(all_scores_list)
    all_masks = torch.cat(all_masks_list)
    all_cats = torch.cat(all_cats_list)

    merged = nms_merge_predictions(
        all_scores, all_masks, all_cats,
        iou_threshold=nms_iou,
        max_preds=max_preds,
        mask_threshold=mask_thresh,
    )
    return merged


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    if args.no_amp:
        args.amp = False
    export_score_thresh = args.export_score_thresh if args.export_score_thresh is not None else args.score_thresh

    device = torch.device(args.device)
    print(f"[SAHI] Device: {device}")
    print(f"[SAHI] Tile size: {args.tile_size}, Overlap: {args.overlap}")
    print(f"[SAHI] NMS IoU: {args.nms_iou}")
    print(f"[SAHI] Score thresh: pre={args.score_thresh}, export={export_score_thresh}")
    print(f"[SAHI] Mask thresh: {args.mask_thresh}")
    print(f"[SAHI] Include full image: {not args.no_full_image}")
    print(f"[SAHI] IoU types: {args.iou_types}")
    print(f"[SAHI] Max preds: {args.max_preds}")

    config = load_config(args.config)
    model = build_model_from_config(config, args.checkpoint, device)
    dataset = build_eval_dataset(config)

    # Limit images if requested
    if args.max_images is not None and args.max_images < len(dataset):
        dataset.image_ids = dataset.image_ids[:args.max_images]
        print(f"[SAHI] Limited to {args.max_images} images")

    print(f"[SAHI] Dataset: {len(dataset)} images")

    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=4, pin_memory=True, collate_fn=collate_fn)

    from pycocotools.coco import COCO
    ann_path = Path(config.data.dataset_root) / config.data.val_ann
    coco_gt = COCO(str(ann_path))

    evaluator = COCOEvaluator(coco_gt=coco_gt, iou_types=args.iou_types, max_dets=100)

    # Estimate number of tiles per image for reporting
    img_size = config.data.image_size
    if img_size > args.tile_size:
        step = args.tile_size - args.overlap
        n_tiles_y = len(range(0, img_size - args.overlap, step)) + 1
        n_tiles_x = len(range(0, img_size - args.overlap, step)) + 1
        est_tiles = n_tiles_y * n_tiles_x + (0 if args.no_full_image else 1)
    else:
        est_tiles = 1
    print(f"[SAHI] Estimated tiles per image: {est_tiles}")
    print(f"\n[SAHI] Starting eval: {len(dataset)} images\n")

    start_time = time.time()
    total_exported = 0

    for batch_idx, batch in enumerate(loader):
        images = batch["images"].to(device)
        depths = batch["depths"].to(device)
        if "image_ids" not in batch:
            raise KeyError("Evaluation batch is missing required image_ids")
        image_ids = [int(image_id) for image_id in batch["image_ids"]]

        merged = sahi_inference_single_image(
            model, images, depths,
            device=device,
            amp_enabled=args.amp,
            tile_size=args.tile_size,
            overlap=args.overlap,
            nms_iou=args.nms_iou,
            score_thresh=args.score_thresh,
            max_preds=args.max_preds,
            mask_thresh=args.mask_thresh,
            include_full_image=not args.no_full_image,
        )

        # Convert to COCO format
        coco_preds = predictions_to_coco_instances(
            predictions=[merged],
            image_ids=image_ids,
            score_threshold=export_score_thresh,
            mask_threshold=args.mask_thresh,
            category_offset=1,
        )
        total_exported += len(coco_preds)
        evaluator.update(coco_preds, image_ids=image_ids)

        if (batch_idx + 1) % args.report_interval == 0:
            elapsed = time.time() - start_time
            rate = (batch_idx + 1) / elapsed
            eta = (len(dataset) - batch_idx - 1) / rate
            print(f"[SAHI] {batch_idx + 1}/{len(dataset)} "
                  f"({rate:.1f} img/s, {elapsed:.0f}s elapsed, ETA {eta:.0f}s, "
                  f"{total_exported} preds so far)", flush=True)

        del images, depths, merged
        if (args.empty_cache_interval > 0
                and device.type == "cuda"
                and (batch_idx + 1) % args.empty_cache_interval == 0):
            torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("SAHI Evaluation Results")
    print("=" * 60)
    metrics = evaluator.summarize()

    elapsed = time.time() - start_time
    print(f"\n[SAHI] Total: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"[SAHI] Exported {total_exported} predictions across {len(dataset)} images")

    output_dir = args.output or str(Path(args.checkpoint).parent)
    results_filename = f"sahi_ts{args.tile_size}_ol{args.overlap}_nms{args.nms_iou}_coco_results.json"
    coco_path = evaluator.dump(Path(output_dir) / results_filename)
    print(f"[SAHI] Saved to {coco_path}")

    return metrics


if __name__ == "__main__":
    main()
