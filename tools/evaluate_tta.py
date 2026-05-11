#!/usr/bin/env python3
"""
Test-Time Augmentation evaluation for MagFormer with WBF support.

Usage:
    python tools/evaluate_tta_wbf.py --config configs/xxx.yaml \
        --checkpoint output/.../model_best.pth \
        --tta-scales 0.75 1.0 1.25 --tta-hflip \
        --merge-method wbf --wbf-iou 0.55 --wbf-conf-type max --wbf-overflow
"""

import argparse
import os
import sys
import time
import gc
from contextlib import nullcontext
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
    parser = argparse.ArgumentParser(description="TTA Evaluation for MagFormer (with WBF)")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tta-scales", type=float, nargs="+", default=[1.0])
    parser.add_argument("--tta-hflip", action="store_true")
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--max-preds", type=int, default=200)
    parser.add_argument("--score-thresh", type=float, default=0.03)
    parser.add_argument("--mask-thresh", type=float, default=0.5)
    parser.add_argument("--iou-types", nargs="+", default=["bbox", "segm"], choices=["bbox", "segm"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--report-interval", type=int, default=50)
    # Ensemble: additional checkpoint paths
    parser.add_argument("--ensemble-checkpoints", type=str, nargs="*", default=[])
    parser.add_argument("--ensemble-configs", type=str, nargs="*", default=[])
    # WBF merge options
    parser.add_argument("--merge-method", type=str, default="nms", choices=["nms", "wbf"])
    parser.add_argument("--wbf-iou", type=float, default=0.55)
    parser.add_argument("--wbf-skip-thr", type=float, default=0.0)
    # WBF v2: fix score destruction
    parser.add_argument("--wbf-conf-type", type=str, default="max",
                        choices=["avg", "max", "and", "or"],
                        help="WBF confidence type. 'max' keeps highest score from any aug (fixes score drop).")
    parser.add_argument("--wbf-overflow", action="store_true", default=True,
                        help="Allow WBF scores > 1.0 when merging (prevents score compression).")
    parser.add_argument("--no-wbf-overflow", dest="wbf_overflow", action="store_false",
                        help="Disallow score overflow (original behavior).")
    return parser.parse_args()


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
    print(f"[TTA] Loaded: {checkpoint_path}")
    print(f"[TTA] Missing: {len(missing)}, Unexpected: {len(unexpected)}")

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


def masks_to_bboxes_vectorized(masks, threshold=0.5):
    """Vectorized bbox extraction from probability masks. Returns (N, 4) xyxy."""
    binary = (masks > threshold).float()
    N = masks.shape[0]
    if N == 0:
        return masks.new_zeros((0, 4))

    H, W = masks.shape[-2], masks.shape[-1]
    row_any = binary.sum(dim=2) > 0
    col_any = binary.sum(dim=1) > 0

    bboxes = masks.new_zeros((N, 4))
    for i in range(N):
        if row_any[i].any():
            rows = torch.where(row_any[i])[0]
            cols = torch.where(col_any[i])[0]
            bboxes[i, 0] = cols[0].float()
            bboxes[i, 1] = rows[0].float()
            bboxes[i, 2] = cols[-1].float()
            bboxes[i, 3] = rows[-1].float()
    return bboxes


def nms_merge(all_scores, all_masks, all_cats, iou_threshold=0.5, max_preds=200):
    if len(all_scores) == 0:
        H, W = all_masks.shape[-2], all_masks.shape[-1] if all_masks.numel() > 0 else (1, 1)
        return {"scores": torch.tensor([]), "masks": torch.zeros((0, H, W)),
                "category_ids": torch.tensor([], dtype=torch.long)}

    bboxes = masks_to_bboxes_vectorized(all_masks)
    keep_indices = []
    unique_cats = all_cats.unique()

    for cat_id in unique_cats:
        cat_mask = (all_cats == cat_id)
        cat_indices = torch.where(cat_mask)[0]
        if len(cat_indices) == 0:
            continue
        cat_bboxes = bboxes[cat_indices]
        cat_scores = all_scores[cat_indices]

        area = (cat_bboxes[:, 2] - cat_bboxes[:, 0]) * (cat_bboxes[:, 3] - cat_bboxes[:, 1])
        valid = area > 0
        if valid.sum() == 0:
            continue

        valid_indices = cat_indices[valid]
        valid_bboxes = cat_bboxes[valid]
        valid_scores = all_scores[valid]

        keep = torchvision_nms(valid_bboxes, valid_scores, iou_threshold)
        keep_indices.extend(valid_indices[keep].tolist())

    if len(keep_indices) == 0:
        return {"scores": torch.tensor([]),
                "masks": torch.zeros((0, all_masks.shape[-2], all_masks.shape[-1])),
                "category_ids": torch.tensor([], dtype=torch.long)}

    keep_indices = torch.tensor(keep_indices, device=all_scores.device)
    sorted_idx = all_scores[keep_indices].argsort(descending=True)
    keep_indices = keep_indices[sorted_idx[:max_preds]]

    return {
        "scores": all_scores[keep_indices],
        "masks": all_masks[keep_indices],
        "category_ids": all_cats[keep_indices],
    }


def wbf_merge(all_scores_list, all_masks_list, all_cats_list,
              img_h, img_w, iou_thr=0.55, skip_box_thr=0.0, max_preds=200,
              conf_type="max", allows_overflow=True):
    """Weighted Box Fusion for TTA predictions.

    Args:
        all_scores_list: list of tensors, one per augmentation
        all_masks_list: list of tensors, one per augmentation
        all_cats_list: list of tensors, one per augmentation
        img_h, img_w: original image dimensions
        conf_type: WBF confidence merge strategy ('max' preserves highest score)
        allows_overflow: If True, allow merged scores > 1.0 (prevents compression)
    """
    from ensemble_boxes import weighted_boxes_fusion

    # Filter empty augmentations
    non_empty = [(s, m, c) for s, m, c in zip(all_scores_list, all_masks_list, all_cats_list) if len(s) > 0]

    if len(non_empty) == 0:
        return {"scores": torch.tensor([]), "masks": torch.zeros((0, img_h, img_w)), "category_ids": torch.tensor([], dtype=torch.long)}

    if len(non_empty) == 1:
        scores, masks, cats = non_empty[0]
        sorted_idx = scores.argsort(descending=True)[:max_preds]
        return {"scores": scores[sorted_idx], "masks": masks[sorted_idx], "category_ids": cats[sorted_idx]}

    boxes_list = []
    scores_list = []
    labels_list = []
    all_masks_flat = []

    for aug_idx, (scores, masks, cats) in enumerate(non_empty):
        bboxes = masks_to_bboxes_vectorized(masks)
        normed = bboxes.float().clone()
        normed[:, [0, 2]] /= img_w
        normed[:, [1, 3]] /= img_h
        normed = normed.clamp(0.0, 1.0)

        # Filter zero-area boxes
        area = (normed[:, 2] - normed[:, 0]) * (normed[:, 3] - normed[:, 1])
        valid = area > 1e-6
        if valid.sum() == 0:
            boxes_list.append(np.zeros((0, 4), dtype=np.float32))
            scores_list.append(np.zeros((0,), dtype=np.float32))
            labels_list.append(np.zeros((0,), dtype=np.float32))
            all_masks_flat.append(torch.zeros((0, img_h, img_w)))
            continue

        boxes_list.append(normed[valid].numpy())
        scores_list.append(scores[valid].numpy())
        labels_list.append(cats[valid].numpy().astype(np.float32))
        all_masks_flat.append(masks[valid])

    # Check if any boxes remain
    total_boxes = sum(len(b) for b in boxes_list)
    if total_boxes == 0:
        return {"scores": torch.tensor([]), "masks": torch.zeros((0, img_h, img_w)), "category_ids": torch.tensor([], dtype=torch.long)}

    fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
        boxes_list, scores_list, labels_list,
        weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr,
        conf_type=conf_type, allows_overflow=allows_overflow,
    )

    fused_scores = torch.from_numpy(fused_scores).float()
    fused_labels = torch.from_numpy(fused_labels).long()
    fused_boxes = torch.from_numpy(fused_boxes).float()

    # Clamp scores to [0, 1] after fusion (overflow allows higher during merge for ranking)
    fused_scores = fused_scores.clamp(0.0, 1.0)

    # Denormalize
    fused_boxes[:, [0, 2]] *= img_w
    fused_boxes[:, [1, 3]] *= img_h

    # Sort and take top-k
    sorted_idx = fused_scores.argsort(descending=True)[:max_preds]
    fused_scores = fused_scores[sorted_idx]
    fused_labels = fused_labels[sorted_idx]
    fused_boxes = fused_boxes[sorted_idx]

    # Create masks from fused bboxes (for bbox eval, exact mask doesn't matter)
    fused_masks = torch.zeros((len(sorted_idx), img_h, img_w))
    for i in range(len(sorted_idx)):
        x1, y1, x2, y2 = fused_boxes[i].int().tolist()
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w, x2 + 1), min(img_h, y2 + 1)
        if x2 > x1 and y2 > y1:
            fused_masks[i, y1:y2, x1:x2] = 1.0

    return {"scores": fused_scores, "masks": fused_masks, "category_ids": fused_labels}


@torch.no_grad()
def run_single_aug(model, images, depths, device, amp_enabled):
    """Run forward_inference_raw for a single augmented view."""
    if amp_enabled and device.type == "cuda":
        with autocast("cuda"):
            outputs = model.forward_inference_raw(images, depths, include_raw_tensors=False)
    else:
        outputs = model.forward_inference_raw(images, depths, include_raw_tensors=False)
    return outputs


@torch.no_grad()
def tta_inference_single_image(model, images, depths, scales, hflip, device,
                                amp_enabled, nms_iou, max_preds, score_thresh,
                                original_h, original_w,
                                merge_method="nms", wbf_iou=0.55, wbf_skip_thr=0.0,
                                wbf_conf_type="max", wbf_overflow=True):
    """Run TTA inference and merge via NMS or WBF."""
    aug_scores_list = []
    aug_masks_list = []
    aug_cats_list = []

    for scale in scales:
        for do_flip in ([False, True] if hflip else [False]):
            aug_images = images
            aug_depths = depths

            if scale != 1.0:
                new_h = int(round(original_h * scale))
                new_w = int(round(original_w * scale))
                aug_images = F.interpolate(aug_images, size=(new_h, new_w),
                                           mode='bilinear', align_corners=False)
                aug_depths = F.interpolate(aug_depths, size=(new_h, new_w),
                                           mode='nearest')

            if do_flip:
                aug_images = torch.flip(aug_images, [-1])
                aug_depths = torch.flip(aug_depths, [-1])

            outputs = run_single_aug(model, aug_images, aug_depths, device, amp_enabled)

            pred = outputs["predictions"][0]
            scores = pred["scores"]
            masks = pred["masks"]
            cats = pred["category_ids"]

            scores = scores.to(device)
            masks = masks.to(device)
            cats = cats.to(device)

            if masks.shape[-2:] != (original_h, original_w):
                masks = F.interpolate(masks.unsqueeze(1).float(),
                                      size=(original_h, original_w),
                                      mode='bilinear', align_corners=False).squeeze(1)

            if do_flip:
                masks = torch.flip(masks, [-1])

            keep = scores > score_thresh
            if keep.sum() > 0:
                aug_scores_list.append(scores[keep].cpu())
                aug_masks_list.append(masks[keep].cpu())
                aug_cats_list.append(cats[keep].cpu())
            else:
                aug_scores_list.append(torch.tensor([]))
                aug_masks_list.append(torch.zeros((0, original_h, original_w)))
                aug_cats_list.append(torch.tensor([], dtype=torch.long))

            del outputs
            torch.cuda.empty_cache()

    # Merge
    if merge_method == "wbf":
        return wbf_merge(aug_scores_list, aug_masks_list, aug_cats_list,
                         img_h=original_h, img_w=original_w,
                         iou_thr=wbf_iou, skip_box_thr=wbf_skip_thr, max_preds=max_preds,
                         conf_type=wbf_conf_type, allows_overflow=wbf_overflow)
    else:
        # NMS: concatenate all aug predictions then do NMS (original behavior)
        all_scores = torch.cat([s for s in aug_scores_list if len(s) > 0]) if any(len(s) > 0 for s in aug_scores_list) else torch.tensor([])
        all_masks = torch.cat([m for m in aug_masks_list if m.numel() > 0]) if any(m.numel() > 0 for m in aug_masks_list) else torch.zeros((0, original_h, original_w))
        all_cats = torch.cat([c for c in aug_cats_list if len(c) > 0]) if any(len(c) > 0 for c in aug_cats_list) else torch.tensor([], dtype=torch.long)
        return nms_merge(all_scores, all_masks, all_cats, iou_threshold=nms_iou, max_preds=max_preds)


def main():
    args = parse_args()
    if args.no_amp:
        args.amp = False

    device = torch.device(args.device)
    print(f"[TTA] Device: {device}")
    print(f"[TTA] Scales: {args.tta_scales}")
    print(f"[TTA] HFlip: {args.tta_hflip}")
    print(f"[TTA] NMS IoU: {args.nms_iou}")
    print(f"[TTA] Merge: {args.merge_method}")
    if args.merge_method == "wbf":
        print(f"[TTA] WBF IoU: {args.wbf_iou}, Skip thr: {args.wbf_skip_thr}")
        print(f"[TTA] WBF conf_type: {args.wbf_conf_type}, allows_overflow: {args.wbf_overflow}")

    config = load_config(args.config)
    model = build_model_from_config(config, args.checkpoint, device)

    # Load ensemble models if specified
    ensemble_models = []
    if args.ensemble_checkpoints:
        for i, (ckpt, cfg) in enumerate(zip(args.ensemble_checkpoints,
                                             args.ensemble_configs or [args.config] * len(args.ensemble_checkpoints))):
            if cfg != args.config:
                ensemble_cfg = load_config(cfg)
            else:
                ensemble_cfg = config
            ensemble_models.append(build_model_from_config(ensemble_cfg, ckpt, device))
            print(f"[TTA] Ensemble model {i}: {ckpt}")

    dataset = build_eval_dataset(config)
    print(f"[TTA] Dataset: {len(dataset)} images")

    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=4, pin_memory=True, collate_fn=collate_fn)

    from pycocotools.coco import COCO
    ann_path = Path(config.data.dataset_root) / config.data.val_ann
    coco_gt = COCO(str(ann_path))

    evaluator = COCOEvaluator(coco_gt=coco_gt, iou_types=args.iou_types, max_dets=100)

    total_augs = len(args.tta_scales) * (2 if args.tta_hflip else 1)
    total_passes = total_augs * (1 + len(ensemble_models))
    print(f"\n[TTA] Starting eval: {len(dataset)} images, {total_augs} augs, {total_passes} total passes")

    start_time = time.time()

    for batch_idx, batch in enumerate(loader):
        images = batch["images"].to(device)
        depths = batch["depths"].to(device)
        image_ids = batch.get("image_ids", [batch_idx])
        original_h, original_w = images.shape[-2], images.shape[-1]

        # Primary model TTA
        merged_pred = tta_inference_single_image(
            model, images, depths,
            scales=args.tta_scales, hflip=args.tta_hflip,
            device=device, amp_enabled=args.amp,
            nms_iou=args.nms_iou, max_preds=args.max_preds,
            score_thresh=args.score_thresh,
            original_h=original_h, original_w=original_w,
            merge_method=args.merge_method, wbf_iou=args.wbf_iou, wbf_skip_thr=args.wbf_skip_thr,
            wbf_conf_type=args.wbf_conf_type, wbf_overflow=args.wbf_overflow,
        )

        # Ensemble models (same TTA)
        for ens_model in ensemble_models:
            ens_pred = tta_inference_single_image(
                ens_model, images, depths,
                scales=args.tta_scales, hflip=args.tta_hflip,
                device=device, amp_enabled=args.amp,
                nms_iou=args.nms_iou, max_preds=args.max_preds,
                score_thresh=args.score_thresh,
                original_h=original_h, original_w=original_w,
                merge_method=args.merge_method, wbf_iou=args.wbf_iou, wbf_skip_thr=args.wbf_skip_thr,
                wbf_conf_type=args.wbf_conf_type, wbf_overflow=args.wbf_overflow,
            )
            # Merge ensemble predictions into primary
            if len(merged_pred["scores"]) > 0 or len(ens_pred["scores"]) > 0:
                merged_pred = nms_merge(
                    torch.cat([merged_pred["scores"], ens_pred["scores"]]),
                    torch.cat([merged_pred["masks"], ens_pred["masks"]]),
                    torch.cat([merged_pred["category_ids"], ens_pred["category_ids"]]),
                    iou_threshold=args.nms_iou, max_preds=args.max_preds,
                )

        # Convert to COCO format
        coco_preds = predictions_to_coco_instances(
            predictions=[merged_pred],
            image_ids=image_ids,
            score_threshold=args.score_thresh,
            mask_threshold=args.mask_thresh,
            category_offset=1,
        )
        evaluator.update(coco_preds)

        if (batch_idx + 1) % args.report_interval == 0:
            elapsed = time.time() - start_time
            rate = (batch_idx + 1) / elapsed
            eta = (len(dataset) - batch_idx - 1) / rate
            print(f"[TTA] {batch_idx + 1}/{len(dataset)} ({rate:.1f} img/s, "
                  f"{elapsed:.0f}s elapsed, ETA {eta:.0f}s)")

        del images, depths, merged_pred
        torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("TTA Evaluation Results")
    print("=" * 60)
    metrics = evaluator.summarize()

    elapsed = time.time() - start_time
    print(f"\n[TTA] Total: {elapsed:.1f}s ({elapsed/60:.1f}min)")

    output_dir = args.output_dir or str(Path(args.checkpoint).parent)
    if args.merge_method == "wbf":
        results_filename = f"tta_wbf_{args.wbf_conf_type}_iou{args.wbf_iou}_coco_results.json"
    else:
        results_filename = "tta_coco_results.json"
    coco_path = evaluator.dump(Path(output_dir) / results_filename)
    print(f"[TTA] Saved to {coco_path}")

    return metrics


if __name__ == "__main__":
    main()
