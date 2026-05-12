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
    parser.add_argument("--pre-score-thresh", type=float, default=None,
                        help="Prediction score threshold before TTA merging. Defaults to --score-thresh.")
    parser.add_argument("--export-score-thresh", type=float, default=None,
                        help="Prediction score threshold for COCO export. Defaults to --score-thresh.")
    parser.add_argument("--mask-thresh", type=float, default=0.5)
    parser.add_argument("--cluster-mask-thresh", type=float, default=None,
                        help="Mask threshold used only for TTA mask-IoU clustering. Defaults to --mask-thresh.")
    parser.add_argument("--export-mask-thresh", type=float, default=None,
                        help="Mask threshold used for final COCO mask export. Defaults to --mask-thresh.")
    parser.add_argument("--merge-iou-mask-size", type=int, default=128,
                        help="Max side length used only for mask-IoU clustering.")
    parser.add_argument("--bbox-prefilter-iou", type=float, default=0.25,
                        help="BBox-IoU prefilter before expensive mask-IoU clustering.")
    parser.add_argument("--pre-merge-topk-factor", type=float, default=2.0,
                        help="Keep at most max_preds * factor candidates before mask clustering.")
    parser.add_argument("--iou-types", nargs="+", default=["bbox", "segm"], choices=["bbox", "segm"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--report-interval", type=int, default=50)
    parser.add_argument("--empty-cache-interval", type=int, default=0,
                        help="Call torch.cuda.empty_cache() every N images. 0 disables per-image cache clearing.")
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


def validate_args(args):
    if args.merge_method == "wbf" and "segm" in args.iou_types:
        raise ValueError(
            "WBF currently fuses boxes only and creates rectangular masks. "
            "Use --iou-types bbox or --merge-method nms for segmentation AP."
        )
    if args.ensemble_configs and len(args.ensemble_configs) != len(args.ensemble_checkpoints):
        raise ValueError(
            "--ensemble-configs must be omitted or have the same length as "
            "--ensemble-checkpoints."
        )


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

    missing, unexpected = model.load_state_dict(new_state_dict, strict=True)
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
            bboxes[i, 2] = cols[-1].float() + 1.0
            bboxes[i, 3] = rows[-1].float() + 1.0
    return bboxes


def _downsample_binary_masks(binary_masks, max_side):
    """Downsample binary masks for clustering only; final masks stay full resolution."""
    if max_side is None or int(max_side) <= 0:
        return binary_masks

    height, width = binary_masks.shape[-2:]
    largest_side = max(height, width)
    if largest_side <= int(max_side):
        return binary_masks

    scale = float(max_side) / float(largest_side)
    new_h = max(1, int(round(height * scale)))
    new_w = max(1, int(round(width * scale)))
    resized = F.adaptive_max_pool2d(
        binary_masks.float().unsqueeze(1),
        output_size=(new_h, new_w),
    ).squeeze(1)
    return resized >= 0.5


def _bbox_iou_against_seed(boxes, seed_box):
    if boxes.numel() == 0:
        return boxes.new_zeros((0,))

    x1 = torch.maximum(boxes[:, 0], seed_box[0])
    y1 = torch.maximum(boxes[:, 1], seed_box[1])
    x2 = torch.minimum(boxes[:, 2], seed_box[2])
    y2 = torch.minimum(boxes[:, 3], seed_box[3])
    inter = (x2 - x1).clamp_min(0) * (y2 - y1).clamp_min(0)

    box_area = (boxes[:, 2] - boxes[:, 0]).clamp_min(0) * (boxes[:, 3] - boxes[:, 1]).clamp_min(0)
    seed_area = (seed_box[2] - seed_box[0]).clamp_min(0) * (seed_box[3] - seed_box[1]).clamp_min(0)
    union = box_area + seed_area - inter
    return torch.where(union > 0, inter / union.clamp_min(1.0), torch.zeros_like(union))


def merge_soft_mask_clusters(scores, masks, category_ids, *,
                             iou_threshold, mask_threshold, max_preds,
                             iou_mask_size=128, bbox_prefilter_iou=0.25,
                             pre_merge_topk_factor=2.0,
                             input_is_logits=False):
    """Merge same-class TTA masks by mask-IoU clusters and score-weighted soft masks."""
    if masks.ndim == 2:
        masks = masks.unsqueeze(0)

    if masks.ndim >= 3:
        height, width = masks.shape[-2:]
    else:
        height, width = 1, 1

    if scores.numel() == 0 or masks.numel() == 0:
        return {
            "scores": scores.new_empty((0,)),
            "masks": scores.new_zeros((0, height, width)),
            "category_ids": category_ids.new_empty((0,), dtype=torch.long),
        }

    scores = scores.reshape(-1)
    category_ids = category_ids.reshape(-1).long()
    masks_float = masks.float()
    masks_for_iou = masks_float.sigmoid() if input_is_logits else masks_float

    if masks_float.shape[0] != scores.shape[0] or category_ids.shape[0] != scores.shape[0]:
        raise ValueError("scores, masks, and category_ids must contain the same number of predictions")

    if max_preds is not None and pre_merge_topk_factor is not None and float(pre_merge_topk_factor) > 0:
        candidate_limit = max(int(max_preds), int(round(int(max_preds) * float(pre_merge_topk_factor))))
        if scores.numel() > candidate_limit:
            keep = scores.argsort(descending=True)[:candidate_limit]
            scores = scores[keep]
            category_ids = category_ids[keep]
            masks_float = masks_float[keep]
            masks_for_iou = masks_for_iou[keep]

    binary_masks = masks_for_iou >= float(mask_threshold)
    cluster_masks = _downsample_binary_masks(binary_masks, iou_mask_size)
    cluster_bboxes = masks_to_bboxes_vectorized(cluster_masks.float(), threshold=0.5)
    merged_scores = []
    merged_masks = []
    merged_cats = []

    for cat_id in category_ids.unique():
        cat_indices = torch.where(category_ids == cat_id)[0]
        if cat_indices.numel() == 0:
            continue

        order = scores[cat_indices].argsort(descending=True)
        remaining = cat_indices[order]

        while remaining.numel() > 0:
            seed_index = remaining[0]
            bbox_ious = _bbox_iou_against_seed(cluster_bboxes[remaining], cluster_bboxes[seed_index])
            prefilter = bbox_ious >= float(bbox_prefilter_iou)
            prefilter[0] = True
            candidate_indices = remaining[prefilter]

            seed_mask = cluster_masks[seed_index].flatten()
            candidate_masks = cluster_masks[candidate_indices].flatten(1)

            intersections = (candidate_masks & seed_mask).sum(dim=1).float()
            unions = (candidate_masks | seed_mask).sum(dim=1).float()
            ious = torch.where(
                unions > 0,
                intersections / unions.clamp_min(1.0),
                torch.zeros_like(unions),
            )

            in_cluster = torch.zeros_like(prefilter)
            in_cluster[prefilter] = ious >= float(iou_threshold)
            in_cluster[0] = True
            cluster_indices = remaining[in_cluster]
            remaining = remaining[~in_cluster]

            weights = scores[cluster_indices].to(dtype=masks_float.dtype)
            weight_sum = weights.sum()
            if weight_sum.abs() <= torch.finfo(masks_float.dtype).eps:
                fused_mask = masks_float[cluster_indices].mean(dim=0)
            else:
                fused_mask = (
                    masks_float[cluster_indices] * weights.view(-1, 1, 1)
                ).sum(dim=0) / weight_sum
            if input_is_logits:
                fused_mask = fused_mask.sigmoid()

            merged_scores.append(scores[cluster_indices].max())
            merged_masks.append(fused_mask)
            merged_cats.append(cat_id)

    if not merged_scores:
        return {
            "scores": scores.new_empty((0,)),
            "masks": scores.new_zeros((0, height, width)),
            "category_ids": category_ids.new_empty((0,), dtype=torch.long),
        }

    out_scores = torch.stack(merged_scores)
    out_masks = torch.stack(merged_masks)
    out_cats = torch.stack(merged_cats).long()

    sorted_idx = out_scores.argsort(descending=True)
    if max_preds is not None:
        sorted_idx = sorted_idx[:max(0, int(max_preds))]

    return {
        "scores": out_scores[sorted_idx],
        "masks": out_masks[sorted_idx],
        "category_ids": out_cats[sorted_idx],
    }


def nms_merge(all_scores, all_masks, all_cats, iou_threshold=0.5, max_preds=200,
              mask_threshold=0.5, iou_mask_size=128, bbox_prefilter_iou=0.25,
              pre_merge_topk_factor=2.0, input_is_logits=False):
    return merge_soft_mask_clusters(
        all_scores,
        all_masks,
        all_cats,
        iou_threshold=iou_threshold,
        mask_threshold=mask_threshold,
        max_preds=max_preds,
        iou_mask_size=iou_mask_size,
        bbox_prefilter_iou=bbox_prefilter_iou,
        pre_merge_topk_factor=pre_merge_topk_factor,
        input_is_logits=input_is_logits,
    )


def _bbox_nms_merge(all_scores, all_masks, all_cats, iou_threshold=0.5, max_preds=200):
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
            outputs = model.forward_inference_raw(
                images,
                depths,
                include_raw_tensors=True,
                move_predictions_to_cpu=False,
            )
    else:
        outputs = model.forward_inference_raw(
            images,
            depths,
            include_raw_tensors=True,
            move_predictions_to_cpu=False,
        )
    return outputs


@torch.no_grad()
def tta_inference_single_image(model, images, depths, scales, hflip, device,
                                amp_enabled, nms_iou, max_preds, score_thresh,
                                original_h, original_w, mask_threshold=0.5,
                                merge_iou_mask_size=128, bbox_prefilter_iou=0.25,
                                pre_merge_topk_factor=2.0,
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
                                           mode='bilinear', align_corners=False)

            if do_flip:
                aug_images = torch.flip(aug_images, [-1])
                aug_depths = torch.flip(aug_depths, [-1])

            outputs = run_single_aug(model, aug_images, aug_depths, device, amp_enabled)

            pred = outputs["predictions"][0]
            scores = pred["scores"]
            masks_are_logits = "mask_logits" in pred
            masks = pred["mask_logits"] if masks_are_logits else (
                pred["mask_probs"] if "mask_probs" in pred else pred["masks"]
            )
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
            if merge_method == "wbf" and masks_are_logits:
                masks = masks.sigmoid()
                masks_are_logits = False

            keep = scores > score_thresh
            if keep.sum() > 0:
                aug_scores_list.append(scores[keep])
                aug_masks_list.append(masks[keep])
                aug_cats_list.append(cats[keep])
            else:
                aug_scores_list.append(scores.new_empty((0,)))
                aug_masks_list.append(masks.new_zeros((0, original_h, original_w)))
                aug_cats_list.append(cats.new_empty((0,), dtype=torch.long))

            del outputs

    # Merge
    if merge_method == "wbf":
        return wbf_merge([s.cpu() for s in aug_scores_list],
                         [m.cpu() for m in aug_masks_list],
                         [c.cpu() for c in aug_cats_list],
                         img_h=original_h, img_w=original_w,
                         iou_thr=wbf_iou, skip_box_thr=wbf_skip_thr, max_preds=max_preds,
                         conf_type=wbf_conf_type, allows_overflow=wbf_overflow)
    else:
        # NMS path for segmentation: cluster by mask IoU and keep score-weighted soft masks.
        all_scores = torch.cat([s for s in aug_scores_list if len(s) > 0]) if any(len(s) > 0 for s in aug_scores_list) else images.new_empty((0,))
        all_masks = torch.cat([m for m in aug_masks_list if m.numel() > 0]) if any(m.numel() > 0 for m in aug_masks_list) else images.new_zeros((0, original_h, original_w))
        all_cats = torch.cat([c for c in aug_cats_list if len(c) > 0]) if any(len(c) > 0 for c in aug_cats_list) else torch.empty((0,), dtype=torch.long, device=device)
        return nms_merge(
            all_scores,
            all_masks,
            all_cats,
            iou_threshold=nms_iou,
            max_preds=max_preds,
            mask_threshold=mask_threshold,
            iou_mask_size=merge_iou_mask_size,
            bbox_prefilter_iou=bbox_prefilter_iou,
            pre_merge_topk_factor=pre_merge_topk_factor,
            input_is_logits=masks_are_logits,
        )


def main():
    args = parse_args()
    validate_args(args)
    if args.no_amp:
        args.amp = False
    pre_score_thresh = args.pre_score_thresh if args.pre_score_thresh is not None else args.score_thresh
    export_score_thresh = args.export_score_thresh if args.export_score_thresh is not None else args.score_thresh
    cluster_mask_thresh = args.cluster_mask_thresh if args.cluster_mask_thresh is not None else args.mask_thresh
    export_mask_thresh = args.export_mask_thresh if args.export_mask_thresh is not None else args.mask_thresh

    device = torch.device(args.device)
    print(f"[TTA] Device: {device}")
    print(f"[TTA] Scales: {args.tta_scales}")
    print(f"[TTA] HFlip: {args.tta_hflip}")
    print(f"[TTA] NMS IoU: {args.nms_iou}")
    print(f"[TTA] Score thresh: pre={pre_score_thresh}, export={export_score_thresh}")
    print(f"[TTA] Mask thresh: cluster={cluster_mask_thresh}, export={export_mask_thresh}")
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
    total_exported_predictions = 0

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
            score_thresh=pre_score_thresh,
            original_h=original_h, original_w=original_w,
            mask_threshold=cluster_mask_thresh,
            merge_iou_mask_size=args.merge_iou_mask_size,
            bbox_prefilter_iou=args.bbox_prefilter_iou,
            pre_merge_topk_factor=args.pre_merge_topk_factor,
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
                score_thresh=pre_score_thresh,
                original_h=original_h, original_w=original_w,
                mask_threshold=cluster_mask_thresh,
                merge_iou_mask_size=args.merge_iou_mask_size,
                bbox_prefilter_iou=args.bbox_prefilter_iou,
                pre_merge_topk_factor=args.pre_merge_topk_factor,
                merge_method=args.merge_method, wbf_iou=args.wbf_iou, wbf_skip_thr=args.wbf_skip_thr,
                wbf_conf_type=args.wbf_conf_type, wbf_overflow=args.wbf_overflow,
            )
            # Merge ensemble predictions into primary
            if len(merged_pred["scores"]) > 0 or len(ens_pred["scores"]) > 0:
                merged_pred = nms_merge(
                    torch.cat([merged_pred["scores"], ens_pred["scores"]]),
                    torch.cat([merged_pred["masks"], ens_pred["masks"]]),
                    torch.cat([merged_pred["category_ids"], ens_pred["category_ids"]]),
                    iou_threshold=args.nms_iou,
                    max_preds=args.max_preds,
                    mask_threshold=cluster_mask_thresh,
                    iou_mask_size=args.merge_iou_mask_size,
                    bbox_prefilter_iou=args.bbox_prefilter_iou,
                    pre_merge_topk_factor=args.pre_merge_topk_factor,
                )

        # Convert to COCO format
        coco_preds = predictions_to_coco_instances(
            predictions=[merged_pred],
            image_ids=image_ids,
            score_threshold=export_score_thresh,
            mask_threshold=export_mask_thresh,
            category_offset=1,
        )
        total_exported_predictions += len(coco_preds)
        evaluator.update(coco_preds)

        if (batch_idx + 1) % args.report_interval == 0:
            elapsed = time.time() - start_time
            rate = (batch_idx + 1) / elapsed
            eta = (len(dataset) - batch_idx - 1) / rate
            print(f"[TTA] {batch_idx + 1}/{len(dataset)} ({rate:.1f} img/s, "
                  f"{elapsed:.0f}s elapsed, ETA {eta:.0f}s)", flush=True)

        del images, depths, merged_pred
        if args.empty_cache_interval > 0 and device.type == "cuda" and (batch_idx + 1) % args.empty_cache_interval == 0:
            torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("TTA Evaluation Results")
    print("=" * 60)
    if total_exported_predictions == 0:
        raise RuntimeError(
            "TTA exported zero COCO predictions for the full dataset. "
            "Check checkpoint compatibility and pre/export thresholds."
        )
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
