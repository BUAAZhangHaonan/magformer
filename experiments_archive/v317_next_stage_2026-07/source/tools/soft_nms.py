#!/usr/bin/env python3
"""
Soft-NMS utility for MagFormer TTA evaluation.

Implements score-decay NMS variants that preserve overlapping predictions
instead of removing them. Useful for dense scenes where hard NMS discards
valid detections.

Methods:
  - gaussian: score *= exp(-IoU^2 / sigma)   (smooth decay, recommended)
  - linear:   score *= (1 - IoU)              (linear decay when IoU > threshold)
  - hard:     standard NMS                     (for comparison)
"""

import torch
import torch.nn.functional as F


def _box_iou_matrix(boxes1, boxes2):
    """Compute pairwise IoU matrix between two sets of boxes (N,4) and (M,4) in xyxy format."""
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp_min(0) * (boxes1[:, 3] - boxes1[:, 1]).clamp_min(0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp_min(0) * (boxes2[:, 3] - boxes2[:, 1]).clamp_min(0)

    # pairwise intersection
    # boxes1: (N,1,4), boxes2: (1,M,4)
    inter_x1 = torch.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    inter_y1 = torch.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    inter_x2 = torch.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    inter_y2 = torch.minimum(boxes1[:, None, 3], boxes2[None, :, 3])

    inter = (inter_x2 - inter_x1).clamp_min(0) * (inter_y2 - inter_y1).clamp_min(0)
    union = area1[:, None] + area2[None, :] - inter
    return torch.where(union > 0, inter / union.clamp_min(1.0), torch.zeros_like(inter))


def _box_iou_against_ref(boxes, ref_box):
    """IoU of each box against a single reference box. Returns (N,) tensor."""
    if boxes.numel() == 0:
        return boxes.new_zeros((0,))

    x1 = torch.maximum(boxes[:, 0], ref_box[0])
    y1 = torch.maximum(boxes[:, 1], ref_box[1])
    x2 = torch.minimum(boxes[:, 2], ref_box[2])
    y2 = torch.minimum(boxes[:, 3], ref_box[3])
    inter = (x2 - x1).clamp_min(0) * (y2 - y1).clamp_min(0)

    box_area = (boxes[:, 2] - boxes[:, 0]).clamp_min(0) * (boxes[:, 3] - boxes[:, 1]).clamp_min(0)
    ref_area = (ref_box[2] - ref_box[0]).clamp_min(0) * (ref_box[3] - ref_box[1]).clamp_min(0)
    union = box_area + ref_area - inter
    return torch.where(union > 0, inter / union.clamp_min(1.0), torch.zeros_like(union))


def masks_to_bboxes_vectorized(masks, threshold=0.5):
    """Vectorized bbox extraction from probability masks. Returns (N, 4) xyxy."""
    binary = (masks > threshold).float()
    N = masks.shape[0]
    if N == 0:
        return masks.new_zeros((0, 4))

    bboxes = masks.new_zeros((N, 4))
    for i in range(N):
        rows = (binary[i].sum(dim=1) > 0).nonzero(as_tuple=False).flatten()
        cols = (binary[i].sum(dim=0) > 0).nonzero(as_tuple=False).flatten()
        if len(rows) > 0 and len(cols) > 0:
            bboxes[i, 0] = cols[0].float()
            bboxes[i, 1] = rows[0].float()
            bboxes[i, 2] = cols[-1].float() + 1.0
            bboxes[i, 3] = rows[-1].float() + 1.0
    return bboxes


def soft_nms(scores, masks, category_ids, *,
             method='gaussian', sigma=0.5, iou_threshold=0.3,
             score_threshold=0.001, max_preds=200,
             mask_threshold=0.5, iou_mask_size=128,
             bbox_prefilter_iou=0.25, input_is_logits=False):
    """
    Soft-NMS that decays scores of overlapping predictions instead of removing them.

    Works on per-image predictions in the format used by evaluate_tta.py.

    Args:
        scores: (N,) tensor of detection scores
        masks: (N, H, W) tensor of masks (probabilities or logits)
        category_ids: (N,) long tensor of class labels
        method: 'gaussian', 'linear', or 'hard'
        sigma: Gaussian decay parameter (lower = more aggressive decay)
        iou_threshold: IoU threshold for linear method; also used for hard fallback
        score_threshold: minimum score to keep after decay
        max_preds: maximum predictions to return
        mask_threshold: threshold for binarizing masks
        iou_mask_size: max side for downsampled mask IoU (speed vs accuracy)
        bbox_prefilter_iou: bbox IoU threshold to skip expensive mask IoU
        input_is_logits: if True, masks are logits and will be sigmoid'd

    Returns:
        dict with 'scores', 'masks', 'category_ids' keys
    """
    if scores.numel() == 0:
        H, W = masks.shape[-2:] if masks.numel() > 0 else (1, 1)
        return {
            "scores": scores.new_empty((0,)),
            "masks": scores.new_zeros((0, H, W)),
            "category_ids": category_ids.new_empty((0,), dtype=torch.long),
        }

    scores = scores.reshape(-1).clone().float()
    category_ids = category_ids.reshape(-1).long()
    masks_float = masks.float()
    if masks_float.ndim == 2:
        masks_float = masks_float.unsqueeze(0)

    H, W = masks_float.shape[-2], masks_float.shape[-3] if masks_float.ndim == 3 else 1

    # Prepare masks for IoU computation
    masks_for_iou = masks_float.sigmoid() if input_is_logits else masks_float

    # Downsample for speed
    if iou_mask_size > 0:
        height, width = masks_for_iou.shape[-2], masks_for_iou.shape[-1]
        largest_side = max(height, width)
        if largest_side > iou_mask_size:
            scale = float(iou_mask_size) / float(largest_side)
            new_h = max(1, int(round(height * scale)))
            new_w = max(1, int(round(width * scale)))
            masks_for_iou = F.adaptive_max_pool2d(
                masks_for_iou.unsqueeze(1), output_size=(new_h, new_w)
            ).squeeze(1)

    # Compute bboxes for all masks
    binary_masks = masks_for_iou >= float(mask_threshold)
    bboxes = masks_to_bboxes_vectorized(binary_masks.float(), threshold=0.5)

    # Process each class independently
    all_keep_indices = []
    unique_cats = category_ids.unique()

    for cat_id in unique_cats:
        cat_mask = (category_ids == cat_id)
        cat_indices = torch.where(cat_mask)[0]
        n = cat_indices.numel()
        if n == 0:
            continue

        cat_scores = scores[cat_indices].clone()
        cat_bboxes = bboxes[cat_indices]
        cat_masks_binary = binary_masks[cat_indices]

        # Filter out zero-area boxes
        areas = (cat_bboxes[:, 2] - cat_bboxes[:, 0]).clamp_min(0) * \
                (cat_bboxes[:, 3] - cat_bboxes[:, 1]).clamp_min(0)
        valid = areas > 0
        if valid.sum() == 0:
            continue

        # Work with valid subset, track original indices within this class
        valid_local = torch.where(valid)[0]
        local_scores = cat_scores[valid_local].clone()
        local_bboxes = cat_bboxes[valid_local]
        local_masks_binary = cat_masks_binary[valid_local]
        n_valid = valid_local.numel()

        # Soft-NMS iterative loop
        # Sort by score descending initially
        order = local_scores.argsort(descending=True)

        # We'll track which local indices are still "alive" and their current scores
        # Instead of reordering arrays, we use an index mapping
        current_scores = local_scores.clone()
        alive = torch.ones(n_valid, dtype=torch.bool, device=scores.device)
        output_local_indices = []

        # Pre-compute flattened masks for vectorized mask IoU
        flat_masks = local_masks_binary.flatten(1).float()  # (n_valid, H*W)

        for _ in range(n_valid):
            # Find best alive detection
            alive_indices = torch.where(alive)[0]
            if alive_indices.numel() == 0:
                break

            best_alive_idx_in_alive = current_scores[alive_indices].argmax()
            best_idx = alive_indices[best_alive_idx_in_alive]

            if current_scores[best_idx] < score_threshold:
                break

            output_local_indices.append(best_idx)
            alive[best_idx] = False

            # Compute IoU of best vs all remaining alive
            remaining = alive_indices[alive_indices != best_idx]
            if remaining.numel() == 0:
                continue

            # Step 1: Bbox IoU prefilter
            bbox_ious = _box_iou_against_ref(local_bboxes[remaining], local_bboxes[best_idx])

            # Step 2: For those passing bbox prefilter, compute mask IoU
            high_bbox_iou = bbox_ious >= bbox_prefilter_iou
            ious = torch.zeros(remaining.numel(), device=scores.device)
            ious[high_bbox_iou] = bbox_ious[high_bbox_iou]  # default to bbox IoU

            if high_bbox_iou.any():
                # Compute mask IoU for close matches
                close_indices = remaining[high_bbox_iou]
                seed_flat = flat_masks[best_idx]  # (H*W,)
                close_flat = flat_masks[close_indices]  # (K, H*W)

                # Binary mask IoU: intersection / union
                intersections = (close_flat * seed_flat[None, :]).sum(dim=1)
                unions = close_flat.sum(dim=1) + seed_flat.sum() - intersections
                mask_ious = torch.where(
                    unions > 0,
                    intersections / unions.clamp_min(1.0),
                    torch.zeros_like(unions),
                )
                ious[high_bbox_iou] = mask_ious

            # Step 3: Decay scores based on method
            if method == 'gaussian':
                decay = torch.exp(-(ious ** 2) / sigma)
            elif method == 'linear':
                # Only decay when IoU exceeds threshold
                decay = torch.where(ious >= iou_threshold, 1.0 - ious, torch.ones_like(ious))
            elif method == 'hard':
                # Hard NMS: zero out overlapping detections
                decay = torch.where(ious >= iou_threshold, torch.zeros_like(ious), torch.ones_like(ious))
            else:
                raise ValueError(f"Unknown soft-NMS method: {method}")

            current_scores[remaining] *= decay

        # Map local indices back to cat_indices then to global indices
        if output_local_indices:
            local_to_cat = valid_local[torch.stack(output_local_indices)]
            global_indices = cat_indices[local_to_cat]

            # Filter by final score threshold
            final_scores = current_scores[torch.stack(output_local_indices)]
            keep = final_scores >= score_threshold
            global_indices = global_indices[keep]

            # Update global scores with decayed values
            scores[global_indices] = final_scores[keep]
            all_keep_indices.append(global_indices)

    if not all_keep_indices:
        H_out, W_out = masks_float.shape[-2], masks_float.shape[-1]
        return {
            "scores": scores.new_empty((0,)),
            "masks": scores.new_zeros((0, H_out, W_out)),
            "category_ids": category_ids.new_empty((0,), dtype=torch.long),
        }

    keep_indices = torch.cat(all_keep_indices)
    keep_scores = scores[keep_indices]

    # Sort by score descending, take top-k
    sorted_idx = keep_scores.argsort(descending=True)
    if max_preds is not None and sorted_idx.numel() > max_preds:
        sorted_idx = sorted_idx[:max_preds]

    keep_indices = keep_indices[sorted_idx]

    return {
        "scores": scores[keep_indices],
        "masks": masks_float[keep_indices],
        "category_ids": category_ids[keep_indices],
    }
