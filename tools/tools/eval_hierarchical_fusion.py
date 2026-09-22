#!/usr/bin/env python3
"""
Hierarchical Query Fusion evaluation for MagFormer.

Aggregates predictions from multiple decoder layers to improve recall.
Uses anchor + supplement strategy: final layer predictions are always kept,
earlier layers only add predictions that don't overlap with final layer.

Usage:
    python -u tools/eval_hierarchical_fusion.py --config configs/eval_1k_full_1024_v83_topk200.yaml
    python -u tools/eval_hierarchical_fusion.py --config configs/eval_1k_full_1024_v83_topk200.yaml --layers 6-10 --nms-iou 0.5
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

from magformer.config import load_config
from magformer.data import CocoRgbdDataset
from magformer.data.transforms import RGBDTransform
from magformer.data.collate import collate_fn
from magformer.engine.coco_export import (
    predictions_to_coco_instances, outputs_to_coco_instances,
    _to_numpy, _mask_to_binary, _encode_mask_rle, _bbox_xyxy_from_binary_mask,
)
from magformer.engine.evaluator import COCOEvaluator
from magformer.engine.eval_runtime import _build_prefix_limited_eval_loader


def parse_args():
    parser = argparse.ArgumentParser(description="Hierarchical Query Fusion Eval for MagFormer")
    parser.add_argument("--config", required=True, help="Path to config yaml")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path (overrides config)")
    parser.add_argument("--layers", default="6-10",
                        help="Decoder layers to fuse, e.g. '6-10'. 1-based: layer 1=initial, layer N+1=final.")
    parser.add_argument("--nms-iou", type=float, default=0.5,
                        help="Mask IoU threshold for NMS.")
    parser.add_argument("--topk", type=int, default=None,
                        help="Top-K predictions per layer. Defaults to model's inference_topk.")
    parser.add_argument("--per-layer-topk", type=int, default=None,
                        help="Keep only top-K per layer BEFORE merging.")
    parser.add_argument("--score-threshold", type=float, default=0.0,
                        help="Minimum score before hierarchical fusion (default: 0.0).")
    parser.add_argument("--export-score-threshold", type=float, default=0.0,
                        help="Minimum score for final COCO export (default: 0.0).")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--max-dets", type=int, default=100)
    parser.add_argument("--output", default=None, help="Output directory")
    parser.add_argument("--baseline", action="store_true",
                        help="Also run baseline (final layer only) for comparison.")
    parser.add_argument("--nms-prefilter-iou", type=float, default=0.3,
                        help="Bbox IoU prefilter for NMS.")
    return parser.parse_args()


def parse_layer_range(layer_str, num_decoder_layers):
    parts = layer_str.split("-")
    if len(parts) != 2:
        raise ValueError(f"Invalid layer range: {layer_str}")
    start, end = int(parts[0]), int(parts[1])
    total_layers = num_decoder_layers + 1
    return max(0, start - 1), min(total_layers - 1, end - 1)


def build_model(config, device, checkpoint_override=None):
    from magformer.models import build_model as _build_model
    from magformer.engine.utils import load_torch_checkpoint

    model = _build_model(config)
    ckpt_path = (checkpoint_override
                 or getattr(config.model, 'finetune_weights', None)
                 or getattr(config.model, 'weights', None))
    if ckpt_path is None:
        raise ValueError("No checkpoint path found. Use --checkpoint.")

    ckpt = load_torch_checkpoint(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
    new_sd = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    print(f"[HFuse] Loaded: {ckpt_path}")
    print(f"[HFuse] Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    return model.to(device).eval()


def build_eval_loader(config, batch_size=1, num_workers=2):
    data_cfg = config.data
    val_split = getattr(data_cfg, "val_split", "val")
    dataset = CocoRgbdDataset(
        dataset_root=data_cfg.dataset_root, ann_file=data_cfg.val_ann,
        split=val_split, transform=None, is_train=False,
    )
    dataset.transform = RGBDTransform(
        image_size=data_cfg.image_size, min_scale=data_cfg.min_scale, max_scale=data_cfg.max_scale,
        random_flip="none", rgb_brightness=0.0, rgb_contrast=0.0, rgb_saturation=0.0, rgb_hue=0.0,
        depth_scale=data_cfg.depth.scale, depth_shift=data_cfg.depth.shift,
        depth_clip_min=data_cfg.depth.clip_min, depth_clip_max=data_cfg.depth.clip_max,
        depth_norm=data_cfg.depth.norm,
        depth_per_sample_norm=getattr(data_cfg.depth, "per_sample_norm", True),
        is_train=False,
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True, collate_fn=collate_fn,
        persistent_workers=num_workers > 0, prefetch_factor=4 if num_workers > 0 else None,
    )
    return dataset, loader


def process_layer_gpu(pred_logits, pred_masks, image_shape, inference_topk):
    """Process one decoder layer on GPU. Returns (scores, cats, binary_masks) all on GPU."""
    B, Nq, _ = pred_logits.shape
    H_img, W_img = image_shape[-2:]

    class_scores = F.softmax(pred_logits, dim=-1)[..., :-1]
    num_classes = class_scores.shape[-1]
    if num_classes <= 0:
        return (pred_logits.new_zeros((B, 0)), pred_logits.new_zeros((B, 0), dtype=torch.long),
                pred_masks.new_zeros((B, 0, H_img, W_img), dtype=torch.uint8))

    topk = min(inference_topk, Nq * num_classes)
    top_scores, top_indices = class_scores.flatten(1).topk(topk, dim=1)
    labels = torch.arange(num_classes, device=pred_logits.device).unsqueeze(0).repeat(Nq, 1).flatten(0, 1)
    query_indices = top_indices // num_classes
    class_indices = labels[top_indices]

    masks = pred_masks.gather(1, query_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, pred_masks.shape[-2], pred_masks.shape[-1]))
    if masks.shape[-2:] != (H_img, W_img):
        masks = F.interpolate(
            masks.reshape(B * topk, 1, masks.shape[-2], masks.shape[-1]),
            size=(H_img, W_img), mode="bilinear", align_corners=False,
        ).reshape(B, topk, H_img, W_img)

    mask_probs = masks.sigmoid()
    binary_masks = mask_probs > 0.5
    mask_scores = (mask_probs.flatten(2) * binary_masks.float().flatten(2)).sum(2) / (
        binary_masks.float().flatten(2).sum(2) + 1e-6)
    final_scores = top_scores * mask_scores
    return final_scores, class_indices, binary_masks.to(torch.uint8)


def fast_mask_nms_gpu(all_scores, all_masks, all_cats, iou_threshold, score_threshold,
                      prefilter_bbox_iou=0.3):
    """Efficient mask IoU NMS with bbox prefilter. All tensors on same device (GPU preferred)."""
    valid = all_scores >= score_threshold
    if valid.sum() == 0:
        return []

    scores = all_scores[valid]
    masks = all_masks[valid]
    cats = all_cats[valid]

    N = len(scores)
    if N <= 1:
        return [0]

    sorted_idx = torch.argsort(scores, descending=True)

    # Downsample masks for IoU computation
    H, W = masks.shape[-2:]
    iou_size = 128
    if H > iou_size or W > iou_size:
        ms = F.interpolate(masks.unsqueeze(1).float(), size=(iou_size, iou_size), mode="nearest").squeeze(1)
    else:
        ms = masks.float()
    ms_flat = ms.flatten(1)
    ms_areas = ms_flat.sum(1)

    # Compute bboxes from masks
    bboxes = masks.new_zeros(N, 4, dtype=torch.float32)
    for i in range(N):
        ys, xs = torch.where(masks[i] > 0)
        if len(xs) > 0:
            bboxes[i, 0] = xs.min().float()
            bboxes[i, 1] = ys.min().float()
            bboxes[i, 2] = xs.max().float() + 1
            bboxes[i, 3] = ys.max().float() + 1

    bbox_areas = (bboxes[:, 2] - bboxes[:, 0]).clamp_min(0) * (bboxes[:, 3] - bboxes[:, 1]).clamp_min(0)

    keep = []
    suppressed = torch.zeros(N, dtype=torch.bool, device=masks.device)

    for i in range(N):
        idx = sorted_idx[i]
        if suppressed[idx]:
            continue
        keep.append(idx.item())
        if i == N - 1:
            break

        remaining_idx = sorted_idx[i+1:]
        remaining_mask = ~suppressed[remaining_idx]
        if not remaining_mask.any():
            break
        remaining = remaining_idx[remaining_mask]

        # Bbox prefilter
        ref_bbox = bboxes[idx]
        inter_x1 = torch.maximum(ref_bbox[0], bboxes[remaining, 0])
        inter_y1 = torch.maximum(ref_bbox[1], bboxes[remaining, 1])
        inter_x2 = torch.minimum(ref_bbox[2], bboxes[remaining, 2])
        inter_y2 = torch.minimum(ref_bbox[3], bboxes[remaining, 3])
        inter = (inter_x2 - inter_x1).clamp_min(0) * (inter_y2 - inter_y1).clamp_min(0)
        union = bbox_areas[idx] + bbox_areas[remaining] - inter
        bbox_iou = torch.where(union > 0, inter / union.clamp_min(1.0), torch.zeros_like(inter))

        same_cat = cats[idx] == cats[remaining]
        check = (bbox_iou > prefilter_bbox_iou) & same_cat
        if not check.any():
            continue

        candidates = remaining[check]
        ref_flat = ms_flat[idx]
        cand_flat = ms_flat[candidates]
        intersection = torch.mv(cand_flat, ref_flat)
        union_m = ms_areas[idx] + ms_areas[candidates] - intersection
        mask_iou = torch.where(union_m > 0, intersection / union_m.clamp_min(1.0), torch.zeros_like(intersection))

        suppress = mask_iou > iou_threshold
        if suppress.any():
            suppressed[candidates[suppress]] = True

    return keep


def hierarchical_fusion(outputs, image_shape, layer_start, layer_end,
                        inference_topk, per_layer_topk, nms_iou,
                        score_threshold, prefilter_bbox_iou, device):
    """Anchor + supplement fusion strategy.

    1. Process the FINAL layer (highest quality) as anchor predictions.
    2. Process earlier layers and add ONLY predictions that don't overlap
       with any anchor (i.e. genuinely new objects the final layer missed).
    3. Run a final NMS pass on the combined set.
    """
    B = outputs["pred_logits"].shape[0]
    aux_outputs = outputs.get("aux_outputs", [])
    num_aux = len(aux_outputs)
    final_layer_idx = num_aux  # index of the final output

    # Build list of earlier layers (exclude final)
    earlier_layers = []
    for idx in range(layer_start, layer_end + 1):
        if idx == final_layer_idx:
            continue
        if idx < num_aux:
            earlier_layers.append((aux_outputs[idx]["pred_logits"], aux_outputs[idx]["pred_masks"]))

    batch_predictions = []
    for img_idx in range(B):
        # Step 1: Process the final layer as anchor
        anchor_logits = outputs["pred_logits"][img_idx:img_idx+1]
        anchor_masks_raw = outputs["pred_masks"][img_idx:img_idx+1]
        anchor_scores, anchor_cats, anchor_bmask = process_layer_gpu(
            anchor_logits, anchor_masks_raw, image_shape, inference_topk)

        a_scores = anchor_scores[0]  # (topk,)
        a_cats = anchor_cats[0]      # (topk,)
        a_masks = anchor_bmask[0]    # (topk, H, W)

        # Filter anchors by score threshold
        a_valid = a_scores >= score_threshold
        a_scores = a_scores[a_valid]
        a_cats = a_cats[a_valid]
        a_masks = a_masks[a_valid]

        if len(a_scores) == 0:
            batch_predictions.append({
                "scores": torch.zeros((0,), device="cpu"),
                "category_ids": torch.zeros((0,), dtype=torch.long, device="cpu"),
                "masks": torch.zeros((0, image_shape[-2], image_shape[-1]), dtype=torch.uint8, device="cpu"),
            })
            continue

        # Step 2: Process earlier layers and find supplementary predictions
        sup_scores_list = []
        sup_cats_list = []
        sup_masks_list = []

        for logits, masks in earlier_layers:
            s_gpu, c_gpu, m_gpu = process_layer_gpu(
                logits[img_idx:img_idx+1], masks[img_idx:img_idx+1], image_shape, inference_topk)
            s = s_gpu[0]
            c = c_gpu[0]
            m = m_gpu[0]

            if per_layer_topk is not None and len(s) > per_layer_topk:
                topk_s, topk_idx = s.topk(min(per_layer_topk, len(s)))
                s, c, m = topk_s, c[topk_idx], m[topk_idx]

            # Filter by score
            valid = s >= score_threshold
            s = s[valid]
            c = c[valid]
            m = m[valid]

            if len(s) == 0:
                continue

            # Check each prediction against anchors
            # Downsample anchors and candidates for IoU
            iou_size = 128
            H, W = a_masks.shape[-2:]
            if H > iou_size or W > iou_size:
                a_ds = F.interpolate(a_masks.unsqueeze(1).float(),
                                     size=(iou_size, iou_size), mode="nearest").squeeze(1)
            else:
                a_ds = a_masks.float()
            a_flat = a_ds.flatten(1)  # (num_anchors, iou_size^2)

            for j in range(len(s)):
                same_cat = a_cats == c[j]
                if not same_cat.any():
                    # New class not in anchors -> keep
                    sup_scores_list.append(s[j])
                    sup_cats_list.append(c[j])
                    sup_masks_list.append(m[j])
                    continue

                cat_a_flat = a_flat[same_cat]  # (num_same_cat_anchors, iou_size^2)

                # Downsample candidate
                if H > iou_size or W > iou_size:
                    cand_ds = F.interpolate(m[j].float().unsqueeze(0).unsqueeze(0),
                                            size=(iou_size, iou_size), mode="nearest").squeeze().flatten()
                else:
                    cand_ds = m[j].float().flatten()

                # Compute mask IoU with all same-class anchors
                intersections = torch.mv(cat_a_flat, cand_ds)
                cand_area = cand_ds.sum()
                a_areas = cat_a_flat.sum(1)
                unions = cand_area + a_areas - intersections
                ious = torch.where(unions > 0, intersections / unions.clamp(min=1), torch.zeros_like(intersections))

                if (ious > nms_iou).any():
                    # Overlaps with an anchor -> skip
                    continue

                sup_scores_list.append(s[j])
                sup_cats_list.append(c[j])
                sup_masks_list.append(m[j])

        # Step 3: Combine anchors + supplements
        if len(sup_scores_list) > 0:
            sup_scores = torch.stack(sup_scores_list)
            sup_cats = torch.stack(sup_cats_list)
            sup_masks = torch.stack(sup_masks_list)

            all_scores = torch.cat([a_scores, sup_scores])
            all_cats = torch.cat([a_cats, sup_cats])
            all_masks = torch.cat([a_masks, sup_masks])
        else:
            all_scores = a_scores
            all_cats = a_cats
            all_masks = a_masks

        # Run NMS on combined set
        keep = fast_mask_nms_gpu(all_scores, all_masks, all_cats,
                                 nms_iou, score_threshold, prefilter_bbox_iou)

        if len(keep) == 0:
            batch_predictions.append({
                "scores": torch.zeros((0,), device="cpu"),
                "category_ids": torch.zeros((0,), dtype=torch.long, device="cpu"),
                "masks": torch.zeros((0, image_shape[-2], image_shape[-1]), dtype=torch.uint8, device="cpu"),
            })
        else:
            keep_t = torch.tensor(keep, dtype=torch.long, device=device)
            kept_scores = all_scores[keep_t]
            sorted_idx = torch.argsort(kept_scores, descending=True)
            keep_t = keep_t[sorted_idx]

            batch_predictions.append({
                "scores": all_scores[keep_t].cpu(),
                "category_ids": all_cats[keep_t].cpu(),
                "masks": all_masks[keep_t].cpu(),
            })

    return batch_predictions


def fused_preds_to_coco(pred_list, image_ids, category_ids_list=None,
                        score_threshold=0.0, category_offset=1):
    rows = []
    if image_ids is None:
        raise ValueError("image_ids are required for hierarchical COCO export")
    img_ids = [int(image_id) for image_id in image_ids]
    if len(img_ids) != len(pred_list):
        raise ValueError("image_ids length must match hierarchical predictions length")

    for batch_idx, pred in enumerate(pred_list):
        image_id = img_ids[batch_idx]
        scores = _to_numpy(pred["scores"])
        cats = _to_numpy(pred["category_ids"])
        masks = _to_numpy(pred["masks"])
        if masks.ndim == 2:
            masks = masks[None, ...]

        for i in range(len(scores)):
            score = float(scores[i])
            if score < score_threshold:
                continue
            binary_mask = _mask_to_binary(masks[i], mask_threshold=0.5)
            bbox = _bbox_xyxy_from_binary_mask(binary_mask)
            if bbox is None:
                continue
            rle_mask = _encode_mask_rle(binary_mask)
            del binary_mask
            if category_ids_list is not None and 0 <= int(cats[i]) < len(category_ids_list):
                category_id = int(category_ids_list[int(cats[i])])
            else:
                category_id = int(cats[i]) + int(category_offset)
            rows.append({"image_id": image_id, "category_id": category_id,
                         "score": score, "mask": rle_mask, "bbox": bbox})
    return rows


def main():
    args = parse_args()

    overrides = {}
    if args.checkpoint:
        overrides.setdefault("model", {})["weights"] = args.checkpoint
    if args.output:
        overrides.setdefault("runtime", {})["output_dir"] = args.output
    config = load_config(args.config, overrides=overrides)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config, device, checkpoint_override=args.checkpoint)
    inference_model = model

    num_decoder_layers = inference_model.decoder.num_layers
    inference_topk = args.topk or inference_model.inference_topk

    layer_start, layer_end = parse_layer_range(args.layers, num_decoder_layers)
    total_layers = layer_end - layer_start + 1

    print(f"[HFuse] Decoder layers: {num_decoder_layers}, Total prediction layers: {num_decoder_layers + 1}")
    print(f"[HFuse] Topk per layer: {inference_topk}, Per-layer topk: {args.per_layer_topk or 'none'}")
    print(f"[HFuse] Fusing layers {layer_start+1}-{layer_end+1} ({total_layers} layers)")
    print(f"[HFuse] NMS IoU: {args.nms_iou}, Bbox prefilter: {args.nms_prefilter_iou}")
    print(f"[HFuse] Score threshold: pre={args.score_threshold}, export={args.export_score_threshold}")

    dataset, loader = build_eval_loader(config, batch_size=args.batch_size, num_workers=args.num_workers)
    loader, total_images = _build_prefix_limited_eval_loader(
        loader, args.max_images
    )
    category_ids = list(getattr(dataset, "category_ids", [])) or None
    amp_enabled = getattr(config.solver, 'amp_enabled', True)

    output_dir = Path(getattr(config.runtime, 'output_dir', 'output/hfuse'))
    output_dir.mkdir(parents=True, exist_ok=True)

    iou_types = ["bbox", "segm"]
    evaluator = COCOEvaluator(coco_gt=dataset.coco, iou_types=iou_types, max_dets=args.max_dets)
    baseline_evaluator = None
    if args.baseline:
        baseline_evaluator = COCOEvaluator(coco_gt=dataset.coco, iou_types=iou_types, max_dets=args.max_dets)

    eval_start = time.time()
    num_evaluated = 0
    num_images_evaluated = 0
    total_preds = 0

    print(f"\n[HFuse] Starting: {total_images} images, {total_layers} layers fused\n")

    for batch in loader:
        images = batch["images"].to(device)
        depths = batch["depths"].to(device)
        noise_masks = batch.get("noise_masks")
        if noise_masks is not None:
            noise_masks = noise_masks.to(device)
        padding_masks = batch.get("padding_masks")
        if padding_masks is not None:
            padding_masks = padding_masks.to(device)

        if "image_ids" not in batch:
            raise KeyError("Evaluation batch is missing required image_ids")
        image_ids = [int(image_id) for image_id in batch["image_ids"]]
        if len(image_ids) != int(images.shape[0]):
            raise ValueError(
                "Evaluation image_ids length must match batch size: "
                f"image_ids={len(image_ids)}, batch={int(images.shape[0])}"
            )
        image_shape = images.shape

        amp_context = autocast("cuda") if amp_enabled and device.type == "cuda" else nullcontext()
        with torch.inference_mode(), amp_context:
            decoder_outputs = inference_model.forward_inference_decoder_outputs(
                images=images, depths=depths,
                padding_masks=padding_masks, depth_noise_masks=noise_masks,
            )

        fused_preds = hierarchical_fusion(
            decoder_outputs, image_shape,
            layer_start=layer_start, layer_end=layer_end,
            inference_topk=inference_topk, per_layer_topk=args.per_layer_topk,
            nms_iou=args.nms_iou, score_threshold=args.score_threshold,
            prefilter_bbox_iou=args.nms_prefilter_iou, device=device,
        )

        coco_preds = fused_preds_to_coco(
            fused_preds, image_ids=image_ids,
            category_ids_list=category_ids, score_threshold=args.export_score_threshold,
        )
        evaluator.update(coco_preds, image_ids=image_ids)
        total_preds += len(coco_preds)

        if baseline_evaluator is not None:
            with torch.inference_mode(), amp_context:
                baseline_outputs = inference_model._inference_raw(
                    decoder_outputs, image_shape,
                    inference_topk=inference_topk, move_predictions_to_cpu=True,
                )
            baseline_coco = outputs_to_coco_instances(
                outputs=baseline_outputs, image_ids=image_ids,
                score_threshold=args.export_score_threshold, category_ids=category_ids,
            )
            baseline_evaluator.update(baseline_coco, image_ids=image_ids)
            del baseline_outputs

        del decoder_outputs, fused_preds

        num_evaluated += 1
        num_images_evaluated += len(image_ids)
        if num_images_evaluated > total_images:
            raise RuntimeError(
                "Hierarchical evaluation exceeded the exact image plan: "
                f"planned={total_images}, observed={num_images_evaluated}"
            )

        if num_evaluated % 25 == 0:
            elapsed = time.time() - eval_start
            rate = num_images_evaluated / elapsed if elapsed > 0 else 0
            remaining = (total_images - num_images_evaluated) / rate if rate > 0 else 0
            avg_preds = total_preds / num_images_evaluated
            print(f"[HFuse] {num_images_evaluated}/{total_images} "
                  f"({rate:.1f} img/s, {avg_preds:.0f} preds/img, ETA {remaining:.0f}s)")
            gc.collect()
            torch.cuda.empty_cache()

        if num_images_evaluated >= total_images:
            break

    if num_images_evaluated != total_images:
        raise RuntimeError(
            "Hierarchical evaluation did not complete the exact image plan: "
            f"planned={total_images}, observed={num_images_evaluated}"
        )
    inference_time = time.time() - eval_start
    print(f"\n[HFuse] Done: {num_images_evaluated} images in {inference_time:.1f}s "
          f"({num_images_evaluated/inference_time:.1f} img/s)")

    print(f"\n{'='*60}")
    print(f"Hierarchical Fusion (layers {args.layers}, NMS IoU={args.nms_iou})")
    print(f"{'='*60}")
    fused_metrics = evaluator.summarize()
    fused_path = evaluator.dump(output_dir / "hfuse_instances_results.json")
    print(f"[HFuse] Saved to {fused_path}")

    if baseline_evaluator is not None:
        print(f"\n{'='*60}")
        print(f"Baseline (final layer only)")
        print(f"{'='*60}")
        baseline_metrics = baseline_evaluator.summarize()
        baseline_path = baseline_evaluator.dump(output_dir / "baseline_instances_results.json")
        print(f"[HFuse] Baseline saved to {baseline_path}")

        print(f"\n{'='*60}")
        print(f"Diff: Fusion - Baseline")
        print(f"{'='*60}")
        for key in fused_metrics:
            f_val = fused_metrics[key]
            b_val = baseline_metrics.get(key, None)
            if b_val is not None:
                diff = f_val - b_val
                sign = "+" if diff >= 0 else ""
                print(f"  {key}: {sign}{diff:.4f}  (fused={f_val:.4f} base={b_val:.4f})")

    print(f"\n[HFuse] Complete.")


if __name__ == "__main__":
    main()
