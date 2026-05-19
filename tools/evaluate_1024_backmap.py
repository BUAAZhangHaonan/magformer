#!/usr/bin/env python3
"""Evaluate 1024 input predictions mapped back to COCO ground-truth size."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import cv2
import numpy as np
import torch
import yaml
from pycocotools import mask as coco_mask
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from magformer.config import load_config, set_seed, setup_device  # noqa: E402
from magformer.config.loader import load_yaml_file, save_yaml_file  # noqa: E402
from magformer.config.schema import merge_configs  # noqa: E402
from magformer.data import CocoRgbdDataset  # noqa: E402
from magformer.data.collate import collate_fn  # noqa: E402
from magformer.data.transforms import (  # noqa: E402
    Compose,
    DepthNormalize,
    FixedSizeCrop,
    InitContentMask,
    ResizeScale,
    ToTensor,
)
from magformer.engine.evaluator import COCOEvaluator  # noqa: E402
from magformer.engine.inference_stats import InferenceStatsAccumulator  # noqa: E402
from magformer.engine.utils import load_torch_checkpoint  # noqa: E402
from magformer.models import build_model  # noqa: E402


class Eval1024Transform:
    def __init__(
        self,
        image_size: int,
        depth_scale: float,
        depth_shift: float,
        depth_clip_min: float,
        depth_clip_max: float,
        depth_norm: str,
        depth_per_sample_norm: bool,
    ) -> None:
        self.transform = Compose(
            [
                InitContentMask(),
                ResizeScale(min_scale=1.0, max_scale=1.0, target_size=image_size),
                FixedSizeCrop((image_size, image_size), random_crop=False),
                DepthNormalize(
                    scale=depth_scale,
                    shift=depth_shift,
                    clip_min=depth_clip_min,
                    clip_max=depth_clip_max,
                    norm=depth_norm,
                    per_sample_norm=depth_per_sample_norm,
                ),
                ToTensor(),
            ]
        )

    def __call__(self, result: Dict[str, Any]) -> Dict[str, Any]:
        return self.transform(result)


def _require_positive_int(value: int | None, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def parse_iou_types(raw: str) -> List[str]:
    iou_types = [part.strip() for part in raw.split(",") if part.strip()]
    if iou_types == ["bbox"] or iou_types == ["bbox", "segm"]:
        return iou_types
    raise ValueError("--iou-types must be either 'bbox' or 'bbox,segm'")


def forward_raw_batch(
    model: torch.nn.Module,
    batch: Dict[str, Any],
    *,
    device: torch.device,
    inference_topk: int,
    collect_inference_stats: bool = False,
) -> Dict[str, Any]:
    images = batch["images"].to(device, non_blocking=True)
    depths = batch["depths"].to(device, non_blocking=True)
    padding_masks = batch.get("padding_masks")
    if padding_masks is not None:
        padding_masks = padding_masks.to(device, non_blocking=True)
    depth_valid_masks = batch.get("depth_valid_masks")
    if depth_valid_masks is not None:
        depth_valid_masks = depth_valid_masks.to(device, non_blocking=True)
    forward_kwargs = {
        "padding_masks": padding_masks,
        "depth_valid_masks": depth_valid_masks,
    }
    if collect_inference_stats:
        forward_kwargs["collect_inference_stats"] = True
    return model.forward_inference_raw(
        images,
        depths,
        inference_topk=inference_topk,
        **forward_kwargs,
    )


def slice_batch_for_max_images(batch: Dict[str, Any], remaining: int) -> Dict[str, Any]:
    remaining = _require_positive_int(remaining, "remaining max-images")
    sliced: Dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value) and value.ndim > 0 and int(value.shape[0]) > remaining:
            sliced[key] = value[:remaining]
        elif isinstance(value, list) and len(value) > remaining:
            sliced[key] = value[:remaining]
        elif isinstance(value, tuple) and len(value) > remaining:
            sliced[key] = value[:remaining]
        else:
            sliced[key] = value
    return sliced


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate 1024 input predictions mapped back to COCO ground-truth size.",
        epilog=(
            "Defaults target the VC-SUDA pseudo-real smoke path, not the Teacher 8499 "
            "original 1.5K reproduction. For Teacher 8499 1.5K, pass: "
            "--base-config configs/finetune_1k_full_1024.yaml "
            "--dataset-root /home/hdd3/zhanghaonan/magformer/magformer_datasets/20260318_1K_1566 "
            "--ann annotations/instances_all.json --split all"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--base-config", default="configs/vc_suda_stage_b_1024_teacher8499.yaml")
    parser.add_argument("--dataset-root", default="magformer_datasets/pseudo_real_512")
    parser.add_argument("--ann", default="annotations/instances_val.json")
    parser.add_argument("--split", default="val")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--inference-topk", type=int, default=100, help="Raw model postprocess top-k per image")
    parser.add_argument("--max-dets", type=int, default=100, help="COCOeval maxDets final value per image")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--iou-types", default="bbox,segm", help="COCO IoU types: bbox or bbox,segm")
    parser.add_argument("--dump-inference-stats", default=None, help="Optional path for per-image inference instrumentation JSON")
    parser.add_argument("--force-pytorch-msda", action="store_true")
    return parser.parse_args()


def strip_module_prefix(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    if state_dict and all(isinstance(k, str) for k in state_dict) and any(
        k.startswith("module.") for k in state_dict
    ):
        return {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
    return state_dict


def checkpoint_state_dict(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    if "model_state_dict" in checkpoint:
        return strip_module_prefix(checkpoint["model_state_dict"])
    if "state_dict" in checkpoint:
        return strip_module_prefix(checkpoint["state_dict"])
    return strip_module_prefix(checkpoint)


def strict_load_with_evidence(model: torch.nn.Module, weights: str) -> None:
    checkpoint = load_torch_checkpoint(weights, map_location="cpu")
    state = checkpoint_state_dict(checkpoint)
    model_state = model.state_dict()
    missing = [k for k in model_state if k not in state]
    unexpected = [k for k in state if k not in model_state]
    shape_mismatch = [
        (k, tuple(state[k].shape), tuple(model_state[k].shape))
        for k in model_state
        if k in state and hasattr(state[k], "shape") and tuple(state[k].shape) != tuple(model_state[k].shape)
    ]
    matched = len(model_state) - len(missing) - len(shape_mismatch)
    print(
        f"[Weights] strict=True matched={matched}/{len(model_state)} "
        f"missing={len(missing)} unexpected={len(unexpected)} shape_mismatch={len(shape_mismatch)}"
    )
    if missing:
        print("[Weights] missing_sample=" + json.dumps(missing[:10]))
    if unexpected:
        print("[Weights] unexpected_sample=" + json.dumps(unexpected[:10]))
    if shape_mismatch:
        print("[Weights] shape_mismatch_sample=" + json.dumps(shape_mismatch[:5]))
    model.load_state_dict(state, strict=True)
    print(f"[Weights] strict load OK: {weights}")


def encode_rle(binary_mask: np.ndarray) -> Dict[str, Any]:
    rle = coco_mask.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    if isinstance(rle["counts"], bytes):
        rle["counts"] = rle["counts"].decode("ascii")
    return {"size": rle["size"], "counts": rle["counts"]}


def bbox_xyxy_from_mask(binary_mask: np.ndarray) -> List[float] | None:
    ys, xs = np.where(binary_mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    if hasattr(x, "cpu"):
        return x.cpu().numpy()
    return np.asarray(x)


def mask_to_prob(mask: Any) -> np.ndarray:
    arr = to_numpy(mask)
    if arr.ndim == 3:
        arr = arr[0]
    if not np.isfinite(arr).all():
        raise ValueError("Mask contains NaN or Inf")
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) if arr.max() <= 1 else arr.astype(np.float32) / 255.0
    if float(arr.min()) < 0.0 or float(arr.max()) > 1.0:
        arr = 1.0 / (1.0 + np.exp(-arr))
    return arr.astype(np.float32)


def content_bounds(content_mask: Any, image_id: int) -> Tuple[int, int, int, int]:
    content = to_numpy(content_mask).astype(bool)
    if content.ndim == 3:
        if content.shape[0] == 1:
            content = content[0]
        elif content.shape[-1] == 1:
            content = content[..., 0]
        else:
            raise ValueError(
                f"content_mask for image_id={image_id} must be 2D or single-channel, "
                f"got shape={content.shape}"
            )
    if content.ndim != 2:
        raise ValueError(f"content_mask for image_id={image_id} must be 2D, got shape={content.shape}")

    ys, xs = np.where(content)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError(f"content_mask for image_id={image_id} has no valid content pixels")
    return int(ys.min()), int(ys.max() + 1), int(xs.min()), int(xs.max() + 1)


def category_id_from_contiguous(contiguous_id: int, category_ids: List[int] | None, offset: int = 1) -> int:
    if category_ids is not None and 0 <= contiguous_id < len(category_ids):
        return int(category_ids[contiguous_id])
    return int(contiguous_id) + offset


def predictions_to_backmapped_coco(
    outputs: Dict[str, Any],
    batch: Dict[str, Any],
    image_size_by_id: Dict[int, Tuple[int, int]],
    category_ids: List[int] | None,
    score_threshold: float,
    mask_threshold: float,
    include_segmentation: bool = True,
    stats_accumulator: InferenceStatsAccumulator | None = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    predictions = outputs["predictions"]
    image_ids = [int(v) for v in batch["image_ids"].detach().cpu().tolist()]
    content_masks = batch.get("content_masks")
    if content_masks is None:
        raise KeyError("batch must contain content_masks for 1024 backmap export; refusing to resize padded masks")

    filter_counts: List[Dict[str, int]] = []
    for batch_idx, pred in enumerate(predictions):
        image_id = image_ids[batch_idx]
        out_h, out_w = image_size_by_id[image_id]
        top, bottom, left, right = content_bounds(content_masks[batch_idx], image_id)
        content_h = bottom - top
        content_w = right - left
        scores = to_numpy(pred.get("scores", []))
        cats = to_numpy(pred.get("category_ids", pred.get("labels", [])))
        masks = to_numpy(pred.get("masks", []))
        if masks.ndim == 2:
            masks = masks[None, ...]

        post_score_count = 0
        post_mask_nonempty_count = 0
        exported_count = 0
        for i, score_value in enumerate(scores):
            score = float(score_value)
            if score < score_threshold:
                continue
            post_score_count += 1
            if i >= len(masks):
                continue
            prob = mask_to_prob(masks[i])
            if prob.shape != tuple(to_numpy(content_masks[batch_idx]).shape[-2:]):
                prob = cv2.resize(prob, (content_masks.shape[-1], content_masks.shape[-2]), interpolation=cv2.INTER_LINEAR)
            prob = prob[top:bottom, left:right]
            if prob.shape != (content_h, content_w):
                raise ValueError(
                    f"Failed to crop prediction for image_id={image_id}: "
                    f"got shape={prob.shape}, expected {(content_h, content_w)}"
                )
            if prob.shape != (out_h, out_w):
                prob = cv2.resize(prob, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
            binary = (prob > mask_threshold).astype(np.uint8)
            bbox = bbox_xyxy_from_mask(binary)
            if bbox is None:
                continue
            post_mask_nonempty_count += 1
            contiguous_id = int(cats[i]) if i < len(cats) else 0
            row = {
                "image_id": image_id,
                "category_id": category_id_from_contiguous(contiguous_id, category_ids),
                "score": score,
                "bbox": bbox,
            }
            if include_segmentation:
                row["mask"] = encode_rle(binary)
            rows.append(row)
            exported_count += 1
        filter_counts.append(
            {
                "post_score_count": post_score_count,
                "post_mask_nonempty_count": post_mask_nonempty_count,
                "exported_count": exported_count,
            }
        )
    if stats_accumulator is not None:
        if "inference_stats" not in outputs:
            raise RuntimeError("--dump-inference-stats requires model outputs to include 'inference_stats'")
        stats_accumulator.add_explicit_records(
            image_ids=image_ids,
            raw_stats=outputs["inference_stats"],
            filter_counts=filter_counts,
        )
    return rows


def depth_stats(t: torch.Tensor) -> Dict[str, float | List[int]]:
    cpu = t.detach().cpu().float()
    return {
        "shape": list(cpu.shape),
        "min": float(cpu.min()),
        "max": float(cpu.max()),
        "mean": float(cpu.mean()),
        "std": float(cpu.std()),
        "unique_rounded": int(torch.unique(torch.round(cpu * 1000) / 1000).numel()),
    }


def summarize_evaluator(evaluator: COCOEvaluator, image_ids: List[int] | None) -> Dict[str, float]:
    if image_ids is None:
        return evaluator.summarize()

    unique_image_ids = sorted(set(int(image_id) for image_id in image_ids))
    if not unique_image_ids:
        raise ValueError("No evaluated image_ids were collected")

    coco_results = evaluator.to_coco_results()
    if not coco_results:
        metrics: Dict[str, float] = {}
        for iou_type in evaluator.iou_types:
            metrics.update(evaluator._zero_metrics(iou_type))
        return metrics

    metrics: Dict[str, float] = {}
    print(f"[Eval] restricting COCOeval to {len(unique_image_ids)} evaluated image_ids")
    for iou_type in evaluator.iou_types:
        coco_dt = evaluator.coco_gt.loadRes(coco_results)
        coco_eval = COCOeval(evaluator.coco_gt, coco_dt, iouType=iou_type)
        coco_eval.params.maxDets = [1, 10, evaluator.max_dets]
        coco_eval.params.imgIds = unique_image_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        metrics.update(evaluator._extract_metrics(coco_eval, iou_type))

    evaluator._print_results(metrics)
    return metrics


def build_eval_runtime_config(base_config: Dict[str, Any], args: argparse.Namespace, output_dir: Path) -> Dict[str, Any]:
    overrides = {
        "data": {
            "dataset_root": args.dataset_root,
            "val_ann": args.ann,
            "val_split": args.split,
            "image_size": args.image_size,
        },
        "model": {
            "weights": args.weights,
            "finetune_weights": args.weights,
        },
        "runtime": {
            "device": "cuda",
            "gpus": [0],
            "ddp_enabled": False,
            "num_workers": args.num_workers,
            "output_dir": str(output_dir),
            "eval_iou_types": args.iou_types,
        },
        "vc_suda": {
            "enabled": False,
            "stage": "A",
            "target_labeled_ann": None,
            "target_unlabeled_ann": None,
        },
    }
    return merge_configs(base_config, overrides)


def main() -> None:
    args = parse_args()
    args.batch_size = _require_positive_int(args.batch_size, "--batch-size")
    args.inference_topk = _require_positive_int(args.inference_topk, "--inference-topk")
    args.max_dets = _require_positive_int(args.max_dets, "--max-dets")
    if args.max_images is not None:
        args.max_images = _require_positive_int(args.max_images, "--max-images")
    iou_types = parse_iou_types(args.iou_types)
    include_segmentation = "segm" in iou_types
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[Eval] iou_types={iou_types} batch_size={args.batch_size} max_images={args.max_images} "
        f"inference_topk={args.inference_topk} max_dets={args.max_dets}"
    )

    if args.force_pytorch_msda:
        import magformer.models.ops.functions.ms_deform_attn_func as msda_func

        msda_func._cuda_available = False
        print("[MSDeformAttn] forced PyTorch core path")

    args.iou_types = iou_types
    cfg_dict = build_eval_runtime_config(load_yaml_file(args.base_config), args, out_dir)
    cfg_path = out_dir / "eval_1024_runtime.yaml"
    save_yaml_file(cfg_dict, str(cfg_path))
    print(f"[Config] wrote {cfg_path}")

    config = load_config(str(cfg_path))
    set_seed(config.runtime.seed)
    random.seed(config.runtime.seed)
    np.random.seed(config.runtime.seed)
    torch.manual_seed(config.runtime.seed)

    dataset = CocoRgbdDataset(
        dataset_root=config.data.dataset_root,
        ann_file=config.data.val_ann,
        split=config.data.val_split,
        transform=None,
        is_train=False,
    )
    depth_cfg = config.data.depth
    dataset.transform = Eval1024Transform(
        image_size=config.data.image_size,
        depth_scale=depth_cfg.scale,
        depth_shift=depth_cfg.shift,
        depth_clip_min=depth_cfg.clip_min,
        depth_clip_max=depth_cfg.clip_max,
        depth_norm=depth_cfg.norm,
        depth_per_sample_norm=depth_cfg.per_sample_norm,
    )
    image_size_by_id = {
        int(img_id): (int(info["height"]), int(info["width"]))
        for img_id, info in dataset.coco.imgs.items()
    }
    print(
        f"[Dataset] ann={config.data.val_ann} split={config.data.val_split} "
        f"images={len(dataset)} gt_sizes={sorted(set(image_size_by_id.values()))}"
    )
    print(
        f"[DepthNorm] scale={depth_cfg.scale} shift={depth_cfg.shift} "
        f"clip_min={depth_cfg.clip_min} clip_max={depth_cfg.clip_max} "
        f"norm={depth_cfg.norm} per_sample_norm={depth_cfg.per_sample_norm}"
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    first_batch = next(iter(loader))
    print(f"[InputEvidence] images={list(first_batch['images'].shape)} depths={list(first_batch['depths'].shape)}")
    print("[DepthEvidence] first_batch=" + json.dumps(depth_stats(first_batch["depths"]), sort_keys=True))
    print(
        f"[CoordEvidence] model_input={args.image_size}x{args.image_size} "
        f"prediction_masks_resized_to_gt={sorted(set(image_size_by_id.values()))}"
    )

    device = setup_device(config.runtime)
    model = build_model(config)
    strict_load_with_evidence(model, args.weights)
    model = model.to(device)
    model.eval()

    evaluator = COCOEvaluator(dataset.coco, iou_types=iou_types, max_dets=args.max_dets)
    stats_accumulator = InferenceStatsAccumulator(dataset.coco) if args.dump_inference_stats else None
    num_eval = 0
    evaluated_image_ids: List[int] = []
    start = time.time()
    category_ids = list(getattr(dataset, "category_ids", [])) or None
    print(f"[Eval] category_ids={category_ids} score_threshold={args.score_threshold} mask_threshold={args.mask_threshold}")

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if args.max_images is not None:
                remaining = args.max_images - num_eval
                if remaining <= 0:
                    break
                batch_size = int(batch["images"].shape[0])
                if batch_size > remaining:
                    batch = slice_batch_for_max_images(batch, remaining)
            images = batch["images"].to(device, non_blocking=True)
            outputs = forward_raw_batch(
                model,
                batch,
                device=device,
                inference_topk=args.inference_topk,
                collect_inference_stats=stats_accumulator is not None,
            )
            rows = predictions_to_backmapped_coco(
                outputs,
                batch,
                image_size_by_id,
                category_ids,
                score_threshold=args.score_threshold,
                mask_threshold=args.mask_threshold,
                include_segmentation=include_segmentation,
                stats_accumulator=stats_accumulator,
            )
            evaluator.update(rows)
            evaluated_image_ids.extend(int(image_id) for image_id in batch["image_ids"].tolist())
            num_eval += int(images.shape[0])
            print(
                f"[Eval] batch={batch_idx} image_ids={batch['image_ids'].tolist()} "
                f"input={list(images.shape)} depth={list(depths.shape)} rows={len(rows)}"
            )
            if args.max_images is not None and num_eval >= args.max_images:
                break
            del outputs
            if device.type == "cuda":
                torch.cuda.empty_cache()

    print(
        f"[Eval] inference_done evaluated_images={num_eval} "
        f"max_images={args.max_images} seconds={time.time() - start:.1f}"
    )
    metrics = summarize_evaluator(
        evaluator,
        evaluated_image_ids if args.max_images is not None else None,
    )
    # Backmapped rows are internal xyxy/mask rows; dump writes standard COCO xywh/segmentation rows.
    results_path = evaluator.dump(out_dir / "coco_instances_results.json")
    metrics_path = out_dir / "metrics.cocoeval.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if stats_accumulator is not None:
        stats_path = stats_accumulator.dump(args.dump_inference_stats)
        print(f"[Eval] inference_stats_path={stats_path}")
    print(f"[Eval] results_path={results_path}")
    print(f"[Eval] metrics_path={metrics_path}")
    print("[Eval] metrics_json=" + json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
