#!/usr/bin/env python3
"""Diagnose R82 4-view TTA pseudo-bank candidate coverage without training."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from magformer.config import load_config, set_seed  # noqa: E402
from magformer.config.loader import load_yaml_file, save_yaml_file  # noqa: E402
from magformer.config.schema import merge_configs  # noqa: E402
from magformer.data import CocoRgbdDataset  # noqa: E402
from magformer.data.collate import collate_fn  # noqa: E402
from magformer.models import build_model  # noqa: E402
from tools.diagnose_vc_suda_pseudo_labels import (  # noqa: E402
    _ann_box,
    _bbox_iou,
    _load_gt_buckets,
    _load_r78_stats,
    _matched_prediction_indices,
)
from tools.evaluate_1024_backmap import (  # noqa: E402
    Eval1024Transform,
    category_id_from_contiguous,
    content_bounds,
    mask_to_prob,
    strict_load_with_evidence,
)

TTA_VIEWS: tuple[tuple[float, bool], ...] = ((1.0, False), (1.0, True), (1.25, False), (1.25, True))
R81_BASELINE = {
    "tiny_area_le_256": {"candidate_gt_coverage_iou50": 0.326, "candidate_gt_coverage_iou75": 0.050, "kept_gt_coverage_iou50": 0.211, "kept_gt_coverage_iou75": 0.031},
    "bottom20_area": {"candidate_gt_coverage_iou50": 0.286, "candidate_gt_coverage_iou75": 0.039, "kept_gt_coverage_iou50": 0.186, "kept_gt_coverage_iou75": 0.025},
}
R82_GATES = {
    "tiny_area_le_256": {"candidate_gt_coverage_iou50": 0.400, "candidate_gt_coverage_iou75": 0.080, "kept_gt_coverage_iou50": 0.270, "kept_gt_coverage_iou75": 0.045},
    "bottom20_area": {"candidate_gt_coverage_iou50": 0.350, "candidate_gt_coverage_iou75": 0.065, "kept_gt_coverage_iou50": 0.240, "kept_gt_coverage_iou75": 0.040},
}


class R82DiagnosisError(RuntimeError):
    pass


@dataclass(frozen=True)
class Candidate:
    quality: float
    category_id: int
    bbox: list[float]
    mask: np.ndarray
    source_view: str


class nullcontext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc_info: Any) -> None:
        return None


def _resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    return (REPO_ROOT / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()


def _require_finite_float(value: Any, name: str) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise R82DiagnosisError(f"{name} is non-finite: {value!r}")
    return out


def _mean(values: list[float]) -> float | None:
    return None if not values else _require_finite_float(sum(values) / len(values), "mean")


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "p10": None, "p50": None, "p90": None}
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise R82DiagnosisError("distribution contains non-finite values")
    return {"count": int(array.size), "mean": float(array.mean()), "p10": float(np.quantile(array, 0.10)), "p50": float(np.quantile(array, 0.50)), "p90": float(np.quantile(array, 0.90))}


def _add_distribution_fields(target: dict[str, Any], prefix: str, values: list[float]) -> None:
    dist = _distribution(values)
    for key, value in dist.items():
        target[f"{prefix}_{key}"] = value


def _bbox_xyxy_from_binary(binary: np.ndarray) -> list[float] | None:
    ys, xs = np.where(binary > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = np.logical_and(mask_a > 0, mask_b > 0).sum(dtype=np.float64)
    union = np.logical_or(mask_a > 0, mask_b > 0).sum(dtype=np.float64)
    return 0.0 if union <= 0 else _require_finite_float(inter / union, "mask_iou")


def _candidate_mask_area(candidate: Candidate) -> float:
    return _require_finite_float(float((candidate.mask > 0).sum()), "mask_area")


def union_cluster_candidates(candidates: list[Candidate], *, mask_iou_threshold: float = 0.55, bbox_iou_threshold: float = 0.75) -> list[Candidate]:
    if not candidates:
        return []
    remaining = sorted(range(len(candidates)), key=lambda idx: candidates[idx].quality, reverse=True)
    merged: list[Candidate] = []
    while remaining:
        seed_idx = remaining[0]
        seed = candidates[seed_idx]
        cluster_indices = [seed_idx]
        next_remaining: list[int] = []
        for idx in remaining[1:]:
            item = candidates[idx]
            if item.category_id != seed.category_id:
                next_remaining.append(idx)
                continue
            if _bbox_iou(seed.bbox, item.bbox) >= bbox_iou_threshold or _mask_iou(seed.mask, item.mask) >= mask_iou_threshold:
                cluster_indices.append(idx)
            else:
                next_remaining.append(idx)
        merged.append(candidates[max(cluster_indices, key=lambda idx: candidates[idx].quality)])
        remaining = next_remaining
    return sorted(merged, key=lambda item: item.quality, reverse=True)


def _empty_bucket() -> dict[str, Any]:
    return {"images": 0, "gt_count": 0, "candidate_count": 0, "kept_count": 0, "candidate_gt_iou50": 0, "candidate_gt_iou75": 0, "kept_gt_iou50": 0, "kept_gt_iou75": 0, "_candidate_qualities": [], "_kept_qualities": [], "_kept_mask_areas": [], "_candidate_best_ious": [], "_kept_best_ious": []}


def _add_gt_coverage(bucket: dict[str, Any], gt_anns: list[dict[str, Any]], gt_image: dict[str, Any], target_size: tuple[int, int], candidate_boxes: list[list[float]], kept_boxes: list[list[float]]) -> None:
    for ann in gt_anns:
        gt_box = _ann_box(ann, gt_image, target_size)
        candidate_best = max((_bbox_iou(gt_box, box) for box in candidate_boxes), default=0.0)
        kept_best = max((_bbox_iou(gt_box, box) for box in kept_boxes), default=0.0)
        bucket["_candidate_best_ious"].append(candidate_best)
        bucket["_kept_best_ious"].append(kept_best)
        bucket["candidate_gt_iou50"] += int(candidate_best >= 0.50)
        bucket["candidate_gt_iou75"] += int(candidate_best >= 0.75)
        bucket["kept_gt_iou50"] += int(kept_best >= 0.50)
        bucket["kept_gt_iou75"] += int(kept_best >= 0.75)


def _add_image_to_bucket(bucket: dict[str, Any], gt_anns: list[dict[str, Any]], gt_image: dict[str, Any], target_size: tuple[int, int], candidates: list[Candidate], kept: list[Candidate]) -> None:
    bucket["images"] += 1
    bucket["gt_count"] += len(gt_anns)
    bucket["candidate_count"] += len(candidates)
    bucket["kept_count"] += len(kept)
    bucket["_candidate_qualities"].extend(item.quality for item in candidates)
    bucket["_kept_qualities"].extend(item.quality for item in kept)
    bucket["_kept_mask_areas"].extend(_candidate_mask_area(item) for item in kept)
    _add_gt_coverage(bucket, gt_anns, gt_image, target_size, [item.bbox for item in candidates], [item.bbox for item in kept])


def _finalize_bucket(bucket: dict[str, Any]) -> dict[str, Any]:
    candidate_qualities = bucket.pop("_candidate_qualities")
    kept_qualities = bucket.pop("_kept_qualities")
    kept_mask_areas = bucket.pop("_kept_mask_areas")
    candidate_best_ious = bucket.pop("_candidate_best_ious")
    kept_best_ious = bucket.pop("_kept_best_ious")
    gt_count = int(bucket["gt_count"])
    candidate_count = int(bucket["candidate_count"])
    kept_count = int(bucket["kept_count"])
    bucket["keep_rate"] = float(kept_count / candidate_count) if candidate_count else None
    bucket["candidate_mean_quality"] = _mean(candidate_qualities)
    bucket["kept_mean_quality"] = _mean(kept_qualities)
    _add_distribution_fields(bucket, "candidate_quality", candidate_qualities)
    _add_distribution_fields(bucket, "kept_quality", kept_qualities)
    _add_distribution_fields(bucket, "kept_mask_area", kept_mask_areas)
    for prefix in ("candidate", "kept"):
        bucket[f"{prefix}_gt_coverage_iou50"] = float(bucket.pop(f"{prefix}_gt_iou50") / gt_count) if gt_count else None
        bucket[f"{prefix}_gt_coverage_iou75"] = float(bucket.pop(f"{prefix}_gt_iou75") / gt_count) if gt_count else None
    bucket["candidate_gt_best_iou_mean"] = _mean(candidate_best_ious)
    bucket["kept_gt_best_iou_mean"] = _mean(kept_best_ious)
    return bucket


def summarize_candidate_bank_buckets(per_image: list[dict[str, Any]], *, gt_by_image: dict[int, dict[str, Any]], r78_by_image: dict[int, dict[str, Any]], bottom20_threshold: float, target_ann_path: str | Path, r78_stats_path: str | Path) -> dict[str, Any]:
    bucket_names = ("overall", "normal", "dense", "dense_tiny", "tiny_area_le_256", "bottom20_area")
    buckets = {name: _empty_bucket() for name in bucket_names}
    sampled_image_ids: list[int] = []
    sampled_image_files: list[str] = []
    for record in per_image:
        image_id = int(record["image_id"])
        if image_id not in gt_by_image or image_id not in r78_by_image:
            raise R82DiagnosisError(f"image_id={image_id} missing from target annotation or R78 stats")
        candidates: list[Candidate] = record["candidates"]
        kept: list[Candidate] = record["kept"]
        gt_all = gt_by_image[image_id]["all"]
        gt_image = gt_by_image[image_id]["image"]
        target_size = tuple(int(v) for v in record["target_size"])
        sampled_image_ids.append(image_id)
        sampled_image_files.append(str(r78_by_image[image_id].get("file_name", "")))
        _add_image_to_bucket(buckets["overall"], gt_all, gt_image, target_size, candidates, kept)
        _add_image_to_bucket(buckets[str(r78_by_image[image_id]["r78_bucket"])], gt_all, gt_image, target_size, candidates, kept)
        for gt_bucket_name in ("tiny_area_le_256", "bottom20_area"):
            gt_subset = gt_by_image[image_id][gt_bucket_name]
            if not gt_subset:
                continue
            matched_candidates = _matched_prediction_indices([item.bbox for item in candidates], gt_subset, gt_image, target_size, min_iou=0.50)
            matched_kept = _matched_prediction_indices([item.bbox for item in kept], gt_subset, gt_image, target_size, min_iou=0.50)
            sub_candidates = [item for idx, item in enumerate(candidates) if idx in matched_candidates]
            sub_kept = [item for idx, item in enumerate(kept) if idx in matched_kept]
            bucket = buckets[gt_bucket_name]
            bucket["images"] += 1
            bucket["gt_count"] += len(gt_subset)
            bucket["candidate_count"] += len(sub_candidates)
            bucket["kept_count"] += len(sub_kept)
            bucket["_candidate_qualities"].extend(item.quality for item in sub_candidates)
            bucket["_kept_qualities"].extend(item.quality for item in sub_kept)
            bucket["_kept_mask_areas"].extend(_candidate_mask_area(item) for item in sub_kept)
            _add_gt_coverage(bucket, gt_subset, gt_image, target_size, [item.bbox for item in candidates], [item.bbox for item in kept])
    if not sampled_image_ids:
        raise R82DiagnosisError("bucket summary saw zero images")
    return {"sampled_images": len(sampled_image_ids), "sampled_image_ids": sampled_image_ids, "sampled_image_files": sampled_image_files, "target_unlabeled_ann": str(target_ann_path), "r78_stats": str(r78_stats_path), "bottom20_area_threshold": float(bottom20_threshold), "bucket_semantics": {"normal_dense_dense_tiny": "image-level R78/R52 prediction-only buckets", "tiny_area_le_256": "GT-area proxy; candidate/kept counts are predictions with bbox IoU >= 0.50 to a bucket GT object", "bottom20_area": "GT-area proxy; candidate/kept counts are predictions with bbox IoU >= 0.50 to a bottom-20% GT object", "coverage": "bbox IoU proxy against target_unlabeled GT; training still receives no GT labels", "quality": "R80 teacher postprocess score; cluster quality is max score across joined views", "union": "same-class diagnostic cluster when mask IoU >= 0.55 OR bbox IoU >= 0.75; representative mask is the highest-quality member"}, "buckets": {name: _finalize_bucket(bucket) for name, bucket in buckets.items()}}


def _candidate_to_json(candidate: Candidate) -> dict[str, Any]:
    return {"score": float(candidate.quality), "category_id": int(candidate.category_id), "bbox": [float(v) for v in candidate.bbox], "mask_area": int((candidate.mask > 0).sum()), "source_view": candidate.source_view}


def _prediction_masks(pred: dict[str, Any]) -> tuple[Any, bool]:
    if "mask_logits" in pred:
        return pred["mask_logits"], True
    if "mask_probs" in pred:
        return pred["mask_probs"], False
    return pred["masks"], False


@torch.no_grad()
def _run_view(model: torch.nn.Module, images: torch.Tensor, depths: torch.Tensor, padding_masks: torch.Tensor | None, *, depth_valid_masks: torch.Tensor | None, scale: float, hflip: bool, device: torch.device, amp_enabled: bool, inference_topk: int) -> dict[str, Any]:
    aug_images = images
    aug_depths = depths
    aug_padding = padding_masks
    aug_depth_valid_masks = depth_valid_masks
    base_h, base_w = images.shape[-2:]
    if scale != 1.0:
        size = (int(round(base_h * scale)), int(round(base_w * scale)))
        aug_images = F.interpolate(aug_images, size=size, mode="bilinear", align_corners=False)
        aug_depths = F.interpolate(aug_depths, size=size, mode="bilinear", align_corners=False)
        if torch.is_tensor(aug_padding):
            aug_padding = F.interpolate(aug_padding.float().unsqueeze(1), size=size, mode="nearest").squeeze(1).bool()
        if torch.is_tensor(aug_depth_valid_masks):
            aug_depth_valid_masks = F.interpolate(aug_depth_valid_masks.float(), size=size, mode="nearest").bool()
    if hflip:
        aug_images = torch.flip(aug_images, [-1])
        aug_depths = torch.flip(aug_depths, [-1])
        if torch.is_tensor(aug_padding):
            aug_padding = torch.flip(aug_padding, [-1])
        if torch.is_tensor(aug_depth_valid_masks):
            aug_depth_valid_masks = torch.flip(aug_depth_valid_masks, [-1])
    ctx = autocast("cuda") if amp_enabled and device.type == "cuda" else nullcontext()
    with ctx:
        return model.forward_inference_raw(aug_images, aug_depths, padding_masks=aug_padding, depth_valid_masks=aug_depth_valid_masks, include_raw_tensors=True, move_predictions_to_cpu=False, inference_topk=inference_topk)


def _view_candidates_to_gt(pred: dict[str, Any], *, image_id: int, content_mask: Any, out_h: int, out_w: int, category_ids: list[int] | None, scale: float, hflip: bool, score_threshold: float, mask_threshold: float, base_size: tuple[int, int]) -> list[Candidate]:
    base_h, base_w = base_size
    scores = pred.get("scores")
    cats = pred.get("category_ids", pred.get("labels"))
    raw_masks, masks_are_logits = _prediction_masks(pred)
    if scores is None or cats is None or raw_masks is None:
        raise R82DiagnosisError(f"prediction for image_id={image_id} lacks scores/category_ids/masks")
    scores_np = scores.detach().float().cpu().numpy()
    cats_np = cats.detach().cpu().numpy()
    masks_tensor = raw_masks.detach().float()
    if masks_are_logits:
        masks_tensor = masks_tensor.sigmoid()
    if masks_tensor.ndim == 2:
        masks_tensor = masks_tensor.unsqueeze(0)
    if masks_tensor.shape[0] != scores_np.shape[0]:
        raise R82DiagnosisError(f"mask/score count mismatch for image_id={image_id}")
    if masks_tensor.shape[-2:] != (base_h, base_w):
        masks_tensor = F.interpolate(masks_tensor.unsqueeze(1), size=(base_h, base_w), mode="bilinear", align_corners=False).squeeze(1)
    if hflip:
        masks_tensor = torch.flip(masks_tensor, [-1])
    top, bottom, left, right = content_bounds(content_mask, image_id)
    content_h = bottom - top
    content_w = right - left
    view_name = f"scale{scale:g}_{'hflip' if hflip else 'noflip'}"
    candidates: list[Candidate] = []
    for idx, score_value in enumerate(scores_np.tolist()):
        quality = _require_finite_float(score_value, f"score[{idx}] image_id={image_id}")
        if quality < score_threshold:
            continue
        prob = mask_to_prob(masks_tensor[idx])
        if prob.shape != (base_h, base_w):
            prob = cv2.resize(prob, (base_w, base_h), interpolation=cv2.INTER_LINEAR)
        prob = prob[top:bottom, left:right]
        if prob.shape != (content_h, content_w):
            raise R82DiagnosisError(f"content crop failed for image_id={image_id}: got {prob.shape}, expected {(content_h, content_w)}")
        if prob.shape != (out_h, out_w):
            prob = cv2.resize(prob, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        binary = (prob > mask_threshold).astype(np.uint8)
        bbox = _bbox_xyxy_from_binary(binary)
        if bbox is None:
            continue
        contiguous_id = int(cats_np[idx]) if idx < len(cats_np) else 0
        candidates.append(Candidate(quality=quality, category_id=category_id_from_contiguous(contiguous_id, category_ids), bbox=bbox, mask=binary, source_view=view_name))
    return candidates


def _evaluate_gates(bucket_summary: dict[str, Any]) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    passed = True
    for bucket_name, gates in R82_GATES.items():
        bucket = bucket_summary["buckets"][bucket_name]
        row: dict[str, Any] = {}
        for metric, threshold in gates.items():
            value = bucket.get(metric)
            metric_passed = value is not None and float(value) >= float(threshold)
            row[metric] = {"value": value, "threshold": float(threshold), "passed": bool(metric_passed), "r81": R81_BASELINE[bucket_name][metric], "delta_vs_r81": None if value is None else float(value) - R81_BASELINE[bucket_name][metric]}
            passed = passed and metric_passed
        rows[bucket_name] = row
    return {"passed": bool(passed), "rows": rows, "r81_baseline": R81_BASELINE, "r82_gates": R82_GATES}


def _format_cov(bucket: dict[str, Any], prefix: str) -> str:
    cov50 = bucket[f"{prefix}_gt_coverage_iou50"]
    cov75 = bucket[f"{prefix}_gt_coverage_iou75"]
    return "NA / NA" if cov50 is None or cov75 is None else f"{cov50:.3f} / {cov75:.3f}"


def write_markdown_report(path: Path, *, summary: dict[str, Any], bucket_summary: dict[str, Any], gate: dict[str, Any]) -> None:
    lines = ["# VC-SUDA R82 4-view TTA candidate bank diagnosis, 2026-05-17", "", "R82 is a no-train diagnostic for the R80 teacher. It checks whether the fixed 4-view TTA union bank improves tiny and bottom20 candidate coverage enough to justify training.", "", "## Scope", "", "- Host: `4029`.", "- Project: `/home/hdd3/zhanghaonan/magformer`.", f"- Config: `{summary['config']}`.", f"- Weights: `{summary['weights']}`.", f"- Target ann: `{summary['target_ann']}`.", f"- GPU: `{summary['device']}` with `CUDA_VISIBLE_DEVICES={summary['cuda_visible_devices']}`.", "- Views: `1.0 noflip`, `1.0 hflip`, `1.25 noflip`, `1.25 hflip`.", f"- Union: same class, mask IoU `>=0.55` or bbox IoU `>=0.75`; quality is max; kept threshold is `{summary['kept_threshold']}`.", "", "No training was run. Training logic was not changed.", "", "## Candidate Coverage", "", "| bucket | images | GT/proxy | candidates | kept | keep-rate | candidate cov@50/@75 | kept cov@50/@75 | quality mean/p10/p50/p90 | kept area mean/p10/p50/p90 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, bucket in bucket_summary["buckets"].items():
        keep_rate = "NA" if bucket["keep_rate"] is None else f"{bucket['keep_rate']:.3f}"
        quality = f"{bucket['candidate_quality_mean']:.3f} / {bucket['candidate_quality_p10']:.3f} / {bucket['candidate_quality_p50']:.3f} / {bucket['candidate_quality_p90']:.3f}" if bucket["candidate_quality_mean"] is not None else "NA / NA / NA / NA"
        area = f"{bucket['kept_mask_area_mean']:.0f} / {bucket['kept_mask_area_p10']:.0f} / {bucket['kept_mask_area_p50']:.0f} / {bucket['kept_mask_area_p90']:.0f}" if bucket["kept_mask_area_mean"] is not None else "NA / NA / NA / NA"
        lines.append(f"| {name} | {bucket['images']} | {bucket['gt_count']} | {bucket['candidate_count']} | {bucket['kept_count']} | `{keep_rate}` | `{_format_cov(bucket, 'candidate')}` | `{_format_cov(bucket, 'kept')}` | `{quality}` | `{area}` |")
    lines.extend(["", "## R82 Gate", "", "| bucket | metric | R81 | R82 | delta | gate | pass |", "|---|---|---:|---:|---:|---:|---:|"])
    for bucket_name, metrics in gate["rows"].items():
        for metric, row in metrics.items():
            value = row["value"]
            value_s = "NA" if value is None else f"{value:.3f}"
            delta = row["delta_vs_r81"]
            delta_s = "NA" if delta is None else f"{delta:+.3f}"
            lines.append(f"| {bucket_name} | {metric} | `{row['r81']:.3f}` | `{value_s}` | `{delta_s}` | `{row['threshold']:.3f}` | `{row['passed']}` |")
    decision = "R82 passes the coverage gate. The next step may train from this TTA bank." if gate["passed"] else "R82 does not pass the coverage gate. Stop the current R80 TTA pseudo-bank route."
    lines.extend(["", "## Decision", "", decision, "", "## Artifacts", "", f"- Bucket JSON: `{summary['bucket_json']}`.", f"- Summary JSON: `{summary['summary_json']}`.", f"- Candidate JSON: `{summary['candidate_json']}`.", f"- Runtime config: `{summary['runtime_config']}`."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", default="configs/vc_suda_stage_c_r80_unsup_schedule_1024.yaml")
    parser.add_argument("--weights", default="output/vc_suda/stage_c_r80_unsup_schedule_1024/checkpoint_iter_0000750.pth")
    parser.add_argument("--dataset-root", default="magformer_datasets/pseudo_real_512")
    parser.add_argument("--target-ann", default="annotations/instances_target_unlabeled.json")
    parser.add_argument("--split", default="train")
    parser.add_argument("--r78-stats", default="output/diagnostics/r52_target_unlabeled_sampling_20260516/target_sampling_stats.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-images", type=int, default=20)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--inference-topk", type=int, default=100)
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--kept-threshold", type=float, default=0.1)
    parser.add_argument("--report-interval", type=int, default=20)
    parser.add_argument("--bucket-json", default=None)
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--candidate-json", default=None)
    parser.add_argument("--markdown", default=None)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.batch_size != 1:
        raise R82DiagnosisError("R82 TTA candidate diagnosis requires --batch-size 1")
    if args.max_images <= 0:
        raise R82DiagnosisError("--max-images must be positive")
    if args.kept_threshold != 0.1:
        raise R82DiagnosisError("R82 fixed diagnostic requires --kept-threshold 0.1")
    os.chdir(REPO_ROOT)
    out_dir = _resolve_project_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bucket_json = _resolve_project_path(args.bucket_json) if args.bucket_json else out_dir / "r82_tta_candidate_buckets.json"
    summary_json = _resolve_project_path(args.summary_json) if args.summary_json else out_dir / "r82_tta_candidate_summary.json"
    candidate_json = _resolve_project_path(args.candidate_json) if args.candidate_json else out_dir / "r82_tta_candidate_predictions.json"
    markdown = _resolve_project_path(args.markdown) if args.markdown else out_dir / "r82_tta_candidate_bank.md"
    overrides = {"data": {"dataset_root": args.dataset_root, "image_size": args.image_size, "min_scale": 1.0, "max_scale": 1.0}, "model": {"weights": args.weights, "finetune_weights": args.weights}, "runtime": {"device": "cuda" if str(args.device).startswith("cuda") else "cpu", "gpus": [0], "ddp_enabled": False, "num_workers": args.num_workers, "output_dir": str(out_dir)}}
    runtime_config = out_dir / "r82_tta_candidate_runtime.yaml"
    save_yaml_file(merge_configs(load_yaml_file(args.base_config), overrides), str(runtime_config))
    cfg = load_config(str(runtime_config))
    set_seed(int(cfg.runtime.seed))
    random.seed(int(cfg.runtime.seed))
    np.random.seed(int(cfg.runtime.seed))
    torch.manual_seed(int(cfg.runtime.seed))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise R82DiagnosisError(f"CUDA device requested but unavailable: {args.device}")
    weights = _resolve_project_path(args.weights)
    if not weights.exists():
        raise R82DiagnosisError(f"weights not found: {weights}")
    dataset = CocoRgbdDataset(dataset_root=args.dataset_root, ann_file=args.target_ann, split=args.split, transform=None, is_train=False)
    if args.max_images > len(dataset):
        raise R82DiagnosisError(f"--max-images={args.max_images} exceeds dataset images={len(dataset)}")
    depth_cfg = cfg.data.depth
    dataset.transform = Eval1024Transform(image_size=cfg.data.image_size, depth_scale=depth_cfg.scale, depth_shift=depth_cfg.shift, depth_clip_min=depth_cfg.clip_min, depth_clip_max=depth_cfg.clip_max, depth_norm=depth_cfg.norm, depth_per_sample_norm=depth_cfg.per_sample_norm)
    image_size_by_id = {int(img_id): (int(info["height"]), int(info["width"])) for img_id, info in dataset.coco.imgs.items()}
    category_ids = list(getattr(dataset, "category_ids", [])) or None
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type == "cuda"), collate_fn=collate_fn)
    model = build_model(cfg)
    strict_load_with_evidence(model, str(weights))
    model = model.to(device)
    model.eval()
    target_ann_path = _resolve_project_path(Path(args.dataset_root) / args.target_ann)
    r78_stats_path = _resolve_project_path(args.r78_stats)
    gt_by_image, bottom20_threshold = _load_gt_buckets(target_ann_path)
    r78_by_image = _load_r78_stats(r78_stats_path)
    print("[R82] " + json.dumps({"views": [(scale, hflip) for scale, hflip in TTA_VIEWS], "device": str(device), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "max_images": args.max_images, "inference_topk": args.inference_topk, "score_threshold": args.score_threshold, "kept_threshold": args.kept_threshold}, sort_keys=True), flush=True)
    per_image: list[dict[str, Any]] = []
    candidate_dump: list[dict[str, Any]] = []
    raw_candidate_count = 0
    start = time.time()
    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= args.max_images:
                break
            image_id = int(batch["image_ids"][0].item())
            out_h, out_w = image_size_by_id[image_id]
            images = batch["images"].to(device, non_blocking=True)
            depths = batch["depths"].to(device, non_blocking=True)
            padding_masks = batch.get("padding_masks")
            if torch.is_tensor(padding_masks):
                padding_masks = padding_masks.to(device, non_blocking=True)
            depth_valid_masks = batch.get("depth_valid_masks")
            if torch.is_tensor(depth_valid_masks):
                depth_valid_masks = depth_valid_masks.to(device, non_blocking=True)
            content_masks = batch.get("content_masks")
            if content_masks is None:
                raise R82DiagnosisError("batch lacks content_masks; refusing padded-coordinate coverage")
            view_candidates: list[Candidate] = []
            for scale, hflip in TTA_VIEWS:
                outputs = _run_view(model, images, depths, padding_masks, depth_valid_masks=depth_valid_masks, scale=scale, hflip=hflip, device=device, amp_enabled=bool(args.amp), inference_topk=int(args.inference_topk))
                view_candidates.extend(_view_candidates_to_gt(outputs["predictions"][0], image_id=image_id, content_mask=content_masks[0], out_h=out_h, out_w=out_w, category_ids=category_ids, scale=scale, hflip=hflip, score_threshold=float(args.score_threshold), mask_threshold=float(args.mask_threshold), base_size=tuple(int(v) for v in images.shape[-2:])))
                del outputs
            if not view_candidates:
                raise R82DiagnosisError(f"zero raw view candidates for image_id={image_id}")
            raw_candidate_count += len(view_candidates)
            merged = union_cluster_candidates(view_candidates)
            kept = [item for item in merged if item.quality >= float(args.kept_threshold)]
            if not merged:
                raise R82DiagnosisError(f"zero union candidates for image_id={image_id}")
            per_image.append({"image_id": image_id, "target_size": (out_h, out_w), "candidates": merged, "kept": kept})
            candidate_dump.append({"image_id": image_id, "raw_view_candidates": len(view_candidates), "union_candidates": len(merged), "kept": len(kept), "candidates": [_candidate_to_json(item) for item in merged]})
            if (batch_idx + 1) % int(args.report_interval) == 0 or (batch_idx + 1) == args.max_images:
                elapsed = time.time() - start
                print(f"[R82] processed={batch_idx + 1}/{args.max_images} raw={raw_candidate_count} union={sum(len(r['candidates']) for r in per_image)} kept={sum(len(r['kept']) for r in per_image)} seconds={elapsed:.1f}", flush=True)
            del images, depths
            if device.type == "cuda":
                torch.cuda.empty_cache()
    bucket_summary = summarize_candidate_bank_buckets(per_image, gt_by_image=gt_by_image, r78_by_image=r78_by_image, bottom20_threshold=bottom20_threshold, target_ann_path=target_ann_path, r78_stats_path=r78_stats_path)
    gate = _evaluate_gates(bucket_summary)
    summary = {"config": str(_resolve_project_path(args.base_config)), "runtime_config": str(runtime_config), "weights": str(weights), "target_ann": str(target_ann_path), "r78_stats": str(r78_stats_path), "device": str(device), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "max_images": int(args.max_images), "views": [{"scale": scale, "hflip": hflip} for scale, hflip in TTA_VIEWS], "raw_view_candidates": int(raw_candidate_count), "union_candidates": int(sum(len(record["candidates"]) for record in per_image)), "kept": int(sum(len(record["kept"]) for record in per_image)), "score_threshold": float(args.score_threshold), "mask_threshold": float(args.mask_threshold), "kept_threshold": float(args.kept_threshold), "inference_topk": int(args.inference_topk), "seconds": float(time.time() - start), "bucket_json": str(bucket_json), "summary_json": str(summary_json), "candidate_json": str(candidate_json), "gate": gate}
    bucket_json.write_text(json.dumps(bucket_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    candidate_json.write_text(json.dumps(candidate_dump, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown_report(markdown, summary=summary, bucket_summary=bucket_summary, gate=gate)
    print("[R82] bucket_json=" + str(bucket_json), flush=True)
    print("[R82] summary_json=" + str(summary_json), flush=True)
    print("[R82] candidate_json=" + str(candidate_json), flush=True)
    print("[R82] markdown=" + str(markdown), flush=True)
    print("[R82] gate_json=" + json.dumps(gate, sort_keys=True), flush=True)
    return summary


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        run(args)
    except R82DiagnosisError as exc:
        print(f"[R82][ERROR] {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
