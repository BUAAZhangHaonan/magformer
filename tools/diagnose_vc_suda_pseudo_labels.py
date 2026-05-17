#!/usr/bin/env python3
"""Diagnose VC-SUDA Stage C pseudo-label keep-rate before training.

The gate loads a Stage C config, warm-starts the teacher from
``model.finetune_weights`` or ``--weights``, samples a small
``target_unlabeled`` batch, runs teacher eval forward, then reuses
``PseudoLabelScorer.score`` and ``filter_by_threshold`` from the training path.

Safe smoke usage:

    python tools/diagnose_vc_suda_pseudo_labels.py \
      --config configs/vc_suda_stage_c_1024_teacher8499.yaml \
      --max-images 2 \
      --device cpu

It fails fast when no scored predictions are produced, or when the configured
threshold fails the keep-rate or empty-image gates.
"""

from __future__ import annotations

import argparse
import math
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
from pycocotools import mask as coco_mask

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from magformer.config import load_config, set_seed  # noqa: E402
from magformer.models.common.pseudo_label_scorer import PseudoLabelScorer  # noqa: E402
from tools import train as train_tool  # noqa: E402

DEFAULT_THRESHOLD_SWEEP = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7)
DEFAULT_MIN_KEEP_RATE = 0.10
DEFAULT_MAX_EMPTY_RATIO = 0.05


class PseudoLabelDiagnosticsError(RuntimeError):
    """Raised when the Stage C pseudo-label diagnostic gate fails."""

    def __init__(self, message: str, summary: dict[str, Any] | None = None):
        super().__init__(message)
        self.summary = summary


def _resolve_project_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


def _cfg_get(obj: Any, key: str, default: Any = None) -> Any:
    return train_tool._cfg_get(obj, key, default)


def _tensor_count(result: dict[str, Any]) -> int:
    scores = result.get("scores")
    if scores is None:
        return 0
    return int(scores.numel())


def _scores_to_cpu(scored_results: list[dict[str, Any]]) -> torch.Tensor:
    tensors = []
    for result in scored_results:
        scores = result.get("scores")
        if torch.is_tensor(scores) and scores.numel() > 0:
            tensors.append(scores.detach().float().cpu())
    if not tensors:
        return torch.empty(0, dtype=torch.float32)
    return torch.cat(tensors)


def _score_distribution(scores: torch.Tensor) -> dict[str, float | int]:
    if scores.numel() == 0:
        return {"count": 0}
    quantiles = torch.quantile(
        scores,
        torch.tensor([0.25, 0.5, 0.75, 0.9, 0.95, 0.99], dtype=torch.float32),
    )
    return {
        "count": int(scores.numel()),
        "min": float(scores.min().item()),
        "mean": float(scores.mean().item()),
        "p25": float(quantiles[0].item()),
        "median": float(quantiles[1].item()),
        "p75": float(quantiles[2].item()),
        "p90": float(quantiles[3].item()),
        "p95": float(quantiles[4].item()),
        "p99": float(quantiles[5].item()),
        "max": float(scores.max().item()),
    }


def _mean_float(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _finite_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    if not math.isfinite(value):
        raise PseudoLabelDiagnosticsError(f"non-finite diagnostic value: {value}")
    return float(value)


def _resolve_target_ann_path(cfg: Any) -> Path:
    ann_path = _resolve_project_path(cfg.vc_suda.target_unlabeled_ann)
    if ann_path is not None and ann_path.exists():
        return ann_path
    dataset_root = _resolve_project_path(cfg.data.dataset_root)
    if dataset_root is None:
        raise PseudoLabelDiagnosticsError("data.dataset_root is required for target annotation diagnostics")
    candidate = dataset_root / str(cfg.vc_suda.target_unlabeled_ann)
    if not candidate.exists():
        raise PseudoLabelDiagnosticsError(f"target_unlabeled annotation not found: {candidate}")
    return candidate


def _load_r78_stats(path: str | Path | None) -> dict[int, dict[str, Any]]:
    stats_path = _resolve_project_path(path)
    if stats_path is None:
        raise PseudoLabelDiagnosticsError("--r78-stats is required when --bucket-output-json is used")
    if not stats_path.exists():
        raise PseudoLabelDiagnosticsError(f"R78/R52 bucket stats not found: {stats_path}")
    payload = json.loads(stats_path.read_text(encoding="utf-8"))
    rows = payload.get("images")
    if not isinstance(rows, list):
        raise PseudoLabelDiagnosticsError(f"bucket stats must contain an images list: {stats_path}")
    by_id: dict[int, dict[str, Any]] = {}
    for idx, row in enumerate(rows):
        if not isinstance(row, dict) or "image_id" not in row:
            raise PseudoLabelDiagnosticsError(f"bucket stats images[{idx}] must contain image_id")
        image_id = int(row["image_id"])
        if bool(row.get("dense_tiny_candidate", False)):
            bucket = "dense_tiny"
        elif bool(row.get("dense_candidate", False)) or row.get("bucket") == "dense":
            bucket = "dense"
        else:
            bucket = "normal"
        by_id[image_id] = {**row, "r78_bucket": bucket}
    if not by_id:
        raise PseudoLabelDiagnosticsError(f"bucket stats contain no images: {stats_path}")
    return by_id


def _ann_area(ann: dict[str, Any], image: dict[str, Any]) -> float:
    if "area" in ann:
        return float(ann["area"])
    segmentation = ann.get("segmentation")
    if segmentation is None:
        bbox = ann.get("bbox")
        if not bbox:
            raise PseudoLabelDiagnosticsError(f"annotation {ann.get('id')} has neither area nor bbox")
        return float(bbox[2]) * float(bbox[3])
    rle = segmentation
    if isinstance(segmentation, list):
        rle = coco_mask.frPyObjects(segmentation, int(image["height"]), int(image["width"]))
        rle = coco_mask.merge(rle)
    return float(coco_mask.area(rle))


def _load_gt_buckets(ann_path: Path) -> tuple[dict[int, dict[str, Any]], float]:
    payload = json.loads(ann_path.read_text(encoding="utf-8"))
    images = {int(img["id"]): img for img in payload.get("images", [])}
    anns_by_image: dict[int, list[dict[str, Any]]] = {image_id: [] for image_id in images}
    areas: list[float] = []
    for ann in payload.get("annotations", []):
        if int(ann.get("iscrowd", 0)) != 0:
            continue
        image_id = int(ann["image_id"])
        if image_id not in images:
            raise PseudoLabelDiagnosticsError(f"annotation references missing image_id={image_id}")
        area = _ann_area(ann, images[image_id])
        row = {**ann, "_area": float(area)}
        anns_by_image.setdefault(image_id, []).append(row)
        areas.append(float(area))
    if not images:
        raise PseudoLabelDiagnosticsError(f"target annotation contains no images: {ann_path}")
    if not areas:
        raise PseudoLabelDiagnosticsError(f"target annotation contains no non-crowd annotations: {ann_path}")
    areas_sorted = sorted(areas)
    cutoff_index = max(0, min(len(areas_sorted) - 1, int(math.ceil(len(areas_sorted) * 0.2)) - 1))
    bottom20_threshold = float(areas_sorted[cutoff_index])
    return (
        {
            image_id: {
                "image": image,
                "all": anns_by_image.get(image_id, []),
                "tiny_area_le_256": [
                    ann for ann in anns_by_image.get(image_id, []) if float(ann["_area"]) <= 256.0
                ],
                "bottom20_area": [
                    ann for ann in anns_by_image.get(image_id, []) if float(ann["_area"]) <= bottom20_threshold
                ],
            }
            for image_id, image in images.items()
        },
        bottom20_threshold,
    )


def _boxes_from_masks(result: dict[str, Any], *, threshold: float = 0.5) -> list[list[float]]:
    masks = result.get("masks")
    if not torch.is_tensor(masks) or masks.numel() == 0:
        return []
    hard = masks.detach() > threshold
    boxes: list[list[float]] = []
    for mask in hard:
        rows = torch.any(mask, dim=1)
        cols = torch.any(mask, dim=0)
        if not bool(rows.any().item()) or not bool(cols.any().item()):
            boxes.append([0.0, 0.0, 0.0, 0.0])
            continue
        ys = torch.where(rows)[0]
        xs = torch.where(cols)[0]
        boxes.append(
            [
                float(xs[0].item()),
                float(ys[0].item()),
                float(xs[-1].item() + 1),
                float(ys[-1].item() + 1),
            ]
        )
    return boxes


def _ann_box(
    ann: dict[str, Any],
    image: dict[str, Any],
    target_size: tuple[int, int],
) -> list[float]:
    bbox = ann.get("bbox")
    if not bbox or len(bbox) != 4:
        raise PseudoLabelDiagnosticsError(f"annotation {ann.get('id')} missing COCO bbox")
    x, y, w, h = [float(v) for v in bbox]
    image_w = float(image.get("width", target_size[1]))
    image_h = float(image.get("height", target_size[0]))
    if image_w <= 0.0 or image_h <= 0.0:
        raise PseudoLabelDiagnosticsError(f"image {image.get('id')} has invalid size {image_w}x{image_h}")
    scale_x = float(target_size[1]) / image_w
    scale_y = float(target_size[0]) / image_h
    return [x * scale_x, y * scale_y, (x + w) * scale_x, (y + h) * scale_y]


def _bbox_iou(box_a: list[float], box_b: list[float]) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)


def _scores_list(result: dict[str, Any]) -> list[float]:
    scores = result.get("scores")
    if not torch.is_tensor(scores) or scores.numel() == 0:
        return []
    values = [float(v) for v in scores.detach().float().cpu().tolist()]
    for value in values:
        if not math.isfinite(value):
            raise PseudoLabelDiagnosticsError(f"non-finite pseudo-label score: {value}")
    return values


def _empty_bucket_summary() -> dict[str, Any]:
    return {
        "images": 0,
        "gt_count": 0,
        "candidate_count": 0,
        "kept_count": 0,
        "keep_rate": None,
        "candidate_mean_quality": None,
        "kept_mean_quality": None,
        "candidate_gt_coverage_iou50": None,
        "kept_gt_coverage_iou50": None,
        "candidate_gt_coverage_iou75": None,
        "kept_gt_coverage_iou75": None,
        "candidate_gt_best_iou_mean": None,
        "kept_gt_best_iou_mean": None,
    }


def _finalize_bucket(bucket: dict[str, Any]) -> dict[str, Any]:
    candidates = int(bucket["candidate_count"])
    kept = int(bucket["kept_count"])
    gt_count = int(bucket["gt_count"])
    bucket["keep_rate"] = float(kept / candidates) if candidates else None
    bucket["candidate_mean_quality"] = _finite_or_none(_mean_float(bucket.pop("_candidate_scores")))
    bucket["kept_mean_quality"] = _finite_or_none(_mean_float(bucket.pop("_kept_scores")))
    for prefix in ("candidate", "kept"):
        bucket[f"{prefix}_gt_coverage_iou50"] = float(bucket[f"{prefix}_gt_iou50"] / gt_count) if gt_count else None
        bucket[f"{prefix}_gt_coverage_iou75"] = float(bucket[f"{prefix}_gt_iou75"] / gt_count) if gt_count else None
        bucket[f"{prefix}_gt_best_iou_mean"] = _finite_or_none(_mean_float(bucket.pop(f"_{prefix}_best_ious")))
        bucket.pop(f"{prefix}_gt_iou50")
        bucket.pop(f"{prefix}_gt_iou75")
    return bucket


def summarize_bucket_diagnostics(
    per_image: list[dict[str, Any]],
    *,
    cfg: Any,
    r78_stats_path: str | Path | None,
) -> dict[str, Any]:
    r78_by_image = _load_r78_stats(r78_stats_path)
    ann_path = _resolve_target_ann_path(cfg)
    gt_by_image, bottom20_threshold = _load_gt_buckets(ann_path)
    bucket_names = ("normal", "dense", "dense_tiny", "tiny_area_le_256", "bottom20_area")
    buckets: dict[str, dict[str, Any]] = {}
    for name in bucket_names:
        bucket = _empty_bucket_summary()
        bucket.update(
            {
                "_candidate_scores": [],
                "_kept_scores": [],
                "_candidate_best_ious": [],
                "_kept_best_ious": [],
                "candidate_gt_iou50": 0,
                "candidate_gt_iou75": 0,
                "kept_gt_iou50": 0,
                "kept_gt_iou75": 0,
            }
        )
        buckets[name] = bucket

    sampled_image_ids: list[int] = []
    sampled_image_files: list[str] = []
    for record in per_image:
        image_id = int(record["image_id"])
        if image_id not in r78_by_image:
            raise PseudoLabelDiagnosticsError(f"sampled image_id={image_id} missing from R78/R52 bucket stats")
        if image_id not in gt_by_image:
            raise PseudoLabelDiagnosticsError(f"sampled image_id={image_id} missing from target annotation")
        sampled_image_ids.append(image_id)
        sampled_image_files.append(str(r78_by_image[image_id].get("file_name", "")))

        candidate_boxes = record["candidate_boxes"]
        kept_boxes = record["kept_boxes"]
        candidate_scores = record["candidate_scores"]
        kept_scores = record["kept_scores"]
        gt_all = gt_by_image[image_id]["all"]
        gt_image = gt_by_image[image_id]["image"]
        target_size = tuple(record["mask_size"])

        image_bucket_name = str(r78_by_image[image_id]["r78_bucket"])
        image_bucket = buckets[image_bucket_name]
        image_bucket["images"] += 1
        image_bucket["gt_count"] += len(gt_all)
        image_bucket["candidate_count"] += len(candidate_boxes)
        image_bucket["kept_count"] += len(kept_boxes)
        image_bucket["_candidate_scores"].extend(candidate_scores)
        image_bucket["_kept_scores"].extend(kept_scores)
        _add_gt_coverage(image_bucket, gt_all, gt_image, target_size, candidate_boxes, kept_boxes)

        for gt_bucket_name in ("tiny_area_le_256", "bottom20_area"):
            gt_subset = gt_by_image[image_id][gt_bucket_name]
            if not gt_subset:
                continue
            gt_bucket = buckets[gt_bucket_name]
            gt_bucket["images"] += 1
            gt_bucket["gt_count"] += len(gt_subset)
            candidate_matches = _matched_prediction_indices(
                candidate_boxes,
                gt_subset,
                gt_image,
                target_size,
                min_iou=0.5,
            )
            kept_matches = _matched_prediction_indices(
                kept_boxes,
                gt_subset,
                gt_image,
                target_size,
                min_iou=0.5,
            )
            gt_bucket["candidate_count"] += len(candidate_matches)
            gt_bucket["kept_count"] += len(kept_matches)
            gt_bucket["_candidate_scores"].extend(candidate_scores[idx] for idx in candidate_matches)
            gt_bucket["_kept_scores"].extend(kept_scores[idx] for idx in kept_matches)
            _add_gt_coverage(gt_bucket, gt_subset, gt_image, target_size, candidate_boxes, kept_boxes)

    if not sampled_image_ids:
        raise PseudoLabelDiagnosticsError("bucket diagnostics saw zero sampled images")
    return {
        "sampled_image_ids": sampled_image_ids,
        "sampled_image_files": sampled_image_files,
        "sampled_images": len(sampled_image_ids),
        "target_unlabeled_ann": str(ann_path),
        "r78_stats": str(_resolve_project_path(r78_stats_path)),
        "bottom20_area_threshold": bottom20_threshold,
        "bucket_semantics": {
            "normal_dense_dense_tiny": "image-level R78/R52 prediction-only buckets",
            "tiny_area_le_256": "GT-area proxy; candidate/kept counts are predictions with bbox IoU >= 0.50 to a bucket GT object",
            "bottom20_area": "GT-area proxy; candidate/kept counts are predictions with bbox IoU >= 0.50 to a bottom-20% GT object",
            "coverage": "bbox IoU proxy against target_unlabeled GT; training still receives no GT labels",
        },
        "buckets": {name: _finalize_bucket(bucket) for name, bucket in buckets.items()},
    }


def _matched_prediction_indices(
    pred_boxes: list[list[float]],
    gt_anns: list[dict[str, Any]],
    image: dict[str, Any],
    target_size: tuple[int, int],
    *,
    min_iou: float,
) -> set[int]:
    gt_boxes = [_ann_box(ann, image, target_size) for ann in gt_anns]
    matched: set[int] = set()
    for pred_idx, pred_box in enumerate(pred_boxes):
        if any(_bbox_iou(pred_box, gt_box) >= min_iou for gt_box in gt_boxes):
            matched.add(pred_idx)
    return matched


def _add_gt_coverage(
    bucket: dict[str, Any],
    gt_anns: list[dict[str, Any]],
    image: dict[str, Any],
    target_size: tuple[int, int],
    candidate_boxes: list[list[float]],
    kept_boxes: list[list[float]],
) -> None:
    for ann in gt_anns:
        gt_box = _ann_box(ann, image, target_size)
        candidate_best = max((_bbox_iou(gt_box, box) for box in candidate_boxes), default=0.0)
        kept_best = max((_bbox_iou(gt_box, box) for box in kept_boxes), default=0.0)
        bucket["_candidate_best_ious"].append(candidate_best)
        bucket["_kept_best_ious"].append(kept_best)
        if candidate_best >= 0.5:
            bucket["candidate_gt_iou50"] += 1
        if candidate_best >= 0.75:
            bucket["candidate_gt_iou75"] += 1
        if kept_best >= 0.5:
            bucket["kept_gt_iou50"] += 1
        if kept_best >= 0.75:
            bucket["kept_gt_iou75"] += 1


def summarize_threshold_sweep(
    scored_results: list[dict[str, Any]],
    *,
    thresholds: list[float] | tuple[float, ...],
) -> list[dict[str, Any]]:
    """Return keep-rate diagnostics for each threshold without applying the gate."""
    predictions = sum(_tensor_count(result) for result in scored_results)
    images = len(scored_results)
    sweep = []
    for threshold in thresholds:
        kept_per_image = []
        for result in scored_results:
            scores = result.get("scores")
            if torch.is_tensor(scores) and scores.numel() > 0:
                kept_per_image.append(int((scores >= float(threshold)).sum().item()))
            else:
                kept_per_image.append(0)

        kept = sum(kept_per_image)
        empty_images = sum(1 for count in kept_per_image if count == 0)
        sweep.append(
            {
                "threshold": float(threshold),
                "kept": kept,
                "keep_rate": kept / predictions if predictions else 0.0,
                "empty_images": empty_images,
                "empty_ratio": empty_images / images if images else 0.0,
                "kept_per_image_mean": kept / images if images else 0.0,
                "kept_per_image": kept_per_image,
                "kept_per_image_min": min(kept_per_image) if kept_per_image else 0,
                "kept_per_image_max": max(kept_per_image) if kept_per_image else 0,
                "zero_image_count": empty_images,
            }
        )
    return sweep


def summarize_pseudo_label_scores(
    scored_results: list[dict[str, Any]],
    filtered_results: list[dict[str, Any]],
    *,
    threshold: float,
    threshold_source: str,
    threshold_config: dict[str, Any] | None = None,
    min_keep_rate: float = DEFAULT_MIN_KEEP_RATE,
    max_empty_ratio: float = DEFAULT_MAX_EMPTY_RATIO,
) -> dict[str, Any]:
    """Return keep-rate diagnostics and fail on empty pseudo-label gates."""
    predictions = sum(_tensor_count(result) for result in scored_results)
    kept_per_image = [_tensor_count(result) for result in filtered_results]
    kept = sum(kept_per_image)
    keep_rate = kept / predictions if predictions else 0.0

    images = len(filtered_results)
    empty_images = sum(1 for count in kept_per_image if count == 0)
    empty_ratio = empty_images / images if images else 0.0
    summary = {
        "images": images,
        "predictions": predictions,
        "kept": kept,
        "keep_rate": keep_rate,
        "kept_per_image": kept_per_image,
        "empty_images": empty_images,
        "empty_ratio": empty_ratio,
        "score_distribution": _score_distribution(_scores_to_cpu(scored_results)),
        "threshold": {
            "value": float(threshold),
            "source": threshold_source,
            "config": threshold_config or {},
        },
        "gate": {
            "min_keep_rate": float(min_keep_rate),
            "max_empty_ratio": float(max_empty_ratio),
        },
    }
    if predictions == 0:
        raise PseudoLabelDiagnosticsError(
            "predictions=0 after PseudoLabelScorer.score; Stage C would train with no pseudo-label candidates.",
            summary=summary,
        )
    if keep_rate == 0.0:
        raise PseudoLabelDiagnosticsError(
            f"keep_rate=0 at threshold={threshold}; Stage C would keep zero pseudo-labels.",
            summary=summary,
        )
    if keep_rate < min_keep_rate:
        raise PseudoLabelDiagnosticsError(
            f"keep_rate={keep_rate:.6f} below min_keep_rate={min_keep_rate:.6f} at threshold={threshold}.",
            summary=summary,
        )
    if empty_ratio > max_empty_ratio:
        raise PseudoLabelDiagnosticsError(
            f"empty_ratio={empty_ratio:.6f} above max_empty_ratio={max_empty_ratio:.6f} at threshold={threshold}.",
            summary=summary,
        )
    return summary


def summarize_pseudo_label_diagnostics(
    scored_results: list[dict[str, Any]],
    filtered_results: list[dict[str, Any]],
    *,
    threshold: float,
    threshold_source: str,
    threshold_config: dict[str, Any] | None = None,
    threshold_sweep: list[float] | tuple[float, ...] = DEFAULT_THRESHOLD_SWEEP,
    min_keep_rate: float = DEFAULT_MIN_KEEP_RATE,
    max_empty_ratio: float = DEFAULT_MAX_EMPTY_RATIO,
) -> dict[str, Any]:
    """Return gate diagnostics and attach threshold sweep on pass or fail."""
    try:
        summary = summarize_pseudo_label_scores(
            scored_results,
            filtered_results,
            threshold=threshold,
            threshold_source=threshold_source,
            threshold_config=threshold_config,
            min_keep_rate=min_keep_rate,
            max_empty_ratio=max_empty_ratio,
        )
    except PseudoLabelDiagnosticsError as exc:
        if exc.summary is not None:
            exc.summary["threshold_sweep"] = summarize_threshold_sweep(
                scored_results,
                thresholds=threshold_sweep,
            )
        raise
    summary["threshold_sweep"] = summarize_threshold_sweep(scored_results, thresholds=threshold_sweep)
    return summary


def _resolve_threshold(
    cfg: Any,
    *,
    epoch: int,
    threshold_override: float | None,
) -> tuple[float, str, dict[str, Any]]:
    pl_cfg = cfg.vc_suda.pseudo_label
    threshold_config = {
        "quality_threshold": float(_cfg_get(pl_cfg, "quality_threshold", 0.5)),
        "use_curriculum": bool(_cfg_get(pl_cfg, "use_curriculum", True)),
        "epoch": int(epoch),
    }
    if threshold_override is not None:
        return (
            float(threshold_override),
            "--threshold",
            {**threshold_config, "override": float(threshold_override)},
        )

    if bool(_cfg_get(pl_cfg, "use_curriculum", True)):
        from magformer.models.common.curriculum import CurriculumScheduler

        cur_cfg = cfg.vc_suda.curriculum
        scheduler = CurriculumScheduler(
            start_threshold=float(_cfg_get(cur_cfg, "start_threshold", 0.7)),
            end_threshold=float(_cfg_get(cur_cfg, "end_threshold", 0.3)),
            warmup_epochs=int(_cfg_get(cur_cfg, "warmup_epochs", 15)),
        )
        value = float(scheduler.get_threshold(epoch))
        threshold_config.update(
            {
                "start_threshold": float(_cfg_get(cur_cfg, "start_threshold", 0.7)),
                "end_threshold": float(_cfg_get(cur_cfg, "end_threshold", 0.3)),
                "warmup_epochs": int(_cfg_get(cur_cfg, "warmup_epochs", 15)),
            }
        )
        return value, "vc_suda.curriculum", threshold_config

    return (
        float(_cfg_get(pl_cfg, "quality_threshold", 0.5)),
        "vc_suda.pseudo_label.quality_threshold",
        threshold_config,
    )


def _build_scorer(cfg: Any) -> PseudoLabelScorer:
    pl_cfg = cfg.vc_suda.pseudo_label
    return PseudoLabelScorer(max_instances=int(_cfg_get(pl_cfg, "max_instances", 100)))


def _slice_tensor(value: Any, count: int) -> Any:
    if torch.is_tensor(value) and value.ndim > 0:
        return value[:count]
    return value


def _target_weak_batch(batch: dict[str, Any], remaining: int) -> dict[str, Any]:
    images = batch.get("target_weak_images")
    depths = batch.get("target_weak_depths")
    if images is None or depths is None:
        raise PseudoLabelDiagnosticsError("batch is missing target_weak_images or target_weak_depths")
    count = min(int(images.shape[0]), remaining)
    return {
        "images": _slice_tensor(images, count),
        "depths": _slice_tensor(depths, count),
        "padding_masks": _slice_tensor(batch.get("target_weak_padding_masks"), count),
        "depth_noise_masks": _slice_tensor(batch.get("target_weak_noise_masks"), count),
        "image_ids": _slice_tensor(batch.get("target_weak_image_ids"), count),
    }


def _teacher_forward(
    model: torch.nn.Module,
    batch: dict[str, Any],
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    images = batch["images"].to(device)
    depths = batch["depths"].to(device)
    padding_masks = batch["padding_masks"]
    depth_noise_masks = batch["depth_noise_masks"]
    if torch.is_tensor(padding_masks):
        padding_masks = padding_masks.to(device)
    if torch.is_tensor(depth_noise_masks):
        depth_noise_masks = depth_noise_masks.to(device)

    if hasattr(model, "forward_inference_decoder_outputs"):
        return model.forward_inference_decoder_outputs(
            images,
            depths,
            padding_masks=padding_masks,
            depth_noise_masks=depth_noise_masks,
        )
    return model(
        images,
        depths,
        targets=None,
        padding_masks=padding_masks,
        depth_noise_masks=depth_noise_masks,
        return_features=True,
    )


def run_diagnostics(
    config_path: str | Path,
    *,
    weights: str | Path | None = None,
    max_images: int = 2,
    batch_size: int = 1,
    num_workers: int = 0,
    device_name: str = "cpu",
    epoch: int = 0,
    threshold_override: float | None = None,
    threshold_sweep: list[float] | tuple[float, ...] = DEFAULT_THRESHOLD_SWEEP,
    min_keep_rate: float = DEFAULT_MIN_KEEP_RATE,
    max_empty_ratio: float = DEFAULT_MAX_EMPTY_RATIO,
    r78_stats: str | Path | None = None,
    bucket_output_json: str | Path | None = None,
) -> dict[str, Any]:
    if max_images <= 0:
        raise PseudoLabelDiagnosticsError("--max-images must be positive")
    if batch_size <= 0:
        raise PseudoLabelDiagnosticsError("--batch-size must be positive")

    os.chdir(PROJECT_ROOT)
    cfg_path = _resolve_project_path(config_path)
    if cfg_path is None:
        raise PseudoLabelDiagnosticsError("config path is required")
    cfg = load_config(str(cfg_path))
    set_seed(int(cfg.runtime.seed))

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise PseudoLabelDiagnosticsError(f"CUDA device requested but unavailable: {device_name}")

    effective_weights = _resolve_project_path(weights or getattr(cfg.model, "finetune_weights", None))
    if effective_weights is None:
        raise PseudoLabelDiagnosticsError("weights are required via --weights or model.finetune_weights")
    if not effective_weights.exists():
        raise PseudoLabelDiagnosticsError(f"weights not found: {effective_weights}")

    train_dataset, val_dataset = train_tool.build_datasets(cfg)
    if getattr(train_dataset, "target_unlabeled", None) is None:
        raise PseudoLabelDiagnosticsError("Stage C dataset did not build target_unlabeled")
    train_loader, _ = train_tool.build_data_loaders(
        cfg,
        train_dataset,
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        is_distributed=False,
    )

    model = train_tool.build_model(cfg, device)
    train_tool.load_finetune_weights(model, str(effective_weights), strict=False)
    model.eval()

    scorer = _build_scorer(cfg)
    threshold, threshold_source, threshold_config = _resolve_threshold(
        cfg,
        epoch=epoch,
        threshold_override=threshold_override,
    )

    scored_all: list[dict[str, Any]] = []
    filtered_all: list[dict[str, Any]] = []
    per_image: list[dict[str, Any]] = []
    seen = 0
    with torch.inference_mode():
        for raw_batch in train_loader:
            remaining = max_images - seen
            if remaining <= 0:
                break
            batch = _target_weak_batch(raw_batch, remaining)
            teacher_outputs = _teacher_forward(model, batch, device=device)
            depths = batch["depths"].to(device)
            scored = scorer.score(teacher_outputs, depths)
            filtered = scorer.filter_by_threshold(scored, threshold)
            scored_all.extend(scored)
            filtered_all.extend(filtered)
            image_ids = batch.get("image_ids")
            if not torch.is_tensor(image_ids):
                raise PseudoLabelDiagnosticsError("batch is missing target_weak_image_ids")
            for image_id, scored_result, filtered_result in zip(
                image_ids.detach().cpu().tolist(),
                scored,
                filtered,
                strict=True,
            ):
                per_image.append(
                    {
                        "image_id": int(image_id),
                        "candidate_count": _tensor_count(scored_result),
                        "kept_count": _tensor_count(filtered_result),
                        "candidate_scores": _scores_list(scored_result),
                        "kept_scores": _scores_list(filtered_result),
                        "candidate_boxes": _boxes_from_masks(scored_result),
                        "kept_boxes": _boxes_from_masks(filtered_result),
                        "mask_size": tuple(int(v) for v in scored_result["masks"].shape[-2:]),
                    }
                )
            seen += int(batch["images"].shape[0])

    if seen == 0:
        raise PseudoLabelDiagnosticsError("target_unlabeled loader produced zero images")

    summary = summarize_pseudo_label_diagnostics(
        scored_all,
        filtered_all,
        threshold=threshold,
        threshold_source=threshold_source,
        threshold_config=threshold_config,
        threshold_sweep=threshold_sweep,
        min_keep_rate=min_keep_rate,
        max_empty_ratio=max_empty_ratio,
    )
    summary.update(
        {
            "config": str(cfg_path),
            "weights": str(effective_weights),
            "device": str(device),
            "max_images": int(max_images),
            "batch_size": int(batch_size),
            "target_unlabeled_ann": str(cfg.vc_suda.target_unlabeled_ann),
        }
    )
    if bucket_output_json:
        bucket_summary = summarize_bucket_diagnostics(per_image, cfg=cfg, r78_stats_path=r78_stats)
        bucket_path = _resolve_project_path(bucket_output_json)
        assert bucket_path is not None
        bucket_path.parent.mkdir(parents=True, exist_ok=True)
        bucket_path.write_text(json.dumps(bucket_summary, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        summary["bucket_diagnostics"] = {
            "output_json": str(bucket_path),
            "sampled_images": int(bucket_summary["sampled_images"]),
            "sampled_image_ids": bucket_summary["sampled_image_ids"],
        }
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/vc_suda_stage_c_1024_teacher8499.yaml")
    parser.add_argument("--weights", "--finetune-weights", dest="weights", default=None)
    parser.add_argument("--max-images", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epoch", type=int, default=0, help="Epoch used only when curriculum is enabled.")
    parser.add_argument("--threshold", type=float, default=None, help="Override the config threshold for probing.")
    parser.add_argument(
        "--min-keep-rate",
        type=float,
        default=DEFAULT_MIN_KEEP_RATE,
        help="Fail when the effective threshold keeps less than this fraction of scored predictions.",
    )
    parser.add_argument(
        "--max-empty-ratio",
        type=float,
        default=DEFAULT_MAX_EMPTY_RATIO,
        help="Fail when more than this fraction of sampled images keep zero pseudo-labels.",
    )
    parser.add_argument(
        "--threshold-sweep",
        type=float,
        nargs="+",
        default=list(DEFAULT_THRESHOLD_SWEEP),
        help="Thresholds to scan and report without changing the fail-fast gate.",
    )
    parser.add_argument("--output-json", default=None, help="Optional path to write the diagnostic JSON.")
    parser.add_argument(
        "--r78-stats",
        default=None,
        help="R78/R52 target_unlabeled bucket stats JSON, required with --bucket-output-json.",
    )
    parser.add_argument(
        "--bucket-output-json",
        default=None,
        help="Optional path to write per-bucket pseudo-label scorer diagnostics.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        summary = run_diagnostics(
            args.config,
            weights=args.weights,
            max_images=args.max_images,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device_name=args.device,
            epoch=args.epoch,
            threshold_override=args.threshold,
            threshold_sweep=args.threshold_sweep,
            min_keep_rate=args.min_keep_rate,
            max_empty_ratio=args.max_empty_ratio,
            r78_stats=args.r78_stats,
            bucket_output_json=args.bucket_output_json,
        )
    except PseudoLabelDiagnosticsError as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        if exc.summary is not None:
            payload = json.dumps(exc.summary, sort_keys=True)
            print(payload)
            if args.output_json:
                output_path = _resolve_project_path(args.output_json)
                assert output_path is not None
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(payload + "\n", encoding="utf-8")
        return 1

    payload = json.dumps(summary, sort_keys=True)
    print("PASS pseudo_label_diagnostics")
    print(payload)
    if args.output_json:
        output_path = _resolve_project_path(args.output_json)
        assert output_path is not None
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
