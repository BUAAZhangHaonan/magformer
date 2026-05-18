#!/usr/bin/env python3
"""Diagnose R120 MagFormer RGB-D error atlas from existing COCO predictions."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from pycocotools import mask as mask_utils


DEFAULT_OUT_DIR = "output/diagnostics/r120_error_atlas_20260518"
DEFAULT_GT_VAL28 = "magformer_datasets/pseudo_real_512/annotations/instances_val.json"
DEFAULT_GT_TARGET200 = "magformer_datasets/pseudo_real_512/annotations/instances_target_unlabeled.json"
DEFAULT_GT_R114_REMAINING75 = (
    "magformer_datasets/pseudo_real_512/annotations/instances_target_unlabeled_r114_balanced_minus125.json"
)
DEFAULT_R114_VAL28_PRED = (
    "output/diagnostics/"
    "r114_magformer_r113warm_target150_iter2000_val28_1024_backmap_topk200_20260518/"
    "coco_instances_results.json"
)
DEFAULT_R114_REMAINING75_PRED = (
    "output/diagnostics/"
    "r114_magformer_r113warm_target150_iter2000_remaining75_1024_backmap_topk200_20260518/"
    "coco_instances_results.json"
)
DEFAULT_R114_FULL200_PRED = (
    "output/diagnostics/"
    "r114_magformer_r113warm_target150_iter2000_full200_reference_1024_backmap_topk200_20260518/"
    "coco_instances_results.json"
)
DEFAULT_R115_TRAIN200_PRED = (
    "output/diagnostics/"
    "r115b_magformer_r114warm_fulltarget200_oracle_iter2000_train200_1024_backmap_topk200_20260518/"
    "coco_instances_results.json"
)
DEFAULT_R115_VAL28_PRED = (
    "output/diagnostics/"
    "r115b_magformer_r114warm_fulltarget200_oracle_iter2000_val28_1024_backmap_topk200_20260518/"
    "coco_instances_results.json"
)

CSV_NAN = ""


class DiagnosisError(RuntimeError):
    """Raised when required inputs are missing or malformed."""


@dataclass(frozen=True)
class RunSpec:
    name: str
    role: str
    gt_path: Path
    pred_path: Path


@dataclass
class RunAnalysis:
    spec: RunSpec
    gt_rows: list[dict[str, Any]]
    pred_rows: list[dict[str, Any]]
    bucket_rows: list[dict[str, Any]]
    summary: dict[str, Any]


def _load_json(path: Path) -> Any:
    if not path.exists():
        raise DiagnosisError(f"missing input file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _require_coco(payload: Any, path: Path) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise DiagnosisError(f"{path} must be a COCO annotation object")
    for key in ("images", "annotations", "categories"):
        if not isinstance(payload.get(key), list):
            raise DiagnosisError(f"{path} is missing COCO field: {key}")
    return payload


def _require_predictions(payload: Any, path: Path) -> list[dict[str, Any]]:
    if not isinstance(payload, list):
        raise DiagnosisError(f"{path} must be a COCO result list")
    for idx, item in enumerate(payload):
        if not isinstance(item, dict):
            raise DiagnosisError(f"{path}[{idx}] must be an object")
        for key in ("image_id", "category_id", "score", "bbox", "segmentation"):
            if key not in item:
                raise DiagnosisError(f"{path}[{idx}] is missing {key}")
    return payload


def _mask_rle(segmentation: Any, height: int, width: int) -> dict[str, Any]:
    if isinstance(segmentation, dict):
        rle = dict(segmentation)
        if isinstance(rle.get("counts"), str):
            rle["counts"] = rle["counts"].encode("ascii")
        return rle
    if isinstance(segmentation, list):
        return mask_utils.merge(mask_utils.frPyObjects(segmentation, height, width))
    raise DiagnosisError(f"unsupported segmentation type: {type(segmentation).__name__}")


def _decode_mask(rle: dict[str, Any]) -> np.ndarray:
    decoded = mask_utils.decode(rle)
    if decoded.ndim == 3:
        decoded = decoded[:, :, 0]
    return decoded.astype(bool)


def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)


def _boundary(mask: np.ndarray) -> np.ndarray:
    item = np.asarray(mask, dtype=bool)
    if item.size == 0 or not item.any():
        return np.zeros_like(item, dtype=bool)
    padded = np.pad(item, 1, mode="constant", constant_values=False)
    up = padded[:-2, 1:-1]
    down = padded[2:, 1:-1]
    left = padded[1:-1, :-2]
    right = padded[1:-1, 2:]
    return item & ~(up & down & left & right)


def _dilate_1px(mask: np.ndarray) -> np.ndarray:
    item = np.asarray(mask, dtype=bool)
    padded = np.pad(item, 1, mode="constant", constant_values=False)
    out = np.zeros_like(item, dtype=bool)
    for dy in range(3):
        for dx in range(3):
            out |= padded[dy : dy + item.shape[0], dx : dx + item.shape[1]]
    return out


def _boundary_f_score(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    pred_boundary = _boundary(pred_mask)
    gt_boundary = _boundary(gt_mask)
    pred_count = int(pred_boundary.sum())
    gt_count = int(gt_boundary.sum())
    if pred_count == 0 and gt_count == 0:
        return 1.0
    if pred_count == 0 or gt_count == 0:
        return 0.0
    pred_hit = np.logical_and(pred_boundary, _dilate_1px(gt_boundary)).sum()
    gt_hit = np.logical_and(gt_boundary, _dilate_1px(pred_boundary)).sum()
    precision = float(pred_hit / pred_count)
    recall = float(gt_hit / gt_count)
    if precision + recall == 0.0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def _area_bucket(area: float) -> str:
    if area <= 256:
        return "tiny_area_le_256"
    if area <= 1024:
        return "small_257_1024"
    if area <= 4096:
        return "medium_1025_4096"
    return "large_gt4096"


def _density_bucket(count: int) -> str:
    if count <= 25:
        return "low25"
    if count <= 50:
        return "mid50"
    return "high100"


def _quantile_bucket(area: float, q20: float, q80: float) -> str:
    if area <= q20:
        return "bottom20_area"
    if area <= q80:
        return "mid60_area"
    return "top20_area"


def _finite_mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return float("nan")
    return float(np.mean(finite))


def _ratio(num: int | float, den: int | float) -> float:
    if den == 0:
        return float("nan")
    return float(num / den)


def _fmt_float(value: Any, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "n/a"
    return f"{number:.{digits}f}"


def _csv_value(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return CSV_NAN
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in fieldnames})


def _prepare_gt(coco: dict[str, Any]) -> tuple[dict[int, dict[str, Any]], dict[tuple[int, int], list[dict[str, Any]]]]:
    images = {int(image["id"]): image for image in coco["images"]}
    density = Counter(int(ann["image_id"]) for ann in coco["annotations"] if int(ann.get("iscrowd", 0)) == 0)
    anns_by_key: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    areas = [float(ann.get("area", 0.0)) for ann in coco["annotations"] if int(ann.get("iscrowd", 0)) == 0]
    q20 = float(np.quantile(areas, 0.20)) if areas else float("nan")
    q80 = float(np.quantile(areas, 0.80)) if areas else float("nan")
    for ann in coco["annotations"]:
        if int(ann.get("iscrowd", 0)) != 0:
            continue
        image_id = int(ann["image_id"])
        category_id = int(ann["category_id"])
        if image_id not in images:
            raise DiagnosisError(f"annotation references missing image_id={image_id}")
        image = images[image_id]
        area = float(ann.get("area", 0.0))
        item = dict(ann)
        item["_image_id"] = image_id
        item["_category_id"] = category_id
        item["_height"] = int(image["height"])
        item["_width"] = int(image["width"])
        item["_file_name"] = image.get("file_name", "")
        item["_density_count"] = int(density[image_id])
        item["_density_bucket"] = _density_bucket(int(density[image_id]))
        item["_area"] = area
        item["_area_bucket"] = _area_bucket(area)
        item["_area_quantile_bucket"] = _quantile_bucket(area, q20, q80)
        item["_rle"] = _mask_rle(ann["segmentation"], int(image["height"]), int(image["width"]))
        anns_by_key[(image_id, category_id)].append(item)
    for key in anns_by_key:
        anns_by_key[key].sort(key=lambda ann: int(ann["id"]))
    return images, anns_by_key


def _prepare_predictions(
    predictions: list[dict[str, Any]],
    images: dict[int, dict[str, Any]],
) -> dict[tuple[int, int], list[dict[str, Any]]]:
    preds_by_key: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for idx, pred in enumerate(predictions):
        image_id = int(pred["image_id"])
        category_id = int(pred["category_id"])
        if image_id not in images:
            raise DiagnosisError(f"prediction references missing image_id={image_id}")
        image = images[image_id]
        item = dict(pred)
        item["_pred_index"] = idx
        item["_image_id"] = image_id
        item["_category_id"] = category_id
        item["_height"] = int(image["height"])
        item["_width"] = int(image["width"])
        item["_file_name"] = image.get("file_name", "")
        item["_rle"] = _mask_rle(pred["segmentation"], int(image["height"]), int(image["width"]))
        preds_by_key[(image_id, category_id)].append(item)
    for key in preds_by_key:
        preds_by_key[key].sort(key=lambda pred: (-float(pred["score"]), int(pred["_pred_index"])))
    return preds_by_key


def _greedy_match(ious: np.ndarray, threshold: float) -> tuple[list[int | None], list[int | None], list[bool]]:
    pred_count, gt_count = ious.shape
    pred_to_gt: list[int | None] = [None] * pred_count
    gt_to_pred: list[int | None] = [None] * gt_count
    duplicate_fp = [False] * pred_count
    matched_gt: set[int] = set()
    for pred_idx in range(pred_count):
        best_gt: int | None = None
        best_iou = threshold
        for gt_idx in range(gt_count):
            if gt_idx in matched_gt:
                continue
            iou = float(ious[pred_idx, gt_idx])
            if iou >= best_iou:
                best_iou = iou
                best_gt = gt_idx
        if best_gt is not None:
            matched_gt.add(best_gt)
            pred_to_gt[pred_idx] = best_gt
            gt_to_pred[best_gt] = pred_idx
            continue
        if gt_count:
            best_any = int(np.argmax(ious[pred_idx]))
            duplicate_fp[pred_idx] = float(ious[pred_idx, best_any]) >= threshold and best_any in matched_gt
    return pred_to_gt, gt_to_pred, duplicate_fp


def _best_boundary(
    pred_rles: list[dict[str, Any]],
    gt_rles: list[dict[str, Any]],
    pred_idx: int | None,
    gt_idx: int | None,
) -> float:
    if pred_idx is None or gt_idx is None or pred_idx < 0 or gt_idx < 0:
        return float("nan")
    return _boundary_f_score(_decode_mask(pred_rles[pred_idx]), _decode_mask(gt_rles[gt_idx]))


def analyze_run(spec: RunSpec) -> RunAnalysis:
    coco = _require_coco(_load_json(spec.gt_path), spec.gt_path)
    predictions = _require_predictions(_load_json(spec.pred_path), spec.pred_path)
    images, anns_by_key = _prepare_gt(coco)
    preds_by_key = _prepare_predictions(predictions, images)

    gt_rows: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []
    keys = sorted(set(anns_by_key) | set(preds_by_key))
    image_density = Counter(int(ann["image_id"]) for ann in coco["annotations"] if int(ann.get("iscrowd", 0)) == 0)

    for key in keys:
        image_id, category_id = key
        anns = anns_by_key.get(key, [])
        preds = preds_by_key.get(key, [])
        gt_rles = [ann["_rle"] for ann in anns]
        pred_rles = [pred["_rle"] for pred in preds]
        iscrowd = [0] * len(anns)
        if gt_rles and pred_rles:
            ious = np.asarray(mask_utils.iou(pred_rles, gt_rles, iscrowd), dtype=np.float64)
        else:
            ious = np.zeros((len(preds), len(anns)), dtype=np.float64)

        p50_to_g, g50_to_p, dup50 = _greedy_match(ious, 0.50)
        p75_to_g, g75_to_p, dup75 = _greedy_match(ious, 0.75)

        gt_best_pred: list[int | None] = [None] * len(anns)
        gt_best_iou = [0.0] * len(anns)
        if len(preds):
            for gt_idx in range(len(anns)):
                pred_idx = int(np.argmax(ious[:, gt_idx]))
                gt_best_pred[gt_idx] = pred_idx
                gt_best_iou[gt_idx] = float(ious[pred_idx, gt_idx])

        pred_best_gt: list[int | None] = [None] * len(preds)
        pred_best_iou = [0.0] * len(preds)
        if len(anns):
            for pred_idx in range(len(preds)):
                gt_idx = int(np.argmax(ious[pred_idx, :]))
                pred_best_gt[pred_idx] = gt_idx
                pred_best_iou[pred_idx] = float(ious[pred_idx, gt_idx])

        for gt_idx, ann in enumerate(anns):
            best_pred_idx = gt_best_pred[gt_idx]
            match50 = g50_to_p[gt_idx]
            match75 = g75_to_p[gt_idx]
            gt_rows.append(
                {
                    "run": spec.name,
                    "role": spec.role,
                    "image_id": image_id,
                    "file_name": ann["_file_name"],
                    "category_id": category_id,
                    "gt_id": int(ann["id"]),
                    "gt_area": ann["_area"],
                    "area_bucket": ann["_area_bucket"],
                    "area_quantile_bucket": ann["_area_quantile_bucket"],
                    "image_gt_count": ann["_density_count"],
                    "density_bucket": ann["_density_bucket"],
                    "best_pred_index": preds[best_pred_idx]["_pred_index"] if best_pred_idx is not None else "",
                    "best_pred_score": float(preds[best_pred_idx]["score"]) if best_pred_idx is not None else float("nan"),
                    "best_iou": gt_best_iou[gt_idx],
                    "boundary_f_best": _best_boundary(pred_rles, gt_rles, best_pred_idx, gt_idx),
                    "tp50": int(match50 is not None),
                    "fn50": int(match50 is None),
                    "matched_pred_index_50": preds[match50]["_pred_index"] if match50 is not None else "",
                    "tp75": int(match75 is not None),
                    "fn75": int(match75 is None),
                    "matched_pred_index_75": preds[match75]["_pred_index"] if match75 is not None else "",
                }
            )

        for pred_idx, pred in enumerate(preds):
            best_gt_idx = pred_best_gt[pred_idx]
            gt_ann = anns[best_gt_idx] if best_gt_idx is not None else None
            density_count = int(image_density.get(image_id, 0))
            pred_rows.append(
                {
                    "run": spec.name,
                    "role": spec.role,
                    "image_id": image_id,
                    "file_name": pred["_file_name"],
                    "category_id": category_id,
                    "pred_index": int(pred["_pred_index"]),
                    "score": float(pred["score"]),
                    "bbox_x": float(pred["bbox"][0]),
                    "bbox_y": float(pred["bbox"][1]),
                    "bbox_w": float(pred["bbox"][2]),
                    "bbox_h": float(pred["bbox"][3]),
                    "image_gt_count": density_count,
                    "density_bucket": _density_bucket(density_count),
                    "best_gt_id": int(gt_ann["id"]) if gt_ann is not None else "",
                    "best_gt_area": gt_ann["_area"] if gt_ann is not None else float("nan"),
                    "best_gt_area_bucket": gt_ann["_area_bucket"] if gt_ann is not None else "no_gt",
                    "best_gt_area_quantile_bucket": gt_ann["_area_quantile_bucket"] if gt_ann is not None else "no_gt",
                    "best_gt_iou": pred_best_iou[pred_idx],
                    "boundary_f_best": _best_boundary(pred_rles, gt_rles, pred_idx, best_gt_idx),
                    "tp50": int(p50_to_g[pred_idx] is not None),
                    "fp50": int(p50_to_g[pred_idx] is None),
                    "duplicate_fp50": int(dup50[pred_idx]),
                    "matched_gt_id_50": int(anns[p50_to_g[pred_idx]]["id"]) if p50_to_g[pred_idx] is not None else "",
                    "tp75": int(p75_to_g[pred_idx] is not None),
                    "fp75": int(p75_to_g[pred_idx] is None),
                    "duplicate_fp75": int(dup75[pred_idx]),
                    "matched_gt_id_75": int(anns[p75_to_g[pred_idx]]["id"]) if p75_to_g[pred_idx] is not None else "",
                }
            )

    bucket_rows = _build_bucket_rows(spec, gt_rows, pred_rows)
    summary = _build_run_summary(spec, gt_rows, pred_rows, bucket_rows, len(images))
    return RunAnalysis(spec=spec, gt_rows=gt_rows, pred_rows=pred_rows, bucket_rows=bucket_rows, summary=summary)


def _build_bucket_rows(
    spec: RunSpec,
    gt_rows: list[dict[str, Any]],
    pred_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    out = []
    bucket_specs = [
        ("density_bucket", "density_bucket", "density_bucket"),
        ("area_bucket", "area_bucket", "best_gt_area_bucket"),
        ("area_quantile_bucket", "area_quantile_bucket", "best_gt_area_quantile_bucket"),
    ]
    for bucket_type, gt_key, pred_key in bucket_specs:
        names = sorted({str(row[gt_key]) for row in gt_rows} | {str(row[pred_key]) for row in pred_rows})
        for name in names:
            if name == "no_gt":
                continue
            gt_bucket = [row for row in gt_rows if row[gt_key] == name]
            pred_bucket = [row for row in pred_rows if row[pred_key] == name]
            tp50 = sum(int(row["tp50"]) for row in pred_bucket)
            tp75 = sum(int(row["tp75"]) for row in pred_bucket)
            fp50 = sum(int(row["fp50"]) for row in pred_bucket)
            fp75 = sum(int(row["fp75"]) for row in pred_bucket)
            gt_count = len(gt_bucket)
            pred_count = len(pred_bucket)
            out.append(
                {
                    "run": spec.name,
                    "role": spec.role,
                    "bucket_type": bucket_type,
                    "bucket": name,
                    "images": len({int(row["image_id"]) for row in gt_bucket}),
                    "gt": gt_count,
                    "pred": pred_count,
                    "tp50": tp50,
                    "fn50": sum(int(row["fn50"]) for row in gt_bucket),
                    "fp50": fp50,
                    "precision50": _ratio(tp50, tp50 + fp50),
                    "recall50": _ratio(sum(int(row["tp50"]) for row in gt_bucket), gt_count),
                    "duplicate_fp50": sum(int(row["duplicate_fp50"]) for row in pred_bucket),
                    "tp75": tp75,
                    "fn75": sum(int(row["fn75"]) for row in gt_bucket),
                    "fp75": fp75,
                    "precision75": _ratio(tp75, tp75 + fp75),
                    "recall75": _ratio(sum(int(row["tp75"]) for row in gt_bucket), gt_count),
                    "duplicate_fp75": sum(int(row["duplicate_fp75"]) for row in pred_bucket),
                    "best_iou_mean": _finite_mean([float(row["best_iou"]) for row in gt_bucket]),
                    "boundary_f_mean": _finite_mean([float(row["boundary_f_best"]) for row in gt_bucket]),
                }
            )
    return out


def _build_run_summary(
    spec: RunSpec,
    gt_rows: list[dict[str, Any]],
    pred_rows: list[dict[str, Any]],
    bucket_rows: list[dict[str, Any]],
    image_count: int,
) -> dict[str, Any]:
    gt_count = len(gt_rows)
    pred_count = len(pred_rows)
    tp50_gt = sum(int(row["tp50"]) for row in gt_rows)
    tp75_gt = sum(int(row["tp75"]) for row in gt_rows)
    tp50_pred = sum(int(row["tp50"]) for row in pred_rows)
    tp75_pred = sum(int(row["tp75"]) for row in pred_rows)
    fp50 = sum(int(row["fp50"]) for row in pred_rows)
    fp75 = sum(int(row["fp75"]) for row in pred_rows)
    return {
        "role": spec.role,
        "gt_path": str(spec.gt_path),
        "pred_path": str(spec.pred_path),
        "images": image_count,
        "gt": gt_count,
        "pred": pred_count,
        "tp50": tp50_gt,
        "fn50": gt_count - tp50_gt,
        "fp50": fp50,
        "precision50": _ratio(tp50_pred, tp50_pred + fp50),
        "recall50": _ratio(tp50_gt, gt_count),
        "duplicate_fp50": sum(int(row["duplicate_fp50"]) for row in pred_rows),
        "tp75": tp75_gt,
        "fn75": gt_count - tp75_gt,
        "fp75": fp75,
        "precision75": _ratio(tp75_pred, tp75_pred + fp75),
        "recall75": _ratio(tp75_gt, gt_count),
        "duplicate_fp75": sum(int(row["duplicate_fp75"]) for row in pred_rows),
        "best_iou_mean": _finite_mean([float(row["best_iou"]) for row in gt_rows]),
        "best_iou_p50": float(np.median([float(row["best_iou"]) for row in gt_rows])) if gt_rows else float("nan"),
        "boundary_f_mean": _finite_mean([float(row["boundary_f_best"]) for row in gt_rows]),
        "buckets": bucket_rows,
    }


def _quality_state(iou: float) -> str:
    if iou < 0.50:
        return "miss"
    if iou < 0.75:
        return "low"
    return "good"


def _delta_label(old_iou: float, new_iou: float) -> str:
    old_state = _quality_state(old_iou)
    new_state = _quality_state(new_iou)
    if old_state == "miss" and new_state != "miss":
        return "miss_to_hit"
    if old_state == "low" and new_state == "good":
        return "low_to_good"
    if old_state == "miss" and new_state == "miss":
        return "both_miss"
    if old_state == "good" and new_state != "good":
        return "good_to_worse"
    if old_state != "miss" and new_state == "miss":
        return "hit_to_miss"
    if old_state == "good" and new_state == "good":
        return "both_good"
    if old_state == "low" and new_state == "low":
        return "both_low"
    return f"{old_state}_to_{new_state}"


def build_delta_rows(old: RunAnalysis, new: RunAnalysis) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    old_by_gt = {int(row["gt_id"]): row for row in old.gt_rows}
    new_by_gt = {int(row["gt_id"]): row for row in new.gt_rows}
    if set(old_by_gt) != set(new_by_gt):
        raise DiagnosisError("delta runs do not share the same target200 GT ids")
    rows = []
    for gt_id in sorted(old_by_gt):
        left = old_by_gt[gt_id]
        right = new_by_gt[gt_id]
        old_iou = float(left["best_iou"])
        new_iou = float(right["best_iou"])
        rows.append(
            {
                "gt_id": gt_id,
                "image_id": int(left["image_id"]),
                "file_name": left["file_name"],
                "category_id": int(left["category_id"]),
                "gt_area": float(left["gt_area"]),
                "area_bucket": left["area_bucket"],
                "area_quantile_bucket": left["area_quantile_bucket"],
                "image_gt_count": int(left["image_gt_count"]),
                "density_bucket": left["density_bucket"],
                "r114_best_iou": old_iou,
                "r115_best_iou": new_iou,
                "delta_iou": new_iou - old_iou,
                "r114_state": _quality_state(old_iou),
                "r115_state": _quality_state(new_iou),
                "delta_label": _delta_label(old_iou, new_iou),
                "r114_tp50": int(left["tp50"]),
                "r115_tp50": int(right["tp50"]),
                "r114_tp75": int(left["tp75"]),
                "r115_tp75": int(right["tp75"]),
                "r114_boundary_f": float(left["boundary_f_best"]),
                "r115_boundary_f": float(right["boundary_f_best"]),
                "delta_boundary_f": float(right["boundary_f_best"]) - float(left["boundary_f_best"]),
            }
        )
    summary = _build_delta_summary(rows)
    return rows, summary


def _build_delta_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_label = Counter(str(row["delta_label"]) for row in rows)
    by_density: dict[str, dict[str, Any]] = {}
    by_area: dict[str, dict[str, Any]] = {}
    for key, bucket in (("density_bucket", by_density), ("area_bucket", by_area)):
        for name in sorted({str(row[key]) for row in rows}):
            chunk = [row for row in rows if row[key] == name]
            bucket[name] = {
                "gt": len(chunk),
                "r114_recall50": _ratio(sum(int(row["r114_tp50"]) for row in chunk), len(chunk)),
                "r115_recall50": _ratio(sum(int(row["r115_tp50"]) for row in chunk), len(chunk)),
                "delta_recall50": _ratio(sum(int(row["r115_tp50"]) - int(row["r114_tp50"]) for row in chunk), len(chunk)),
                "r114_recall75": _ratio(sum(int(row["r114_tp75"]) for row in chunk), len(chunk)),
                "r115_recall75": _ratio(sum(int(row["r115_tp75"]) for row in chunk), len(chunk)),
                "delta_recall75": _ratio(sum(int(row["r115_tp75"]) - int(row["r114_tp75"]) for row in chunk), len(chunk)),
                "mean_delta_iou": _finite_mean([float(row["delta_iou"]) for row in chunk]),
                "miss_to_hit": sum(1 for row in chunk if row["delta_label"] == "miss_to_hit"),
                "low_to_good": sum(1 for row in chunk if row["delta_label"] == "low_to_good"),
                "both_miss": sum(1 for row in chunk if row["delta_label"] == "both_miss"),
                "good_to_worse": sum(1 for row in chunk if row["delta_label"] == "good_to_worse"),
            }
    return {
        "gt": len(rows),
        "label_counts": dict(sorted(by_label.items())),
        "mean_delta_iou": _finite_mean([float(row["delta_iou"]) for row in rows]),
        "mean_delta_boundary_f": _finite_mean([float(row["delta_boundary_f"]) for row in rows]),
        "by_density_bucket": by_density,
        "by_area_bucket": by_area,
    }


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    run_rows = []
    for name, item in summary["runs"].items():
        run_rows.append(
            [
                name,
                item["role"],
                item["images"],
                item["gt"],
                item["pred"],
                _fmt_float(item["precision50"]),
                _fmt_float(item["recall50"]),
                _fmt_float(item["precision75"]),
                _fmt_float(item["recall75"]),
                _fmt_float(item["best_iou_p50"]),
                _fmt_float(item["boundary_f_mean"]),
            ]
        )
    lines = [
        "# R120 Error Atlas Summary",
        "",
        "Mode: no training, no new inference. This script reads existing COCO GT and prediction JSON files.",
        "",
        _markdown_table(
            [
                "run",
                "role",
                "images",
                "GT",
                "pred",
                "P50",
                "R50",
                "P75",
                "R75",
                "GT best IoU p50",
                "boundary F mean",
            ],
            run_rows,
        ),
        "",
        "## Target200 Delta",
        "",
        "R114 full200 is a reference row. R115 train200 is an oracle row trained on target200 GT.",
        "",
        _markdown_table(
            ["delta_label", "GT"],
            [[key, value] for key, value in summary["delta"]["label_counts"].items()],
        ),
        "",
        f"Mean delta IoU: `{_fmt_float(summary['delta']['mean_delta_iou'])}`.",
        f"Mean delta boundary F: `{_fmt_float(summary['delta']['mean_delta_boundary_f'])}`.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_model_delta_md(path: Path, delta: dict[str, Any]) -> None:
    density_rows = []
    for name, item in delta["by_density_bucket"].items():
        density_rows.append(
            [
                name,
                item["gt"],
                _fmt_float(item["r114_recall50"]),
                _fmt_float(item["r115_recall50"]),
                _fmt_float(item["r114_recall75"]),
                _fmt_float(item["r115_recall75"]),
                _fmt_float(item["mean_delta_iou"]),
                item["miss_to_hit"],
                item["low_to_good"],
                item["both_miss"],
                item["good_to_worse"],
            ]
        )
    area_rows = []
    for name, item in delta["by_area_bucket"].items():
        area_rows.append(
            [
                name,
                item["gt"],
                _fmt_float(item["r114_recall50"]),
                _fmt_float(item["r115_recall50"]),
                _fmt_float(item["r114_recall75"]),
                _fmt_float(item["r115_recall75"]),
                _fmt_float(item["mean_delta_iou"]),
                item["miss_to_hit"],
                item["low_to_good"],
                item["both_miss"],
                item["good_to_worse"],
            ]
        )
    lines = [
        "# R120 Model Delta Summary",
        "",
        "Comparison: R114 full200 reference vs R115 train200 oracle on the same target200 GT.",
        "",
        _markdown_table(["delta_label", "GT"], [[k, v] for k, v in delta["label_counts"].items()]),
        "",
        "## Density Buckets",
        "",
        _markdown_table(
            [
                "bucket",
                "GT",
                "R114 R50",
                "R115 R50",
                "R114 R75",
                "R115 R75",
                "mean dIoU",
                "miss_to_hit",
                "low_to_good",
                "both_miss",
                "good_to_worse",
            ],
            density_rows,
        ),
        "",
        "## Area Buckets",
        "",
        _markdown_table(
            [
                "bucket",
                "GT",
                "R114 R50",
                "R115 R50",
                "R114 R75",
                "R115 R75",
                "mean dIoU",
                "miss_to_hit",
                "low_to_good",
                "both_miss",
                "good_to_worse",
            ],
            area_rows,
        ),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _build_specs(args: argparse.Namespace) -> list[RunSpec]:
    return [
        RunSpec("r114_val28", "held-out", Path(args.gt_val28), Path(args.r114_val28_pred)),
        RunSpec("r114_remaining75", "held-out", Path(args.gt_r114_remaining75), Path(args.r114_remaining75_pred)),
        RunSpec("r114_full200_reference", "reference", Path(args.gt_target200), Path(args.r114_full200_pred)),
        RunSpec("r115_train200_oracle", "oracle", Path(args.gt_target200), Path(args.r115_train200_pred)),
        RunSpec("r115_val28_oracle_model", "held-out_eval_from_oracle_model", Path(args.gt_val28), Path(args.r115_val28_pred)),
    ]


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = _build_specs(args)
    analyses = [analyze_run(spec) for spec in specs]
    by_name = {analysis.spec.name: analysis for analysis in analyses}
    delta_rows, delta_summary = build_delta_rows(by_name["r114_full200_reference"], by_name["r115_train200_oracle"])

    gt_rows = [row for analysis in analyses for row in analysis.gt_rows]
    pred_rows = [row for analysis in analyses for row in analysis.pred_rows]
    bucket_rows = [row for analysis in analyses for row in analysis.bucket_rows]

    _write_csv(
        out_dir / "instance_match.csv",
        gt_rows,
        [
            "run",
            "role",
            "image_id",
            "file_name",
            "category_id",
            "gt_id",
            "gt_area",
            "area_bucket",
            "area_quantile_bucket",
            "image_gt_count",
            "density_bucket",
            "best_pred_index",
            "best_pred_score",
            "best_iou",
            "boundary_f_best",
            "tp50",
            "fn50",
            "matched_pred_index_50",
            "tp75",
            "fn75",
            "matched_pred_index_75",
        ],
    )
    _write_csv(
        out_dir / "prediction_fp.csv",
        pred_rows,
        [
            "run",
            "role",
            "image_id",
            "file_name",
            "category_id",
            "pred_index",
            "score",
            "bbox_x",
            "bbox_y",
            "bbox_w",
            "bbox_h",
            "image_gt_count",
            "density_bucket",
            "best_gt_id",
            "best_gt_area",
            "best_gt_area_bucket",
            "best_gt_area_quantile_bucket",
            "best_gt_iou",
            "boundary_f_best",
            "tp50",
            "fp50",
            "duplicate_fp50",
            "matched_gt_id_50",
            "tp75",
            "fp75",
            "duplicate_fp75",
            "matched_gt_id_75",
        ],
    )
    _write_csv(
        out_dir / "bucket_compare.csv",
        bucket_rows,
        [
            "run",
            "role",
            "bucket_type",
            "bucket",
            "images",
            "gt",
            "pred",
            "tp50",
            "fn50",
            "fp50",
            "precision50",
            "recall50",
            "duplicate_fp50",
            "tp75",
            "fn75",
            "fp75",
            "precision75",
            "recall75",
            "duplicate_fp75",
            "best_iou_mean",
            "boundary_f_mean",
        ],
    )
    _write_csv(
        out_dir / "delta_atlas.csv",
        delta_rows,
        [
            "gt_id",
            "image_id",
            "file_name",
            "category_id",
            "gt_area",
            "area_bucket",
            "area_quantile_bucket",
            "image_gt_count",
            "density_bucket",
            "r114_best_iou",
            "r115_best_iou",
            "delta_iou",
            "r114_state",
            "r115_state",
            "delta_label",
            "r114_tp50",
            "r115_tp50",
            "r114_tp75",
            "r115_tp75",
            "r114_boundary_f",
            "r115_boundary_f",
            "delta_boundary_f",
        ],
    )

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "read existing GT and prediction JSON; no training; no new inference",
        "out_dir": str(out_dir),
        "runs": {analysis.spec.name: analysis.summary for analysis in analyses},
        "delta": delta_summary,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_summary_md(out_dir / "summary.md", summary)
    write_model_delta_md(out_dir / "model_delta_summary.md", delta_summary)
    print(f"wrote {out_dir}")


def _self_test() -> None:
    a = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 0]], dtype=bool)
    b = np.array([[1, 0, 0], [0, 1, 1], [0, 0, 0]], dtype=bool)
    assert abs(_mask_iou(a, b) - 0.5) < 1e-9

    ious = np.array([[0.90, 0.10], [0.80, 0.20], [0.10, 0.76]], dtype=np.float64)
    p_to_g, g_to_p, dup = _greedy_match(ious, 0.50)
    assert p_to_g == [0, None, 1]
    assert g_to_p == [0, 2]
    assert dup == [False, True, False]
    p_to_g75, g_to_p75, dup75 = _greedy_match(ious, 0.75)
    assert p_to_g75 == [0, None, 1]
    assert g_to_p75 == [0, 2]
    assert dup75 == [False, True, False]

    square = np.zeros((6, 6), dtype=bool)
    square[2:4, 2:4] = True
    shifted = np.zeros((6, 6), dtype=bool)
    shifted[2:4, 3:5] = True
    assert _boundary_f_score(square, square) == 1.0
    shifted_score = _boundary_f_score(shifted, square)
    assert 0.0 < shifted_score <= 1.0

    assert _delta_label(0.20, 0.60) == "miss_to_hit"
    assert _delta_label(0.60, 0.80) == "low_to_good"
    assert _delta_label(0.10, 0.20) == "both_miss"
    assert _delta_label(0.90, 0.60) == "good_to_worse"
    print("self-test passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build R120 error atlas from existing MagFormer COCO GT and prediction JSON files."
    )
    parser.add_argument("--self-test", action="store_true", help="run synthetic checks for IoU, matching, boundary F, delta labels")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--gt-val28", default=DEFAULT_GT_VAL28)
    parser.add_argument("--gt-target200", default=DEFAULT_GT_TARGET200)
    parser.add_argument("--gt-r114-remaining75", default=DEFAULT_GT_R114_REMAINING75)
    parser.add_argument("--r114-val28-pred", default=DEFAULT_R114_VAL28_PRED)
    parser.add_argument("--r114-remaining75-pred", default=DEFAULT_R114_REMAINING75_PRED)
    parser.add_argument("--r114-full200-pred", default=DEFAULT_R114_FULL200_PRED)
    parser.add_argument("--r115-train200-pred", default=DEFAULT_R115_TRAIN200_PRED)
    parser.add_argument("--r115-val28-pred", default=DEFAULT_R115_VAL28_PRED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        _self_test()
        return
    run(args)


if __name__ == "__main__":
    main()
