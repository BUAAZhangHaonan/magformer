#!/usr/bin/env python3
"""Build high-precision offline pseudo banks from COCO prediction JSON."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from pycocotools import mask as mask_utils

from tools.audit_dense_geometry import (
    AuditError,
    BUCKETS,
    _bucket_name,
    _index_gt,
    _load_json,
    _mask_iou,
    _require_segmentation,
    _segmentation_to_rle,
    _summary,
)


class BankError(AuditError):
    """Raised when pseudo-bank inputs are missing or malformed."""


@dataclass(frozen=True)
class BankConfig:
    score_min: float = 0.0
    fill_min: float = 0.0
    area_min: float = 0.0
    bbox_min_side: float = 0.0
    bbox_max_side: float = math.inf
    per_image_topk: int | None = None


@dataclass(frozen=True)
class DynamicBucketRule:
    low: int
    high: int | float
    score_min: float | None = None
    fill_min: float | None = None
    area_min: float | None = None
    bbox_min_side: float | None = None
    bbox_max_side: float | None = None
    per_image_topk: int | None = None

    def matches(self, pred_count: int) -> bool:
        return self.low <= pred_count <= self.high

    def apply(self, config: BankConfig) -> BankConfig:
        kwargs: dict[str, Any] = {}
        for field in ("score_min", "fill_min", "area_min", "bbox_min_side", "bbox_max_side", "per_image_topk"):
            value = getattr(self, field)
            if value is not None:
                kwargs[field] = value
        return replace(config, **kwargs)

    def label(self) -> str:
        high = "inf" if self.high == math.inf else str(int(self.high))
        parts = []
        if self.score_min is not None:
            parts.append(f"score>={self.score_min}")
        if self.fill_min is not None:
            parts.append(f"fill>={self.fill_min}")
        if self.area_min is not None:
            parts.append(f"area>={self.area_min}")
        if self.bbox_min_side is not None:
            parts.append(f"min_side>={self.bbox_min_side}")
        if self.bbox_max_side is not None:
            parts.append(f"max_side<={self.bbox_max_side}")
        if self.per_image_topk is not None:
            parts.append(f"topk<={self.per_image_topk}")
        return f"{self.low}-{high}:" + ",".join(parts)


@dataclass(frozen=True)
class Candidate:
    pred: dict[str, Any]
    image: dict[str, Any]
    rle: dict[str, Any]
    segmentation: dict[str, Any]
    bbox: list[float]
    area: float
    fill_ratio: float
    score: float
    category_id: int


def _require_coco(payload: Any, context: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise BankError(f"{context} must be a COCO annotation object.")
    if not isinstance(payload.get("images"), list) or not isinstance(payload.get("categories"), list):
        raise BankError(f"{context} must contain images and categories lists.")
    return payload


def _require_predictions(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, list):
        raise BankError("Predictions JSON must be a COCO result list.")
    for idx, pred in enumerate(payload):
        if not isinstance(pred, dict):
            raise BankError(f"Prediction {idx} must be an object.")
        for key in ("image_id", "category_id", "score"):
            if key not in pred:
                raise BankError(f"Prediction {idx} missing {key}.")
        _require_segmentation(pred, f"prediction {idx}")
    return payload


def _rle_for_mask_utils(rle: dict[str, Any]) -> dict[str, Any]:
    counts = rle.get("counts")
    if isinstance(counts, str):
        return {"size": [int(rle["size"][0]), int(rle["size"][1])], "counts": counts.encode("ascii")}
    if isinstance(counts, bytes):
        return {"size": [int(rle["size"][0]), int(rle["size"][1])], "counts": counts}
    raise BankError("RLE counts must be compressed bytes or ASCII string.")


def _rle_for_json(rle: dict[str, Any]) -> dict[str, Any]:
    counts = rle.get("counts")
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    if not isinstance(counts, str):
        raise BankError("RLE counts must be compressed bytes or ASCII string.")
    return {"size": [int(rle["size"][0]), int(rle["size"][1])], "counts": counts}


def _mask_geometry(segmentation: dict[str, Any] | list[Any], height: int, width: int) -> tuple[dict[str, Any], dict[str, Any], list[float], float, float]:
    rle = _segmentation_to_rle(segmentation, height, width)
    json_rle = _rle_for_json(rle)
    py_rle = _rle_for_mask_utils(json_rle)
    area = float(mask_utils.area(py_rle))
    bbox = [float(v) for v in mask_utils.toBbox(py_rle).tolist()]
    bbox_area = max(0.0, bbox[2]) * max(0.0, bbox[3])
    fill_ratio = area / bbox_area if bbox_area > 0 else 0.0
    return py_rle, json_rle, bbox, area, float(fill_ratio)


def _index_images(coco: dict[str, Any], context: str) -> dict[int, dict[str, Any]]:
    images: dict[int, dict[str, Any]] = {}
    for idx, image in enumerate(coco.get("images", [])):
        if not isinstance(image, dict) or "id" not in image or "height" not in image or "width" not in image:
            raise BankError(f"{context} image {idx} missing id/height/width.")
        images[int(image["id"])] = image
    if not images:
        raise BankError(f"{context} has no images.")
    return images


def _category_ids(coco: dict[str, Any]) -> set[int]:
    ids = set()
    for category in coco.get("categories", []):
        if not isinstance(category, dict) or "id" not in category:
            raise BankError("Source category missing id.")
        ids.add(int(category["id"]))
    if not ids:
        raise BankError("Source COCO has no categories.")
    return ids


def _candidate_from_prediction(pred: dict[str, Any], image: dict[str, Any], category_ids: set[int]) -> Candidate:
    category_id = int(pred["category_id"])
    if category_id not in category_ids:
        raise BankError(f"Prediction category_id {category_id} is not present in source categories.")
    score = float(pred["score"])
    height = int(image["height"])
    width = int(image["width"])
    py_rle, json_rle, bbox, area, fill_ratio = _mask_geometry(_require_segmentation(pred, "prediction"), height, width)
    return Candidate(
        pred=pred,
        image=image,
        rle=py_rle,
        segmentation=json_rle,
        bbox=bbox,
        area=area,
        fill_ratio=fill_ratio,
        score=score,
        category_id=category_id,
    )


def _config_for_count(base: BankConfig, rules: Iterable[DynamicBucketRule], pred_count: int) -> BankConfig:
    for rule in rules:
        if rule.matches(pred_count):
            return rule.apply(base)
    return base


def _passes(candidate: Candidate, config: BankConfig) -> bool:
    width = float(candidate.bbox[2])
    height = float(candidate.bbox[3])
    return (
        candidate.score >= config.score_min
        and candidate.fill_ratio >= config.fill_min
        and candidate.area >= config.area_min
        and min(width, height) >= config.bbox_min_side
        and max(width, height) <= config.bbox_max_side
    )


def _remove_overlaps(candidates: list[Candidate], nms_iou: float | None) -> tuple[list[Candidate], int]:
    if nms_iou is None:
        return candidates, 0
    kept: list[Candidate] = []
    removed = 0
    for candidate in candidates:
        duplicate = False
        for existing in kept:
            if existing.category_id != candidate.category_id:
                continue
            iou = float(mask_utils.iou([candidate.rle], [existing.rle], [0])[0][0])
            if iou >= nms_iou:
                duplicate = True
                break
        if duplicate:
            removed += 1
        else:
            kept.append(candidate)
    return kept, removed


def _build_annotation(ann_id: int, image_id: int, candidate: Candidate) -> dict[str, Any]:
    return {
        "id": ann_id,
        "image_id": int(image_id),
        "category_id": candidate.category_id,
        "segmentation": candidate.segmentation,
        "area": candidate.area,
        "bbox": candidate.bbox,
        "iscrowd": 0,
        "score": candidate.score,
        "pseudo_fill_ratio": candidate.fill_ratio,
    }


def _empty_bucket_rows() -> dict[str, dict[str, Any]]:
    return {
        name: {
            "images": 0,
            "gt": 0,
            "raw_predictions": 0,
            "kept": 0,
            "tp50": 0,
            "fp50": 0,
            "fn50": 0,
            "tp75": 0,
            "fp75": 0,
            "fn75": 0,
            "P50": None,
            "R50": None,
            "P75": None,
            "R75": None,
        }
        for name, _, _ in BUCKETS
    }


def _match_counts(
    *,
    gt_anns: list[dict[str, Any]],
    candidates: list[Candidate],
    image_info: dict[str, Any],
    threshold: float,
) -> tuple[int, int, int, list[float], list[float]]:
    if not candidates:
        return 0, 0, len(gt_anns), [], []
    if not gt_anns:
        return 0, len(candidates), 0, [], []

    height = int(image_info["height"])
    width = int(image_info["width"])
    gt_rles = []
    gt_areas = []
    for gt in gt_anns:
        gt_rle = _segmentation_to_rle(_require_segmentation(gt, "annotation"), height, width)
        json_rle = _rle_for_json(gt_rle)
        py_rle = _rle_for_mask_utils(json_rle)
        gt_rles.append(py_rle)
        gt_areas.append(float(mask_utils.area(py_rle)))

    iou_matrix = mask_utils.iou([candidate.rle for candidate in candidates], gt_rles, [0] * len(gt_rles))
    matched_gt: set[int] = set()
    tp_ious: list[float] = []
    area_ratios: list[float] = []
    for pred_idx, candidate in enumerate(candidates):
        row = iou_matrix[pred_idx]
        best_gt_idx = int(np.argmax(row))
        best_iou = float(row[best_gt_idx])
        if best_iou >= threshold and best_gt_idx not in matched_gt:
            matched_gt.add(best_gt_idx)
            tp_ious.append(best_iou)
            gt_area = gt_areas[best_gt_idx]
            if gt_area > 0:
                area_ratios.append(candidate.area / gt_area)
    tp = len(tp_ious)
    fp = len(candidates) - tp
    fn = len(gt_anns) - len(matched_gt)
    return tp, fp, fn, tp_ious, area_ratios


def _safe_pr(tp: int, fp: int, fn: int) -> tuple[float | None, float | None]:
    precision = float(tp / (tp + fp)) if (tp + fp) else None
    recall = float(tp / (tp + fn)) if (tp + fn) else None
    return precision, recall


def _evaluate_hidden_gt(
    hidden_gt: dict[str, Any],
    target_images: dict[int, dict[str, Any]],
    raw_by_image: dict[int, list[dict[str, Any]]],
    kept_by_image: dict[int, list[Candidate]],
) -> dict[str, Any]:
    gt_images, anns_by_image = _index_gt(hidden_gt)
    missing = sorted(set(target_images) - set(gt_images))
    if missing:
        raise BankError(f"Hidden GT missing target image ids: {missing[:10]}")
    bucket_rows = _empty_bucket_rows()
    totals = {"gt": 0, "kept": 0, "tp50": 0, "fp50": 0, "fn50": 0, "tp75": 0, "fp75": 0, "fn75": 0}
    tp50_ious: list[float] = []
    tp75_ious: list[float] = []
    area_ratios: list[float] = []
    for image_id, image_info in target_images.items():
        gt_anns = anns_by_image.get(image_id, [])
        candidates = kept_by_image.get(image_id, [])
        tp50, fp50, fn50, ious50, ratios50 = _match_counts(gt_anns=gt_anns, candidates=candidates, image_info=image_info, threshold=0.5)
        tp75, fp75, fn75, ious75, _ratios75 = _match_counts(gt_anns=gt_anns, candidates=candidates, image_info=image_info, threshold=0.75)
        bucket = _bucket_name(len(gt_anns))
        row = bucket_rows[bucket]
        row["images"] += 1
        row["gt"] += len(gt_anns)
        row["raw_predictions"] += len(raw_by_image.get(image_id, []))
        row["kept"] += len(candidates)
        for key, value in (("tp50", tp50), ("fp50", fp50), ("fn50", fn50), ("tp75", tp75), ("fp75", fp75), ("fn75", fn75)):
            row[key] += value
            totals[key] += value
        totals["gt"] += len(gt_anns)
        totals["kept"] += len(candidates)
        tp50_ious.extend(ious50)
        tp75_ious.extend(ious75)
        area_ratios.extend(ratios50)
    for row in bucket_rows.values():
        row["P50"], row["R50"] = _safe_pr(int(row["tp50"]), int(row["fp50"]), int(row["fn50"]))
        row["P75"], row["R75"] = _safe_pr(int(row["tp75"]), int(row["fp75"]), int(row["fn75"]))
    p50, r50 = _safe_pr(totals["tp50"], totals["fp50"], totals["fn50"])
    p75, r75 = _safe_pr(totals["tp75"], totals["fp75"], totals["fn75"])
    return {
        "mask": {"P50": p50, "R50": r50, "P75": p75, "R75": r75},
        "totals": totals,
        "buckets": bucket_rows,
        "matched_tp_mask_iou_50": _summary(tp50_ious),
        "matched_tp_mask_iou_75": _summary(tp75_ious),
        "matched_tp_pred_gt_mask_area_ratio_50": _summary(area_ratios),
    }


def _non_eval_buckets(target_images: dict[int, dict[str, Any]], raw_by_image: dict[int, list[dict[str, Any]]], kept_by_image: dict[int, list[Candidate]]) -> dict[str, dict[str, Any]]:
    rows = _empty_bucket_rows()
    for image_id in target_images:
        raw_count = len(raw_by_image.get(image_id, []))
        row = rows[_bucket_name(raw_count)]
        row["images"] += 1
        row["raw_predictions"] += raw_count
        row["kept"] += len(kept_by_image.get(image_id, []))
    return rows


def _build_markdown(summary: dict[str, Any]) -> str:
    quality = summary.get("quality", {}).get("mask", {})
    inputs = summary["inputs"]
    outputs = summary["outputs"]
    totals = summary["totals"]
    config = summary["config"]
    lines = [
        "# Pseudo Bank Quality: {}".format(summary["name"]),
        "",
        "- predictions: `{}`".format(inputs["predictions"]),
        "- source_coco: `{}`".format(inputs["source_coco"]),
        "- target_coco: `{}`".format(inputs["target_coco"]),
        "- output: `{}`".format(outputs["bank_json"]),
        "- raw predictions: {}".format(totals["raw_predictions"]),
        "- kept: {}".format(totals["kept"]),
        "- kept/image: {:.3f}".format(totals["kept_per_image"]),
        "- P50/R50/P75/R75: {} / {} / {} / {}".format(
            quality.get("P50"), quality.get("R50"), quality.get("P75"), quality.get("R75")
        ),
        "",
        "## Filters",
        "",
        "- score_min: {}".format(config["score_min"]),
        "- fill_min: {}".format(config["fill_min"]),
        "- area_min: {}".format(config["area_min"]),
        "- bbox_min_side: {}".format(config["bbox_min_side"]),
        "- bbox_max_side: {}".format(config["bbox_max_side"]),
        "- per_image_topk: {}".format(config["per_image_topk"]),
        "- overlap_nms_iou: {}".format(config["overlap_nms_iou"]),
        "",
        "## Dense Buckets",
        "",
        "| bucket | images | gt | raw_pred | kept | P50 | R50 | P75 | R75 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for bucket, row in summary["buckets"].items():
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                bucket,
                row["images"],
                row["gt"],
                row["raw_predictions"],
                row["kept"],
                row["P50"],
                row["R50"],
                row["P75"],
                row["R75"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def build_pseudo_bank(
    *,
    predictions_json: str | Path,
    source_coco_json: str | Path,
    target_coco_json: str | Path,
    output_json: str | Path,
    metrics_json: str | Path,
    metrics_md: str | Path,
    config: BankConfig,
    hidden_gt_json: str | Path | None = None,
    dynamic_rules: list[DynamicBucketRule] | None = None,
    overlap_nms_iou: float | None = None,
    name: str = "pseudo_bank",
) -> dict[str, Any]:
    dynamic_rules = dynamic_rules or []
    predictions = _require_predictions(_load_json(predictions_json))
    source_coco = _require_coco(_load_json(source_coco_json), "Source COCO")
    target_coco = _require_coco(_load_json(target_coco_json), "Target COCO")
    target_images = _index_images(target_coco, "Target COCO")
    category_ids = _category_ids(source_coco)

    raw_by_image: dict[int, list[dict[str, Any]]] = {image_id: [] for image_id in target_images}
    for pred in predictions:
        image_id = int(pred["image_id"])
        if image_id not in target_images:
            raise BankError(f"Prediction references image_id {image_id}, which is not in target COCO.")
        raw_by_image[image_id].append(pred)
    for rows in raw_by_image.values():
        rows.sort(key=lambda item: float(item["score"]), reverse=True)

    kept_by_image: dict[int, list[Candidate]] = {image_id: [] for image_id in target_images}
    filtered_counts = {"score": 0, "fill": 0, "area": 0, "side": 0}
    nms_removed = 0
    score_values: list[float] = []
    fill_values: list[float] = []
    area_values: list[float] = []
    annotations: list[dict[str, Any]] = []
    ann_id = 1
    for image_id, rows in raw_by_image.items():
        image_config = _config_for_count(config, dynamic_rules, len(rows))
        passed: list[Candidate] = []
        for pred in rows:
            if float(pred["score"]) < image_config.score_min:
                filtered_counts["score"] += 1
                continue
            candidate = _candidate_from_prediction(pred, target_images[image_id], category_ids)
            if candidate.fill_ratio < image_config.fill_min:
                filtered_counts["fill"] += 1
            elif candidate.area < image_config.area_min:
                filtered_counts["area"] += 1
            elif min(candidate.bbox[2], candidate.bbox[3]) < image_config.bbox_min_side or max(candidate.bbox[2], candidate.bbox[3]) > image_config.bbox_max_side:
                filtered_counts["side"] += 1
            elif _passes(candidate, image_config):
                passed.append(candidate)
        passed.sort(key=lambda item: item.score, reverse=True)
        passed, removed = _remove_overlaps(passed, overlap_nms_iou)
        nms_removed += removed
        if image_config.per_image_topk is not None:
            passed = passed[: image_config.per_image_topk]
        kept_by_image[image_id] = passed
        for candidate in passed:
            score_values.append(candidate.score)
            fill_values.append(candidate.fill_ratio)
            area_values.append(candidate.area)
            annotations.append(_build_annotation(ann_id, image_id, candidate))
            ann_id += 1

    output = {
        "images": target_coco["images"],
        "categories": source_coco["categories"],
        "annotations": annotations,
    }
    for optional_key in ("info", "licenses"):
        if optional_key in target_coco:
            output[optional_key] = target_coco[optional_key]

    hidden_quality: dict[str, Any] | None = None
    if hidden_gt_json is not None:
        hidden_gt = _require_coco(_load_json(hidden_gt_json), "Hidden GT COCO")
        hidden_quality = _evaluate_hidden_gt(hidden_gt, target_images, raw_by_image, kept_by_image)
        buckets = hidden_quality["buckets"]
    else:
        buckets = _non_eval_buckets(target_images, raw_by_image, kept_by_image)

    totals = {
        "images": len(target_images),
        "raw_predictions": len(predictions),
        "kept": len(annotations),
        "kept_per_image": float(len(annotations) / len(target_images)) if target_images else 0.0,
        "filtered": filtered_counts,
        "overlap_nms_removed": nms_removed,
    }
    summary = {
        "name": name,
        "inputs": {
            "predictions": str(predictions_json),
            "source_coco": str(source_coco_json),
            "target_coco": str(target_coco_json),
            "hidden_gt": str(hidden_gt_json) if hidden_gt_json is not None else None,
        },
        "outputs": {"bank_json": str(output_json), "metrics_json": str(metrics_json), "metrics_md": str(metrics_md)},
        "config": {
            **config.__dict__,
            "bbox_max_side": config.bbox_max_side if math.isfinite(config.bbox_max_side) else "inf",
            "overlap_nms_iou": overlap_nms_iou,
        },
        "dynamic_rules": [rule.label() for rule in dynamic_rules],
        "totals": totals,
        "score_distribution": _summary(score_values),
        "fill_ratio_distribution": _summary(fill_values),
        "mask_area_distribution": _summary(area_values),
        "quality": hidden_quality or {},
        "buckets": buckets,
    }

    output_path = Path(output_json)
    metrics_json_path = Path(metrics_json)
    metrics_md_path = Path(metrics_md)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_json_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_md_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    metrics_json_path.write_text(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    metrics_md_path.write_text(_build_markdown(summary), encoding="utf-8")
    return summary


def _parse_topk(value: str) -> int | None:
    if value.lower() in {"none", "null"}:
        return None
    parsed = int(value)
    if parsed < 1:
        raise BankError("per-image topk must be positive or none.")
    return parsed


def _parse_high(value: str) -> int | float:
    if value.lower() in {"inf", "infinity", "+"}:
        return math.inf
    return int(value)


def _parse_dynamic_rule(value: str) -> DynamicBucketRule:
    if ":" not in value or "-" not in value.split(":", 1)[0]:
        raise BankError(f"Dynamic rule must look like LOW-HIGH:key=value,...: {value}")
    span, body = value.split(":", 1)
    low_text, high_text = span.split("-", 1)
    kwargs: dict[str, Any] = {"low": int(low_text), "high": _parse_high(high_text)}
    key_map = {
        "score": "score_min",
        "score_min": "score_min",
        "fill": "fill_min",
        "fill_min": "fill_min",
        "area": "area_min",
        "area_min": "area_min",
        "min_side": "bbox_min_side",
        "bbox_min_side": "bbox_min_side",
        "max_side": "bbox_max_side",
        "bbox_max_side": "bbox_max_side",
        "topk": "per_image_topk",
        "per_image_topk": "per_image_topk",
    }
    for item in body.split(","):
        if not item:
            continue
        if "=" not in item:
            raise BankError(f"Dynamic rule item must use key=value: {item}")
        key, raw = item.split("=", 1)
        if key not in key_map:
            raise BankError(f"Unknown dynamic rule key: {key}")
        field = key_map[key]
        kwargs[field] = _parse_topk(raw) if field == "per_image_topk" else float(raw)
    return DynamicBucketRule(**kwargs)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="COCO result prediction JSON.")
    parser.add_argument("--source-coco", required=True, help="Source COCO annotation JSON providing categories.")
    parser.add_argument("--target-coco", required=True, help="Target COCO annotation JSON providing pseudo-bank images.")
    parser.add_argument("--output-json", required=True, help="Output pseudo-bank COCO annotation JSON.")
    parser.add_argument("--metrics-json", required=True, help="Output metrics JSON.")
    parser.add_argument("--metrics-md", required=True, help="Output metrics Markdown.")
    parser.add_argument("--hidden-gt", default=None, help="Optional hidden-GT COCO annotation JSON for quality evaluation.")
    parser.add_argument("--name", default="pseudo_bank")
    parser.add_argument("--score-min", type=float, default=0.0)
    parser.add_argument("--fill-min", type=float, default=0.0)
    parser.add_argument("--area-min", type=float, default=0.0)
    parser.add_argument("--bbox-min-side", type=float, default=0.0)
    parser.add_argument("--bbox-max-side", type=float, default=math.inf)
    parser.add_argument("--per-image-topk", default="none", help="Per-image kept top-k after filtering, or none.")
    parser.add_argument("--dynamic-bucket", action="append", default=[], help="LOW-HIGH:key=value,... override by raw pred_count bucket.")
    parser.add_argument("--overlap-nms-iou", type=float, default=None, help="Optional same-image/category mask NMS IoU threshold. Default off.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = BankConfig(
        score_min=args.score_min,
        fill_min=args.fill_min,
        area_min=args.area_min,
        bbox_min_side=args.bbox_min_side,
        bbox_max_side=args.bbox_max_side,
        per_image_topk=_parse_topk(args.per_image_topk),
    )
    build_pseudo_bank(
        predictions_json=args.predictions,
        source_coco_json=args.source_coco,
        target_coco_json=args.target_coco,
        output_json=args.output_json,
        metrics_json=args.metrics_json,
        metrics_md=args.metrics_md,
        config=config,
        hidden_gt_json=args.hidden_gt,
        dynamic_rules=[_parse_dynamic_rule(item) for item in args.dynamic_bucket],
        overlap_nms_iou=args.overlap_nms_iou,
        name=args.name,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
