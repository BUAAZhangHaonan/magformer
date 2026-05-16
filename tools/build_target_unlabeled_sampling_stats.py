#!/usr/bin/env python3
"""Build prediction-only target-unlabeled sampling stats for VC-SUDA R52."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from pycocotools import mask as mask_utils


FORBIDDEN_FIELDS = {"gt_count", "gt_density_bucket", "annotations"}


def _load_json(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _check_forbidden_fields(value: Any, *, context: str, allow_target_coco_annotations: bool = False) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in FORBIDDEN_FIELDS and not (allow_target_coco_annotations and key == "annotations"):
                raise ValueError(f"{context} contains forbidden prediction/GT field: {key}")
            _check_forbidden_fields(child, context=f"{context}.{key}", allow_target_coco_annotations=False)
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            _check_forbidden_fields(child, context=f"{context}[{idx}]", allow_target_coco_annotations=False)


def _percentile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(float(v) for v in values)
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _mask_area(segmentation: Any) -> float:
    if segmentation is None:
        raise ValueError("prediction record is missing segmentation; mask_area stats require masks")
    if isinstance(segmentation, list):
        rles = mask_utils.frPyObjects(segmentation, 1024, 1024)
        return float(mask_utils.area(rles).sum())
    if isinstance(segmentation, Mapping):
        counts = segmentation.get("counts")
        rle = dict(segmentation)
        if isinstance(counts, str):
            rle["counts"] = counts.encode("ascii")
        return float(mask_utils.area(rle))
    raise ValueError(f"unsupported segmentation type for mask area: {type(segmentation).__name__}")


def _score_stats(scores: Sequence[float]) -> Dict[str, Optional[float]]:
    if not scores:
        return {"min": None, "p10": None, "p50": None, "p90": None, "max": None, "mean": None}
    values = [float(score) for score in scores]
    return {
        "min": min(values),
        "p10": _percentile(values, 0.10),
        "p50": _percentile(values, 0.50),
        "p90": _percentile(values, 0.90),
        "max": max(values),
        "mean": float(mean(values)),
    }


def _aggregate_predictions(path: Path, *, label: str) -> Dict[int, Dict[str, Any]]:
    payload = _load_json(path)
    _check_forbidden_fields(payload, context=label)
    if not isinstance(payload, list):
        raise ValueError(f"{label} prediction JSON must be a COCO results list")

    grouped: Dict[int, List[Mapping[str, Any]]] = defaultdict(list)
    for idx, item in enumerate(payload):
        if not isinstance(item, Mapping):
            raise ValueError(f"{label}[{idx}] must be an object")
        if "image_id" not in item:
            raise ValueError(f"{label}[{idx}] is missing image_id")
        grouped[int(item["image_id"])].append(item)

    stats = {}
    for image_id, items in grouped.items():
        areas = [_mask_area(item.get("segmentation")) for item in items]
        scores = [float(item.get("score", 0.0)) for item in items]
        topk_values = [bool(item["topk_truncated"]) for item in items if "topk_truncated" in item]
        entry = {
            "pred_count": len(items),
            "mask_area_p50": _percentile(areas, 0.50),
            "mask_area_p10": _percentile(areas, 0.10),
            "small256_ratio": float(sum(area <= 256.0 for area in areas) / len(areas)) if areas else 0.0,
            "small300_ratio": float(sum(area <= 300.0 for area in areas) / len(areas)) if areas else 0.0,
            "score": _score_stats(scores),
        }
        if topk_values:
            entry["topk_truncated"] = any(topk_values)
        stats[image_id] = entry
    return stats


def _empty_prediction_stats() -> Dict[str, Any]:
    return {
        "pred_count": 0,
        "mask_area_p50": None,
        "mask_area_p10": None,
        "small256_ratio": 0.0,
        "small300_ratio": 0.0,
        "score": _score_stats([]),
    }


def _load_target_images(path: Path) -> List[Dict[str, Any]]:
    payload = _load_json(path)
    _check_forbidden_fields(payload, context="target_ann", allow_target_coco_annotations=True)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("images"), list):
        raise ValueError("target annotation JSON must contain an images list")
    images = []
    for idx, item in enumerate(payload["images"]):
        if "id" not in item:
            raise ValueError(f"target_ann.images[{idx}] is missing id")
        images.append({"id": int(item["id"]), "file_name": item.get("file_name")})
    return images


def build_sampling_stats(
    target_ann: str | Path,
    r46_predictions: str | Path,
    r12_predictions: str | Path,
    output: str | Path,
) -> Dict[str, Any]:
    target_ann = Path(target_ann)
    r46_predictions = Path(r46_predictions)
    r12_predictions = Path(r12_predictions)
    output = Path(output)

    images = _load_target_images(target_ann)
    r46 = _aggregate_predictions(r46_predictions, label="r46_predictions")
    r12 = _aggregate_predictions(r12_predictions, label="r12_predictions")

    rows = []
    repeat_counts = Counter()
    sequence_length = 0
    for image in images:
        image_id = image["id"]
        r46_stats = r46.get(image_id, _empty_prediction_stats())
        r12_stats = r12.get(image_id, _empty_prediction_stats())
        dense_candidate = min(r46_stats["pred_count"], r12_stats["pred_count"]) >= 90
        dense_tiny_candidate = dense_candidate and (
            min(r46_stats["small256_ratio"], r12_stats["small256_ratio"]) >= 0.30
            or min(r46_stats["small300_ratio"], r12_stats["small300_ratio"]) >= 0.35
        )
        if dense_tiny_candidate:
            bucket = "dense_tiny"
            repeat = 3
        elif dense_candidate:
            bucket = "dense"
            repeat = 2
        else:
            bucket = "normal"
            repeat = 1

        repeat_counts[bucket] += 1
        sequence_length += repeat
        row = {
            "image_id": image_id,
            "file_name": image.get("file_name"),
            "bucket": bucket,
            "repeat": repeat,
            "dense_candidate": dense_candidate,
            "dense_tiny_candidate": dense_tiny_candidate,
            "r46": r46_stats,
            "r12": r12_stats,
        }
        rows.append(row)

    summary = {
        "image_count": len(rows),
        "repeat_counts": {
            "normal": int(repeat_counts.get("normal", 0)),
            "dense": int(repeat_counts.get("dense", 0)),
            "dense_tiny": int(repeat_counts.get("dense_tiny", 0)),
        },
        "sequence_length": int(sequence_length),
    }
    payload = {"version": 1, "summary": summary, "images": rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build prediction-only target_unlabeled sampling stats for R52."
    )
    parser.add_argument("--target-ann", required=True, type=Path, help="Target COCO annotation JSON; only images are used.")
    parser.add_argument("--r46-pred", required=True, type=Path, help="R46 COCO instance prediction JSON.")
    parser.add_argument("--r12-pred", required=True, type=Path, help="R12/R33 COCO instance prediction JSON.")
    parser.add_argument("--output", required=True, type=Path, help="Output sampling stats JSON.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    summary = build_sampling_stats(args.target_ann, args.r46_pred, args.r12_pred, args.output)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
