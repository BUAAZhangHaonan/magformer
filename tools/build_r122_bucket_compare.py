#!/usr/bin/env python3
"""Build R122 remaining75 bucket_compare.csv from existing COCO GT and predictions."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    from pycocotools import mask as mask_utils
except ImportError:  # pragma: no cover - exercised only in missing dependency envs.
    mask_utils = None


DEFAULT_GT_JSON = (
    "magformer_datasets/pseudo_real_512/annotations/"
    "instances_target_unlabeled_r114_balanced_minus125.json"
)
DEFAULT_PRED_JSON = (
    "output/diagnostics/"
    "r122_depth_boundary_w001_iter0099_remaining75_1024_backmap_topk200_20260518/"
    "coco_instances_results.json"
)
DEFAULT_OUTPUT_CSV = (
    "output/diagnostics/"
    "r122_depth_boundary_w001_iter0099_bucket_compare_20260518/"
    "bucket_compare.csv"
)
CSV_NAN = ""


class BuilderError(RuntimeError):
    """Raised when required inputs are missing or malformed."""


@dataclass(frozen=True)
class RunSpec:
    name: str
    gt_json: Path
    pred_json: Path


def _load_json(path: Path) -> Any:
    if not path.exists():
        raise BuilderError(f"missing input file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _require_pycocotools() -> None:
    if mask_utils is None:
        raise BuilderError("pycocotools is required to decode COCO segmentation masks")


def _require_keys(item: dict[str, Any], keys: tuple[str, ...], label: str) -> None:
    for key in keys:
        if key not in item:
            raise BuilderError(f"{label} is missing {key}")


def _require_coco(payload: Any, path: Path) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise BuilderError(f"{path} must be a COCO annotation object")
    for key in ("images", "annotations", "categories"):
        if not isinstance(payload.get(key), list):
            raise BuilderError(f"{path} is missing COCO field: {key}")
    for idx, image in enumerate(payload["images"]):
        if not isinstance(image, dict):
            raise BuilderError(f"images[{idx}] must be an object")
        _require_keys(image, ("id", "height", "width"), f"images[{idx}]")
    for idx, ann in enumerate(payload["annotations"]):
        if not isinstance(ann, dict):
            raise BuilderError(f"annotations[{idx}] must be an object")
        _require_keys(
            ann,
            ("id", "image_id", "category_id", "segmentation", "area"),
            f"annotations[{idx}]",
        )
    return payload


def _require_predictions(payload: Any, path: Path) -> list[dict[str, Any]]:
    if not isinstance(payload, list):
        raise BuilderError(f"{path} must be a COCO result list")
    for idx, pred in enumerate(payload):
        if not isinstance(pred, dict):
            raise BuilderError(f"predictions[{idx}] must be an object")
        _require_keys(
            pred,
            ("image_id", "category_id", "score", "bbox", "segmentation"),
            f"predictions[{idx}]",
        )
    return payload


def _build_images(coco: dict[str, Any]) -> dict[int, dict[str, Any]]:
    images: dict[int, dict[str, Any]] = {}
    for image in coco["images"]:
        image_id = int(image["id"])
        if image_id in images:
            raise BuilderError(f"duplicate image id={image_id}")
        images[image_id] = image
    return images


def _mask_rle(segmentation: Any, height: int, width: int, label: str) -> dict[str, Any]:
    _require_pycocotools()
    try:
        if isinstance(segmentation, dict):
            rle = dict(segmentation)
            if isinstance(rle.get("counts"), str):
                rle["counts"] = rle["counts"].encode("ascii")
            return rle
        if isinstance(segmentation, list):
            return mask_utils.merge(mask_utils.frPyObjects(segmentation, height, width))
    except Exception as exc:  # noqa: BLE001 - convert pycocotools detail to input-specific error.
        raise BuilderError(f"{label} has invalid segmentation: {exc}") from exc
    raise BuilderError(f"{label} has unsupported segmentation type: {type(segmentation).__name__}")


def _decode_mask(rle: dict[str, Any], label: str) -> np.ndarray:
    _require_pycocotools()
    try:
        decoded = mask_utils.decode(rle)
    except Exception as exc:  # noqa: BLE001
        raise BuilderError(f"{label} could not be decoded: {exc}") from exc
    if decoded.ndim == 3:
        decoded = decoded[:, :, 0]
    return decoded.astype(bool)


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
    return "large_gt1024"


def _density_bucket(count: int) -> str:
    if count <= 25:
        return "low25"
    if count <= 50:
        return "mid50"
    return "high100"


def _finite_mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return float("nan")
    return float(np.mean(finite))


def _ratio(num: int | float, den: int | float) -> float:
    if den == 0:
        return float("nan")
    return float(num / den)


def _csv_value(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return CSV_NAN
    return value


def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)


def _prepare_gt(coco: dict[str, Any]) -> tuple[dict[int, dict[str, Any]], dict[tuple[int, int], list[dict[str, Any]]]]:
    images = _build_images(coco)
    density = Counter(int(ann["image_id"]) for ann in coco["annotations"] if int(ann.get("iscrowd", 0)) == 0)
    anns_by_key: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    seen_ann_ids: set[int] = set()
    for idx, ann in enumerate(coco["annotations"]):
        if int(ann.get("iscrowd", 0)) != 0:
            continue
        ann_id = int(ann["id"])
        if ann_id in seen_ann_ids:
            raise BuilderError(f"duplicate annotation id={ann_id}")
        seen_ann_ids.add(ann_id)
        image_id = int(ann["image_id"])
        category_id = int(ann["category_id"])
        if image_id not in images:
            raise BuilderError(f"annotation references unknown image_id={image_id}")
        image = images[image_id]
        height = int(image["height"])
        width = int(image["width"])
        item = dict(ann)
        area = float(ann["area"])
        item["_image_id"] = image_id
        item["_category_id"] = category_id
        item["_file_name"] = image.get("file_name", "")
        item["_area"] = area
        item["_area_bucket"] = _area_bucket(area)
        item["_density_count"] = int(density[image_id])
        item["_density_bucket"] = _density_bucket(int(density[image_id]))
        item["_rle"] = _mask_rle(ann["segmentation"], height, width, f"annotations[{idx}]")
        item["_mask"] = _decode_mask(item["_rle"], f"annotations[{idx}]")
        anns_by_key[(image_id, category_id)].append(item)
    for key in anns_by_key:
        anns_by_key[key].sort(key=lambda row: int(row["id"]))
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
            raise BuilderError(f"prediction references unknown image_id={image_id}")
        image = images[image_id]
        height = int(image["height"])
        width = int(image["width"])
        item = dict(pred)
        item["_pred_index"] = idx
        item["_image_id"] = image_id
        item["_category_id"] = category_id
        item["_file_name"] = image.get("file_name", "")
        item["_rle"] = _mask_rle(pred["segmentation"], height, width, f"predictions[{idx}]")
        item["_mask"] = _decode_mask(item["_rle"], f"predictions[{idx}]")
        preds_by_key[(image_id, category_id)].append(item)
    for key in preds_by_key:
        preds_by_key[key].sort(key=lambda row: (-float(row["score"]), int(row["_pred_index"])))
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


def _build_rows(spec: RunSpec) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    coco = _require_coco(_load_json(spec.gt_json), spec.gt_json)
    predictions = _require_predictions(_load_json(spec.pred_json), spec.pred_json)
    images, anns_by_key = _prepare_gt(coco)
    preds_by_key = _prepare_predictions(predictions, images)
    image_density = Counter(int(ann["image_id"]) for ann in coco["annotations"] if int(ann.get("iscrowd", 0)) == 0)

    gt_rows: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []
    for key in sorted(set(anns_by_key) | set(preds_by_key)):
        image_id, category_id = key
        anns = anns_by_key.get(key, [])
        preds = preds_by_key.get(key, [])
        ious = np.zeros((len(preds), len(anns)), dtype=np.float64)
        for pred_idx, pred in enumerate(preds):
            for gt_idx, ann in enumerate(anns):
                ious[pred_idx, gt_idx] = _mask_iou(pred["_mask"], ann["_mask"])
        pred_to_gt75, gt_to_pred75, dup75 = _greedy_match(ious, 0.75)

        gt_best_pred: list[int | None] = [None] * len(anns)
        gt_best_iou = [0.0] * len(anns)
        if preds:
            for gt_idx in range(len(anns)):
                pred_idx = int(np.argmax(ious[:, gt_idx]))
                gt_best_pred[gt_idx] = pred_idx
                gt_best_iou[gt_idx] = float(ious[pred_idx, gt_idx])

        pred_best_gt: list[int | None] = [None] * len(preds)
        pred_best_iou = [0.0] * len(preds)
        if anns:
            for pred_idx in range(len(preds)):
                gt_idx = int(np.argmax(ious[pred_idx, :]))
                pred_best_gt[pred_idx] = gt_idx
                pred_best_iou[pred_idx] = float(ious[pred_idx, gt_idx])

        for gt_idx, ann in enumerate(anns):
            best_pred_idx = gt_best_pred[gt_idx]
            boundary_f = float("nan")
            if best_pred_idx is not None:
                boundary_f = _boundary_f_score(preds[best_pred_idx]["_mask"], ann["_mask"])
            match75 = gt_to_pred75[gt_idx]
            gt_rows.append(
                {
                    "run": spec.name,
                    "role": "held-out",
                    "image_id": image_id,
                    "file_name": ann["_file_name"],
                    "category_id": category_id,
                    "gt_id": int(ann["id"]),
                    "gt_area": ann["_area"],
                    "area_bucket": ann["_area_bucket"],
                    "image_gt_count": ann["_density_count"],
                    "density_bucket": ann["_density_bucket"],
                    "best_pred_index": preds[best_pred_idx]["_pred_index"] if best_pred_idx is not None else "",
                    "best_pred_score": float(preds[best_pred_idx]["score"]) if best_pred_idx is not None else float("nan"),
                    "best_iou": gt_best_iou[gt_idx],
                    "boundary_f_best": boundary_f,
                    "tp75": int(match75 is not None),
                    "fn75": int(match75 is None),
                    "matched_pred_index_75": preds[match75]["_pred_index"] if match75 is not None else "",
                }
            )

        for pred_idx, pred in enumerate(preds):
            best_gt_idx = pred_best_gt[pred_idx]
            gt_ann = anns[best_gt_idx] if best_gt_idx is not None else None
            density_count = int(image_density.get(image_id, 0))
            boundary_f = float("nan")
            if gt_ann is not None:
                boundary_f = _boundary_f_score(pred["_mask"], gt_ann["_mask"])
            pred_rows.append(
                {
                    "run": spec.name,
                    "role": "held-out",
                    "image_id": image_id,
                    "file_name": pred["_file_name"],
                    "category_id": category_id,
                    "pred_index": int(pred["_pred_index"]),
                    "score": float(pred["score"]),
                    "image_gt_count": density_count,
                    "density_bucket": _density_bucket(density_count),
                    "best_gt_id": int(gt_ann["id"]) if gt_ann is not None else "",
                    "best_gt_area": gt_ann["_area"] if gt_ann is not None else float("nan"),
                    "best_gt_area_bucket": gt_ann["_area_bucket"] if gt_ann is not None else "no_gt",
                    "best_gt_iou": pred_best_iou[pred_idx],
                    "boundary_f_best": boundary_f,
                    "tp75": int(pred_to_gt75[pred_idx] is not None),
                    "fp75": int(pred_to_gt75[pred_idx] is None),
                    "duplicate_fp75": int(dup75[pred_idx]),
                    "matched_gt_id_75": int(anns[pred_to_gt75[pred_idx]]["id"]) if pred_to_gt75[pred_idx] is not None else "",
                }
            )

    bucket_rows = _build_bucket_rows(spec, gt_rows, pred_rows)
    summary = {
        "run": spec.name,
        "role": "held-out",
        "gt_json": str(spec.gt_json),
        "pred_json": str(spec.pred_json),
        "images": len(images),
        "gt": len(gt_rows),
        "pred": len(pred_rows),
        "tp75": sum(int(row["tp75"]) for row in gt_rows),
        "fn75": sum(int(row["fn75"]) for row in gt_rows),
        "fp75": sum(int(row["fp75"]) for row in pred_rows),
        "recall75": _ratio(sum(int(row["tp75"]) for row in gt_rows), len(gt_rows)),
        "best_iou_mean": _finite_mean([float(row["best_iou"]) for row in gt_rows]),
        "boundary_f_mean": _finite_mean([float(row["boundary_f_best"]) for row in gt_rows]),
        "buckets": bucket_rows,
    }
    return gt_rows, pred_rows, bucket_rows, summary


def _build_bucket_rows(
    spec: RunSpec,
    gt_rows: list[dict[str, Any]],
    pred_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    bucket_specs = [
        ("area_bucket", "area_bucket", "best_gt_area_bucket"),
        ("density_bucket", "density_bucket", "density_bucket"),
    ]
    for bucket_type, gt_key, pred_key in bucket_specs:
        names = sorted({str(row[gt_key]) for row in gt_rows} | {str(row[pred_key]) for row in pred_rows})
        for name in names:
            if name == "no_gt":
                continue
            gt_bucket = [row for row in gt_rows if row[gt_key] == name]
            pred_bucket = [row for row in pred_rows if row[pred_key] == name]
            tp75_pred = sum(int(row["tp75"]) for row in pred_bucket)
            fp75 = sum(int(row["fp75"]) for row in pred_bucket)
            gt_count = len(gt_bucket)
            pred_count = len(pred_bucket)
            rows.append(
                {
                    "run": spec.name,
                    "role": "held-out",
                    "bucket_type": bucket_type,
                    "bucket": name,
                    "images": len({int(row["image_id"]) for row in gt_bucket}),
                    "gt": gt_count,
                    "pred": pred_count,
                    "gt_count": gt_count,
                    "pred_count": pred_count,
                    "tp75": tp75_pred,
                    "fn75": sum(int(row["fn75"]) for row in gt_bucket),
                    "fp75": fp75,
                    "precision75": _ratio(tp75_pred, tp75_pred + fp75),
                    "recall75": _ratio(sum(int(row["tp75"]) for row in gt_bucket), gt_count),
                    "duplicate_fp75": sum(int(row["duplicate_fp75"]) for row in pred_bucket),
                    "best_iou_mean": _finite_mean([float(row["best_iou"]) for row in gt_bucket]),
                    "boundary_f_mean": _finite_mean([float(row["boundary_f_best"]) for row in gt_bucket]),
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run",
        "role",
        "bucket_type",
        "bucket",
        "images",
        "gt",
        "pred",
        "gt_count",
        "pred_count",
        "tp75",
        "fn75",
        "fp75",
        "precision75",
        "recall75",
        "duplicate_fp75",
        "best_iou_mean",
        "boundary_f_mean",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in fieldnames})


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    spec = RunSpec(args.run_name, Path(args.gt_json), Path(args.pred_json))
    _, _, bucket_rows, summary = _build_rows(spec)
    _write_csv(Path(args.output_csv), bucket_rows)
    if args.output_json:
        _write_json(Path(args.output_json), summary)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-json", default=DEFAULT_GT_JSON)
    parser.add_argument("--pred-json", default=DEFAULT_PRED_JSON)
    parser.add_argument("--run-name", default="r122_remaining75")
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run(args)
    except BuilderError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
