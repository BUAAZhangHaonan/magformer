#!/usr/bin/env python3
"""Audit dense-image geometry and prediction truncation from COCO JSON files."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from pycocotools import mask as mask_utils


BUCKETS = (
    ("0-30", 0, 30),
    ("31-60", 31, 60),
    ("61-89", 61, 89),
    ("90-100", 90, 100),
    ("101+", 101, math.inf),
)


TOPK_CODE_LOCATIONS = [
    {
        "path": "magformer/models/magformer/arch.py",
        "setting": "forward_inference_raw topk=min(100, Nq*num_classes)",
    },
    {
        "path": "magformer/engine/evaluator.py",
        "setting": "COCOEvaluator max_dets=100, COCOeval params.maxDets=[1,10,100]",
    },
    {
        "path": "magformer/engine/eval_runtime.py",
        "setting": "run_inference_evaluation score_threshold=0.05, max_dets=100",
    },
    {
        "path": "tools/evaluate_1024_backmap.py",
        "setting": "backmap export score_threshold, mask_threshold, max_dets=100",
    },
]


class AuditError(RuntimeError):
    """Raised when audit inputs are missing or malformed."""


def _load_json(path: str | Path) -> Any:
    resolved = Path(path)
    if not resolved.exists():
        raise AuditError(f"Input file not found: {resolved}")
    with resolved.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _bucket_name(count: int) -> str:
    for name, low, high in BUCKETS:
        if low <= count <= high:
            return name
    raise AssertionError(f"unreachable bucket for count={count}")


def _summary(values: Iterable[float]) -> dict[str, float | int | None]:
    arr = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=np.float64)
    if arr.size == 0:
        return {"count": 0, "min": None, "p25": None, "mean": None, "median": None, "p75": None, "p90": None, "max": None}
    return {
        "count": int(arr.size),
        "min": float(arr.min()),
        "p25": float(np.percentile(arr, 25)),
        "mean": float(arr.mean()),
        "median": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(arr.max()),
    }


def _bbox_iou(box_a: Iterable[float], box_b: Iterable[float]) -> float:
    ax, ay, aw, ah = [float(v) for v in box_a]
    bx, by, bw, bh = [float(v) for v in box_b]
    ax2, ay2 = ax + max(0.0, aw), ay + max(0.0, ah)
    bx2, by2 = bx + max(0.0, bw), by + max(0.0, bh)
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = max(0.0, aw) * max(0.0, ah) + max(0.0, bw) * max(0.0, bh) - inter
    return float(inter / union) if union > 0 else 0.0


def _mask_area(segmentation: dict[str, Any] | list[Any], height: int, width: int) -> float:
    rle = _segmentation_to_rle(segmentation, height, width)
    return float(mask_utils.area(rle))


def _segmentation_to_rle(segmentation: dict[str, Any] | list[Any], height: int, width: int) -> dict[str, Any]:
    if isinstance(segmentation, list):
        rles = mask_utils.frPyObjects(segmentation, height, width)
        return mask_utils.merge(rles)
    if isinstance(segmentation, dict) and "counts" in segmentation:
        return segmentation
    raise AuditError("Segmentation must be COCO polygon or RLE.")


def _mask_iou(seg_a: dict[str, Any] | list[Any], seg_b: dict[str, Any] | list[Any], height: int, width: int) -> float:
    rle_a = _segmentation_to_rle(seg_a, height, width)
    rle_b = _segmentation_to_rle(seg_b, height, width)
    return float(mask_utils.iou([rle_a], [rle_b], [0])[0][0])


def _require_bbox(row: dict[str, Any], context: str) -> list[float]:
    bbox = row.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise AuditError(f"{context} missing COCO bbox [x,y,w,h].")
    return [float(v) for v in bbox]


def _require_segmentation(row: dict[str, Any], context: str) -> dict[str, Any] | list[Any]:
    if "segmentation" in row:
        return row["segmentation"]
    if "mask" in row:
        return row["mask"]
    raise AuditError(f"{context} missing segmentation/mask for mask audit.")


def _index_gt(coco: dict[str, Any]) -> tuple[dict[int, dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    images = {int(image["id"]): image for image in coco.get("images", [])}
    if not images:
        raise AuditError("GT COCO JSON has no images.")
    anns_by_image: dict[int, list[dict[str, Any]]] = {image_id: [] for image_id in images}
    for ann in coco.get("annotations", []):
        if int(ann.get("iscrowd", 0)) != 0:
            continue
        image_id = int(ann["image_id"])
        if image_id in anns_by_image:
            anns_by_image[image_id].append(ann)
    return images, anns_by_image


def _index_predictions(predictions: list[dict[str, Any]], image_ids: Iterable[int]) -> dict[int, list[dict[str, Any]]]:
    preds_by_image: dict[int, list[dict[str, Any]]] = {int(image_id): [] for image_id in image_ids}
    for pred in predictions:
        image_id = int(pred["image_id"])
        if image_id in preds_by_image:
            preds_by_image[image_id].append(pred)
    for preds in preds_by_image.values():
        preds.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    return preds_by_image


def _match_one_type(
    *,
    gt_anns: list[dict[str, Any]],
    preds: list[dict[str, Any]],
    image_info: dict[str, Any],
    iou_type: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    matched_gt: set[int] = set()
    tp_ious: list[float] = []
    fp_rows: list[dict[str, Any]] = []
    height = int(image_info["height"])
    width = int(image_info["width"])

    for pred_idx, pred in enumerate(preds):
        best_iou = 0.0
        best_gt_idx = None
        for gt_idx, gt in enumerate(gt_anns):
            if iou_type == "bbox":
                iou = _bbox_iou(_require_bbox(pred, "prediction"), _require_bbox(gt, "annotation"))
            else:
                iou = _mask_iou(
                    _require_segmentation(pred, "prediction"),
                    _require_segmentation(gt, "annotation"),
                    height,
                    width,
                )
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_gt_idx is not None and best_iou >= 0.5 and best_gt_idx not in matched_gt:
            matched_gt.add(best_gt_idx)
            tp_ious.append(best_iou)
            fp_type = "tp"
        elif best_gt_idx is not None and best_iou >= 0.5:
            fp_type = "duplicate_fp"
        elif best_iou > 0.0:
            fp_type = "low_iou_fp"
        else:
            fp_type = "background_fp"

        if fp_type != "tp":
            fp_rows.append(
                {
                    "pred_index": pred_idx,
                    "score": float(pred.get("score", 0.0)),
                    "type": fp_type,
                    "max_iou": float(best_iou),
                    "best_gt_index": best_gt_idx,
                }
            )

    counts = {
        "tp": len(tp_ious),
        "fp": len(preds) - len(tp_ious),
        "fn": len(gt_anns) - len(matched_gt),
        "duplicate_fp": sum(1 for row in fp_rows if row["type"] == "duplicate_fp"),
        "low_iou_fp": sum(1 for row in fp_rows if row["type"] == "low_iou_fp"),
        "background_fp": sum(1 for row in fp_rows if row["type"] == "background_fp"),
        "tp_mean_iou": float(np.mean(tp_ious)) if tp_ious else None,
        "tp_ious": tp_ious,
    }
    return counts, fp_rows


def _matched_geometry(
    gt_anns: list[dict[str, Any]],
    preds: list[dict[str, Any]],
    image_info: dict[str, Any],
) -> tuple[list[float], list[float], list[float], list[float], list[float]]:
    matched_gt: set[int] = set()
    bbox_ious: list[float] = []
    mask_ious: list[float] = []
    scale_ratios: list[float] = []
    mask_area_ratios: list[float] = []
    low_iou_fp_max_ious: list[float] = []
    height = int(image_info["height"])
    width = int(image_info["width"])
    for pred in preds:
        best_iou = 0.0
        best_gt_idx = None
        for gt_idx, gt in enumerate(gt_anns):
            iou = _mask_iou(
                _require_segmentation(pred, "prediction"),
                _require_segmentation(gt, "annotation"),
                height,
                width,
            )
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        if best_gt_idx is not None and best_iou >= 0.5 and best_gt_idx not in matched_gt:
            matched_gt.add(best_gt_idx)
            gt = gt_anns[best_gt_idx]
            pred_bbox = _require_bbox(pred, "prediction")
            gt_bbox = _require_bbox(gt, "annotation")
            pred_area = max(0.0, pred_bbox[2]) * max(0.0, pred_bbox[3])
            gt_area = max(0.0, gt_bbox[2]) * max(0.0, gt_bbox[3])
            pred_mask_area = _mask_area(_require_segmentation(pred, "prediction"), height, width)
            gt_mask_area = _mask_area(_require_segmentation(gt, "annotation"), height, width)
            bbox_ious.append(_bbox_iou(pred_bbox, gt_bbox))
            mask_ious.append(best_iou)
            if gt_area > 0:
                scale_ratios.append(pred_area / gt_area)
            if gt_mask_area > 0:
                mask_area_ratios.append(pred_mask_area / gt_mask_area)
        elif 0.0 < best_iou < 0.5:
            low_iou_fp_max_ious.append(best_iou)
    return bbox_ious, mask_ious, scale_ratios, mask_area_ratios, low_iou_fp_max_ious


def _bbox_geometry(rows: Iterable[dict[str, Any]]) -> tuple[list[float], list[float], list[float]]:
    widths: list[float] = []
    heights: list[float] = []
    areas: list[float] = []
    for row in rows:
        bbox = _require_bbox(row, "geometry row")
        widths.append(float(bbox[2]))
        heights.append(float(bbox[3]))
        areas.append(float(max(0.0, bbox[2]) * max(0.0, bbox[3])))
    return widths, heights, areas


def _build_markdown(summary: dict[str, Any]) -> str:
    lines = [
        f"# R13 Dense Geometry Audit: {summary['name']}",
        "",
        f"- gt: `{summary['inputs']['gt']}`",
        f"- predictions: `{summary['inputs']['predictions']}`",
        f"- images: {summary['totals']['images']}",
        f"- predictions: {summary['totals']['predictions']}",
        f"- mask TP/FP/FN: {summary['totals']['mask']['tp']}/{summary['totals']['mask']['fp']}/{summary['totals']['mask']['fn']}",
        f"- bbox TP/FP/FN: {summary['totals']['bbox']['tp']}/{summary['totals']['bbox']['fp']}/{summary['totals']['bbox']['fn']}",
        f"- postprocess-pretopk: {summary['max_dets']['postprocess_pretopk']}",
        "",
        "## Dense Buckets",
        "",
        "| bucket | images | gt | pred | mask_tp | low_iou_fp | duplicate_fp | background_fp | fn |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for bucket, row in summary["buckets"].items():
        mask = row["mask"]
        lines.append(
            f"| {bucket} | {row['images']} | {row['gt']} | {row['pred']} | {mask['tp']} | "
            f"{mask['low_iou_fp']} | {mask['duplicate_fp']} | {mask['background_fp']} | {mask['fn']} |"
        )
    lines.extend(["", "## Top Worst Images", ""])
    for row in summary["top_worst_images"]:
        lines.append(
            f"- image_id={row['image_id']} gt={row['gt_count']} pred={row['pred_count']} "
            f"mask_fp={row['mask']['fp']} mask_fn={row['mask']['fn']} worst_score={row['worst_score']:.3f}"
        )
    if summary.get("comparison_worst_overlap"):
        lines.extend(["", "## Worst-Image Overlap", ""])
        for name, overlap in summary["comparison_worst_overlap"].items():
            lines.append(f"- {name}: {overlap}")
    lines.extend(["", "## TopK / maxDets Locations", ""])
    for item in summary["max_dets"]["code_locations"]:
        lines.append(f"- `{item['path']}`: {item['setting']}")
    lines.append("")
    return "\n".join(lines)


def _write_outputs(summary: dict[str, Any], output_dir: str | Path, name: str) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}_audit.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (out_dir / f"{name}_audit.md").write_text(_build_markdown(summary), encoding="utf-8")


def _audit_in_memory(
    gt_coco: dict[str, Any],
    predictions: list[dict[str, Any]],
    *,
    name: str,
    gt_path: str,
    pred_path: str,
    top_k: int,
) -> dict[str, Any]:
    images, anns_by_image = _index_gt(gt_coco)
    preds_by_image = _index_predictions(predictions, images.keys())

    image_rows: dict[str, Any] = {}
    bucket_rows: dict[str, Any] = {
        name_: {
            "images": 0,
            "gt": 0,
            "pred": 0,
            "mask": {"tp": 0, "fp": 0, "fn": 0, "duplicate_fp": 0, "low_iou_fp": 0, "background_fp": 0},
            "bbox": {"tp": 0, "fp": 0, "fn": 0, "duplicate_fp": 0, "low_iou_fp": 0, "background_fp": 0},
        }
        for name_, _, _ in BUCKETS
    }
    totals = {
        "images": len(images),
        "gt": 0,
        "predictions": len(predictions),
        "mask": {"tp": 0, "fp": 0, "fn": 0, "duplicate_fp": 0, "low_iou_fp": 0, "background_fp": 0},
        "bbox": {"tp": 0, "fp": 0, "fn": 0, "duplicate_fp": 0, "low_iou_fp": 0, "background_fp": 0},
    }
    score_values = [float(pred.get("score", 0.0)) for pred in predictions]
    gt_widths, gt_heights, gt_bbox_areas = _bbox_geometry(gt_coco.get("annotations", []))
    pred_widths, pred_heights, pred_bbox_areas = _bbox_geometry(predictions)
    gt_mask_areas: list[float] = []
    pred_mask_areas: list[float] = []
    matched_bbox_ious: list[float] = []
    matched_mask_ious: list[float] = []
    scale_ratios: list[float] = []
    mask_area_ratios: list[float] = []
    low_iou_fp_max_ious: list[float] = []

    for image_id, image_info in images.items():
        gt_anns = anns_by_image[image_id]
        preds = preds_by_image[image_id]
        mask_counts, mask_fps = _match_one_type(gt_anns=gt_anns, preds=preds, image_info=image_info, iou_type="mask")
        bbox_counts, _bbox_fps = _match_one_type(gt_anns=gt_anns, preds=preds, image_info=image_info, iou_type="bbox")
        bbox_ious, mask_ious, ratios, area_ratios, low_ious = _matched_geometry(gt_anns, preds, image_info)
        matched_bbox_ious.extend(bbox_ious)
        matched_mask_ious.extend(mask_ious)
        scale_ratios.extend(ratios)
        mask_area_ratios.extend(area_ratios)
        low_iou_fp_max_ious.extend(low_ious)
        height = int(image_info["height"])
        width = int(image_info["width"])
        gt_mask_areas.extend(
            _mask_area(_require_segmentation(ann, "annotation"), height, width) for ann in gt_anns
        )
        pred_mask_areas.extend(
            _mask_area(_require_segmentation(pred, "prediction"), height, width) for pred in preds
        )

        image_row = {
            "image_id": image_id,
            "file_name": image_info.get("file_name"),
            "gt_count": len(gt_anns),
            "pred_count": len(preds),
            "score": _summary(float(pred.get("score", 0.0)) for pred in preds),
            "mask": {key: value for key, value in mask_counts.items() if key != "tp_ious"},
            "bbox": {key: value for key, value in bbox_counts.items() if key != "tp_ious"},
            "mask_fp_examples": mask_fps[:10],
        }
        image_row["worst_score"] = (
            float(image_row["mask"]["fp"])
            + float(image_row["mask"]["fn"]) * 2.0
            + float(image_row["gt_count"]) * 0.01
            - float(image_row["mask"]["tp"]) * 0.1
        )
        image_rows[str(image_id)] = image_row
        bucket = _bucket_name(len(gt_anns))
        bucket_row = bucket_rows[bucket]
        bucket_row["images"] += 1
        bucket_row["gt"] += len(gt_anns)
        bucket_row["pred"] += len(preds)
        totals["gt"] += len(gt_anns)
        for metric in ("mask", "bbox"):
            for key in ("tp", "fp", "fn", "duplicate_fp", "low_iou_fp", "background_fp"):
                bucket_row[metric][key] += int(image_row[metric][key])
                totals[metric][key] += int(image_row[metric][key])

    pred_counts = [len(preds_by_image[image_id]) for image_id in images]
    top_worst = sorted(image_rows.values(), key=lambda item: item["worst_score"], reverse=True)[:top_k]
    return {
        "name": name,
        "inputs": {"gt": gt_path, "predictions": pred_path},
        "totals": totals,
        "score_distribution": _summary(score_values),
        "images": image_rows,
        "buckets": bucket_rows,
        "top_worst_images": top_worst,
        "geometry": {
            "gt_bbox_width": _summary(gt_widths),
            "gt_bbox_height": _summary(gt_heights),
            "gt_bbox_area": _summary(gt_bbox_areas),
            "gt_mask_area": _summary(gt_mask_areas),
            "pred_bbox_width": _summary(pred_widths),
            "pred_bbox_height": _summary(pred_heights),
            "pred_bbox_area": _summary(pred_bbox_areas),
            "pred_mask_area": _summary(pred_mask_areas),
            "matched_tp_bbox_iou": _summary(matched_bbox_ious),
            "matched_tp_mask_iou": _summary(matched_mask_ious),
            "matched_tp_pred_gt_bbox_area_ratio": _summary(scale_ratios),
            "matched_tp_pred_gt_mask_area_ratio": _summary(mask_area_ratios),
            "low_iou_fp_max_iou": _summary(low_iou_fp_max_ious),
        },
        "max_dets": {
            "cocoeval_max_dets": [1, 10, 100],
            "predictions_per_image": _summary(pred_counts),
            "images_at_or_above_100_predictions": [int(image_id) for image_id, preds in preds_by_image.items() if len(preds) >= 100],
            "possible_json_or_cocoeval_truncation": any(len(preds) >= 100 for preds in preds_by_image.values()),
            "postprocess_pretopk": "postprocess-pretopk unavailable",
            "code_locations": TOPK_CODE_LOCATIONS,
        },
    }


def run_audit(
    gt_json: str | Path,
    predictions_json: str | Path,
    *,
    name: str,
    output_dir: str | Path,
    top_k: int = 10,
    comparison_worst: dict[str, list[int]] | None = None,
) -> dict[str, Any]:
    gt_coco = _load_json(gt_json)
    predictions = _load_json(predictions_json)
    if not isinstance(predictions, list):
        raise AuditError("Predictions JSON must be a COCO result list.")
    summary = _audit_in_memory(
        gt_coco,
        predictions,
        name=name,
        gt_path=str(gt_json),
        pred_path=str(predictions_json),
        top_k=top_k,
    )
    if comparison_worst:
        own = {int(row["image_id"]) for row in summary["top_worst_images"]}
        summary["comparison_worst_overlap"] = {
            label: sorted(own & {int(image_id) for image_id in image_ids})
            for label, image_ids in comparison_worst.items()
        }
    _write_outputs(summary, output_dir, name)
    return summary


def _parse_comparison(values: list[str]) -> dict[str, list[int]]:
    parsed: dict[str, list[int]] = {}
    for value in values:
        if "=" not in value:
            raise AuditError(f"Comparison must use NAME=path syntax: {value}")
        name, path = value.split("=", 1)
        report = _load_json(path)
        parsed[name] = [int(row["image_id"]) for row in report.get("top_worst_images", [])]
    return parsed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", required=True, help="COCO GT annotation JSON.")
    parser.add_argument("--predictions", required=True, help="COCO predictions/result JSON.")
    parser.add_argument("--name", required=True, help="Output report prefix.")
    parser.add_argument("--output-dir", default="output/diagnostics/r13_dense_geometry_audit")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--compare-worst",
        action="append",
        default=[],
        help="Existing audit report comparison in NAME=report.json form.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    comparisons = _parse_comparison(args.compare_worst)
    run_audit(
        args.gt,
        args.predictions,
        name=args.name,
        output_dir=args.output_dir,
        top_k=args.top_k,
        comparison_worst=comparisons,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
