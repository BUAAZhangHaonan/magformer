#!/usr/bin/env python3
"""Diagnose R119 protocol stability from existing COCO GT and predictions."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


IOU_THRESHOLDS = np.arange(0.50, 0.96, 0.05)
METRIC_KEYS = ("AP", "AP50", "AP75")


DEFAULT_RUNS = {
    "r114_val28": {
        "ann": "magformer_datasets/pseudo_real_512/annotations/instances_val.json",
        "pred": "output/diagnostics/r114_magformer_r113warm_target150_iter2000_val28_1024_backmap_topk200_20260518/coco_instances_results.json",
        "metrics": "output/diagnostics/r114_magformer_r113warm_target150_iter2000_val28_1024_backmap_topk200_20260518/metrics.cocoeval.json",
    },
    "r114_remaining75": {
        "ann": "magformer_datasets/pseudo_real_512/annotations/instances_target_unlabeled_r114_balanced_minus125.json",
        "pred": "output/diagnostics/r114_magformer_r113warm_target150_iter2000_remaining75_1024_backmap_topk200_20260518/coco_instances_results.json",
        "metrics": "output/diagnostics/r114_magformer_r113warm_target150_iter2000_remaining75_1024_backmap_topk200_20260518/metrics.cocoeval.json",
    },
    "r114_full200_reference": {
        "ann": "magformer_datasets/pseudo_real_512/annotations/instances_target_unlabeled.json",
        "pred": "output/diagnostics/r114_magformer_r113warm_target150_iter2000_full200_reference_1024_backmap_topk200_20260518/coco_instances_results.json",
        "metrics": "output/diagnostics/r114_magformer_r113warm_target150_iter2000_full200_reference_1024_backmap_topk200_20260518/metrics.cocoeval.json",
    },
    "r115_train200_oracle": {
        "ann": "magformer_datasets/pseudo_real_512/annotations/instances_target_unlabeled.json",
        "pred": "output/diagnostics/r115b_magformer_r114warm_fulltarget200_oracle_iter2000_train200_1024_backmap_topk200_20260518/coco_instances_results.json",
        "metrics": "output/diagnostics/r115b_magformer_r114warm_fulltarget200_oracle_iter2000_train200_1024_backmap_topk200_20260518/metrics.cocoeval.json",
    },
    "r118_val28_smallstep": {
        "ann": "magformer_datasets/pseudo_real_512/annotations/instances_val.json",
        "pred": "output/diagnostics/r118_iter0099_val28_1024_backmap_topk200_20260518/coco_instances_results.json",
        "metrics": "output/diagnostics/r118_iter0099_val28_1024_backmap_topk200_20260518/metrics.cocoeval.json",
    },
    "r118_remaining75_smallstep": {
        "ann": "magformer_datasets/pseudo_real_512/annotations/instances_target_unlabeled_r114_balanced_minus125.json",
        "pred": "output/diagnostics/r118_iter0099_remaining75_1024_backmap_topk200_20260518/coco_instances_results.json",
        "metrics": "output/diagnostics/r118_iter0099_remaining75_1024_backmap_topk200_20260518/metrics.cocoeval.json",
    },
    "r118_full200_smallstep": {
        "ann": "magformer_datasets/pseudo_real_512/annotations/instances_target_unlabeled.json",
        "pred": "output/diagnostics/r118_iter0099_full200_reference_1024_backmap_topk200_20260518/coco_instances_results.json",
        "metrics": "output/diagnostics/r118_iter0099_full200_reference_1024_backmap_topk200_20260518/metrics.cocoeval.json",
    },
}


class DiagnosisError(RuntimeError):
    """Raised when a diagnostic input is missing or malformed."""


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
        for key in ("image_id", "category_id", "bbox", "segmentation", "score"):
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


def _ann_area(ann: dict[str, Any], images: dict[int, dict[str, Any]]) -> float:
    if ann.get("area") is not None:
        return float(ann["area"])
    image = images[int(ann["image_id"])]
    return float(mask_utils.area(_mask_rle(ann["segmentation"], int(image["height"]), int(image["width"]))))


def _mean_coco_metrics(eval_obj: COCOeval) -> dict[str, float]:
    precision = eval_obj.eval["precision"]
    area_idx = list(eval_obj.params.areaRngLbl).index("all")
    max_det_idx = list(eval_obj.params.maxDets).index(200)

    def mean_precision(iou_thr: float | None = None) -> float:
        if iou_thr is None:
            values = precision[:, :, :, area_idx, max_det_idx]
        else:
            idx = np.where(np.isclose(eval_obj.params.iouThrs, iou_thr))[0]
            values = precision[idx, :, :, area_idx, max_det_idx]
        values = values[values > -1]
        return float(np.mean(values)) if values.size else -1.0

    return {
        "AP": mean_precision(),
        "AP50": mean_precision(0.50),
        "AP75": mean_precision(0.75),
    }


def _eval_subset(
    coco: dict[str, Any],
    predictions: list[dict[str, Any]],
    *,
    image_ids: set[int],
    positive_ann_ids: set[int],
    iou_type: str,
) -> dict[str, float]:
    if not image_ids:
        return {"AP": float("nan"), "AP50": float("nan"), "AP75": float("nan")}
    if not positive_ann_ids:
        return {"AP": float("nan"), "AP50": float("nan"), "AP75": float("nan")}

    subset_images = [copy.deepcopy(image) for image in coco["images"] if int(image["id"]) in image_ids]
    subset_anns = []
    for ann in coco["annotations"]:
        image_id = int(ann["image_id"])
        if image_id not in image_ids:
            continue
        item = copy.deepcopy(ann)
        if int(item["id"]) in positive_ann_ids:
            item["ignore"] = 0
            item["iscrowd"] = int(item.get("iscrowd", 0))
        else:
            item["ignore"] = 1
            item["iscrowd"] = 1
        subset_anns.append(item)
    subset_predictions = [pred for pred in predictions if int(pred["image_id"]) in image_ids]

    payload = {
        "images": subset_images,
        "annotations": subset_anns,
        "categories": copy.deepcopy(coco["categories"]),
        "info": copy.deepcopy(coco.get("info", {})),
        "licenses": copy.deepcopy(coco.get("licenses", [])),
    }
    with tempfile.TemporaryDirectory(prefix="r119_coco_") as tmpdir:
        ann_path = Path(tmpdir) / "ann.json"
        pred_path = Path(tmpdir) / "pred.json"
        ann_path.write_text(json.dumps(payload), encoding="utf-8")
        pred_path.write_text(json.dumps(subset_predictions), encoding="utf-8")
        coco_gt = COCO(str(ann_path))
        coco_dt = coco_gt.loadRes(str(pred_path))
        eval_obj = COCOeval(coco_gt, coco_dt, iou_type)
        eval_obj.params.maxDets = [1, 10, 200]
        eval_obj.evaluate()
        eval_obj.accumulate()
    return _mean_coco_metrics(eval_obj)


def _ap_from_sorted_matches(matches: np.ndarray, scores_count: int, gt_count: int) -> float:
    if gt_count <= 0:
        return float("nan")
    if scores_count <= 0:
        return 0.0
    false_positives = 1.0 - matches
    tp = np.cumsum(matches)
    fp = np.cumsum(false_positives)
    recall = tp / gt_count
    precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for idx in range(mpre.size - 1, 0, -1):
        mpre[idx - 1] = max(mpre[idx - 1], mpre[idx])
    recall_grid = np.linspace(0.0, 1.0, 101)
    values = [mpre[np.where(mrec >= recall_point)[0][0]] for recall_point in recall_grid]
    return float(np.mean(values))


def _per_image_proxy(
    coco: dict[str, Any],
    predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    images = {int(image["id"]): image for image in coco["images"]}
    anns_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    preds_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in coco["annotations"]:
        if int(ann.get("iscrowd", 0)) == 0:
            anns_by_image[int(ann["image_id"])].append(ann)
    for pred in predictions:
        preds_by_image[int(pred["image_id"])].append(pred)

    rows = []
    for image_id, image in sorted(images.items()):
        anns = anns_by_image.get(image_id, [])
        preds = sorted(preds_by_image.get(image_id, []), key=lambda item: float(item["score"]), reverse=True)
        height = int(image["height"])
        width = int(image["width"])
        gt_rles = [_mask_rle(ann["segmentation"], height, width) for ann in anns]
        pred_rles = [_mask_rle(pred["segmentation"], height, width) for pred in preds]
        if not anns:
            ap_by_thr = [float("nan")] * len(IOU_THRESHOLDS)
        elif not preds:
            ap_by_thr = [0.0] * len(IOU_THRESHOLDS)
        else:
            ious = mask_utils.iou(pred_rles, gt_rles, [int(ann.get("iscrowd", 0)) for ann in anns])
            ap_by_thr = []
            for threshold in IOU_THRESHOLDS:
                matched_gt: set[int] = set()
                matches = np.zeros((len(preds),), dtype=np.float64)
                for pred_idx in range(len(preds)):
                    best_gt = -1
                    best_iou = float(threshold)
                    for gt_idx in range(len(anns)):
                        if gt_idx in matched_gt:
                            continue
                        iou = float(ious[pred_idx, gt_idx])
                        if iou >= best_iou:
                            best_iou = iou
                            best_gt = gt_idx
                    if best_gt >= 0:
                        matched_gt.add(best_gt)
                        matches[pred_idx] = 1.0
                ap_by_thr.append(_ap_from_sorted_matches(matches, len(preds), len(anns)))
        ann_areas = [_ann_area(ann, images) for ann in anns]
        rows.append(
            {
                "image_id": image_id,
                "file_name": image.get("file_name", ""),
                "annotations": len(anns),
                "predictions": len(preds),
                "mean_gt_area": float(np.mean(ann_areas)) if ann_areas else float("nan"),
                "median_gt_area": float(np.median(ann_areas)) if ann_areas else float("nan"),
                "ap_proxy": float(np.nanmean(ap_by_thr)),
                "ap50_proxy": float(ap_by_thr[0]),
                "ap75_proxy": float(ap_by_thr[5]),
            }
        )
    return rows


def _bootstrap(values: list[float], *, samples: int, seed: int) -> dict[str, float]:
    arr = np.asarray([value for value in values if not math.isnan(value)], dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "std": float("nan"), "samples": samples}
    rng = np.random.default_rng(seed)
    means = np.empty((samples,), dtype=np.float64)
    for idx in range(samples):
        draw = rng.choice(arr, size=arr.size, replace=True)
        means[idx] = float(np.mean(draw))
    return {
        "mean": float(np.mean(arr)),
        "ci_low": float(np.percentile(means, 2.5)),
        "ci_high": float(np.percentile(means, 97.5)),
        "std": float(np.std(means, ddof=1)) if samples > 1 else 0.0,
        "samples": samples,
    }


def _pearson(x_values: list[float], y_values: list[float]) -> float:
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]
    if x.size < 2 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _linear_slope(x_values: list[float], y_values: list[float]) -> float:
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]
    if x.size < 2 or float(np.var(x)) == 0.0:
        return float("nan")
    slope, _intercept = np.polyfit(x, y, 1)
    return float(slope)


def _density_name(count: int) -> str:
    if count <= 25:
        return "low25"
    if count <= 50:
        return "mid50"
    return "high100"


def _summarize_distribution(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"min": float("nan"), "p25": float("nan"), "median": float("nan"), "p75": float("nan"), "max": float("nan"), "mean": float("nan")}
    return {
        "min": float(np.min(arr)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def _collect_buckets(
    coco: dict[str, Any],
    per_image: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    images = {int(image["id"]): image for image in coco["images"]}
    anns = [ann for ann in coco["annotations"] if int(ann.get("iscrowd", 0)) == 0]
    anns_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in anns:
        anns_by_image[int(ann["image_id"])].append(ann)
    ann_area = {int(ann["id"]): _ann_area(ann, images) for ann in anns}

    density_defs = []
    for bucket in ("low25", "mid50", "high100"):
        image_ids = {row["image_id"] for row in per_image if _density_name(int(row["annotations"])) == bucket}
        ann_ids = {int(ann["id"]) for image_id in image_ids for ann in anns_by_image.get(int(image_id), [])}
        density_defs.append({"name": bucket, "image_ids": image_ids, "positive_ann_ids": ann_ids})

    sorted_areas = sorted(ann_area.values())
    q20 = float(np.percentile(sorted_areas, 20))
    q50 = float(np.percentile(sorted_areas, 50))
    q80 = float(np.percentile(sorted_areas, 80))

    area_defs = [
        {"name": "tiny_area_le_256", "positive_ann_ids": {ann_id for ann_id, area in ann_area.items() if area <= 256.0}},
        {"name": "bottom20_area", "positive_ann_ids": {ann_id for ann_id, area in ann_area.items() if area <= q20}},
        {"name": "mid60_area", "positive_ann_ids": {ann_id for ann_id, area in ann_area.items() if q20 < area <= q80}},
        {"name": "top20_area", "positive_ann_ids": {ann_id for ann_id, area in ann_area.items() if area > q80}},
    ]
    for bucket in area_defs:
        bucket["image_ids"] = {int(ann["image_id"]) for ann in anns if int(ann["id"]) in bucket["positive_ann_ids"]}

    metadata = {
        "annotation_area": _summarize_distribution(list(ann_area.values())),
        "area_quantiles": {"q20": q20, "q50": q50, "q80": q80},
    }
    return density_defs, area_defs, metadata


def _bucket_eval(
    coco: dict[str, Any],
    predictions: list[dict[str, Any]],
    bucket_defs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for bucket in bucket_defs:
        image_ids = set(bucket["image_ids"])
        ann_ids = set(bucket["positive_ann_ids"])
        bbox = _eval_subset(coco, predictions, image_ids=image_ids, positive_ann_ids=ann_ids, iou_type="bbox")
        segm = _eval_subset(coco, predictions, image_ids=image_ids, positive_ann_ids=ann_ids, iou_type="segm")
        rows.append(
            {
                "bucket": bucket["name"],
                "images": len(image_ids),
                "annotations": len(ann_ids),
                "bbox": bbox,
                "segm": segm,
            }
        )
    return rows


def _expected_metric_check(actual: dict[str, Any], metrics_path: Path, tolerance: float) -> dict[str, float]:
    expected = _load_json(metrics_path)
    checks = {
        "bbox_AP": actual["overall"]["bbox"]["AP"],
        "bbox_AP50": actual["overall"]["bbox"]["AP50"],
        "bbox_AP75": actual["overall"]["bbox"]["AP75"],
        "segm_AP": actual["overall"]["segm"]["AP"],
        "segm_AP50": actual["overall"]["segm"]["AP50"],
        "segm_AP75": actual["overall"]["segm"]["AP75"],
    }
    diffs = {}
    for key, value in checks.items():
        diff = abs(float(expected[key]) - float(value))
        if diff > tolerance:
            raise DiagnosisError(f"{metrics_path} mismatch for {key}: expected {expected[key]}, got {value}")
        diffs[key] = diff
    return diffs


def _diagnose_one(label: str, config: dict[str, str], *, args: argparse.Namespace) -> dict[str, Any]:
    ann_path = Path(config["ann"])
    pred_path = Path(config["pred"])
    metrics_path = Path(config["metrics"])
    coco = _require_coco(_load_json(ann_path), ann_path)
    predictions = _require_predictions(_load_json(pred_path), pred_path)
    image_ids = {int(image["id"]) for image in coco["images"]}
    ann_ids = {int(ann["id"]) for ann in coco["annotations"] if int(ann.get("iscrowd", 0)) == 0}

    bbox = _eval_subset(coco, predictions, image_ids=image_ids, positive_ann_ids=ann_ids, iou_type="bbox")
    segm = _eval_subset(coco, predictions, image_ids=image_ids, positive_ann_ids=ann_ids, iou_type="segm")
    per_image = _per_image_proxy(coco, predictions)
    density_defs, area_defs, bucket_metadata = _collect_buckets(coco, per_image)
    density_rows = _bucket_eval(coco, predictions, density_defs)
    area_rows = _bucket_eval(coco, predictions, area_defs)

    ann_counts = [float(row["annotations"]) for row in per_image]
    ap_values = [float(row["ap_proxy"]) for row in per_image]
    density_distribution = defaultdict(lambda: {"images": 0, "annotations": 0})
    for row in per_image:
        bucket = _density_name(int(row["annotations"]))
        density_distribution[bucket]["images"] += 1
        density_distribution[bucket]["annotations"] += int(row["annotations"])

    payload = {
        "label": label,
        "annotation_path": str(ann_path),
        "prediction_path": str(pred_path),
        "metrics_path": str(metrics_path),
        "counts": {
            "images": len(image_ids),
            "annotations": len(ann_ids),
            "predictions": len(predictions),
            "annotation_count_per_image": _summarize_distribution(ann_counts),
            "density_distribution": dict(sorted(density_distribution.items())),
        },
        "overall": {"bbox": bbox, "segm": segm},
        "metric_check_max_abs_diff": _expected_metric_check({"overall": {"bbox": bbox, "segm": segm}}, metrics_path, args.tolerance),
        "proxy_bootstrap": {
            "segm_ap_proxy": _bootstrap(ap_values, samples=args.bootstrap_samples, seed=args.seed),
            "segm_ap50_proxy": _bootstrap([float(row["ap50_proxy"]) for row in per_image], samples=args.bootstrap_samples, seed=args.seed + 1),
            "segm_ap75_proxy": _bootstrap([float(row["ap75_proxy"]) for row in per_image], samples=args.bootstrap_samples, seed=args.seed + 2),
        },
        "label_count_trend": {
            "pearson_r_count_vs_ap_proxy": _pearson(ann_counts, ap_values),
            "linear_slope_ap_proxy_per_annotation": _linear_slope(ann_counts, ap_values),
        },
        "bucket_metadata": bucket_metadata,
        "density_buckets": density_rows,
        "area_buckets": area_rows,
        "per_image_proxy": per_image,
    }
    return payload


def _format_triplet(metrics: dict[str, float]) -> str:
    return " / ".join(f"{float(metrics[key]):.6f}" for key in METRIC_KEYS)


def _write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# R119 Protocol Stability Diagnostics",
        "",
        f"Bootstrap proxy samples: `{summary['args']['bootstrap_samples']}`.",
        "",
        "## Overall Metrics",
        "",
        "| run | images | annotations | segm AP/AP50/AP75 | proxy AP 95% CI | count-vs-proxy r |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in summary["runs"]:
        boot = run["proxy_bootstrap"]["segm_ap_proxy"]
        trend = run["label_count_trend"]
        lines.append(
            f"| {run['label']} | {run['counts']['images']} | {run['counts']['annotations']} | "
            f"{_format_triplet(run['overall']['segm'])} | "
            f"{boot['mean']:.6f} [{boot['ci_low']:.6f}, {boot['ci_high']:.6f}] | "
            f"{trend['pearson_r_count_vs_ap_proxy']:.6f} |"
        )
    lines.extend(["", "## Density Buckets", ""])
    for run in summary["runs"]:
        lines.extend([
            f"### {run['label']}",
            "",
            "| bucket | images | annotations | segm AP/AP50/AP75 | bbox AP/AP50/AP75 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ])
        for row in run["density_buckets"]:
            lines.append(
                f"| {row['bucket']} | {row['images']} | {row['annotations']} | "
                f"{_format_triplet(row['segm'])} | {_format_triplet(row['bbox'])} |"
            )
        lines.append("")
    lines.extend(["## Area Buckets", ""])
    for run in summary["runs"]:
        lines.extend([
            f"### {run['label']}",
            "",
            "| bucket | images | annotations | segm AP/AP50/AP75 | bbox AP/AP50/AP75 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ])
        for row in run["area_buckets"]:
            lines.append(
                f"| {row['bucket']} | {row['images']} | {row['annotations']} | "
                f"{_format_triplet(row['segm'])} | {_format_triplet(row['bbox'])} |"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_per_image_csv(runs: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run",
                "image_id",
                "file_name",
                "annotations",
                "predictions",
                "mean_gt_area",
                "median_gt_area",
                "ap_proxy",
                "ap50_proxy",
                "ap75_proxy",
            ],
        )
        writer.writeheader()
        for run in runs:
            for row in run["per_image_proxy"]:
                item = dict(row)
                item["run"] = run["label"]
                writer.writerow(item)


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = args.run if args.run else list(DEFAULT_RUNS)
    runs = [_diagnose_one(label, DEFAULT_RUNS[label], args=args) for label in labels]
    summary = {
        "args": {
            "bootstrap_samples": args.bootstrap_samples,
            "seed": args.seed,
            "tolerance": args.tolerance,
        },
        "runs": runs,
    }

    json_safe = copy.deepcopy(summary)
    for run_payload in json_safe["runs"]:
        run_payload.pop("per_image_proxy", None)
    (out_dir / "summary.json").write_text(json.dumps(json_safe, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "per_image_proxy.json").write_text(
        json.dumps(
            [{"run": run_payload["label"], "rows": run_payload["per_image_proxy"]} for run_payload in runs],
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_per_image_csv(runs, out_dir / "per_image_proxy.csv")
    _write_markdown(summary, out_dir / "summary.md")
    print(json.dumps({"out_dir": str(out_dir), "runs": labels}, sort_keys=True))
    return summary


def self_test() -> int:
    mask = {"size": [4, 4], "counts": mask_utils.encode(np.asfortranarray(np.ones((4, 4), dtype=np.uint8)))["counts"].decode("ascii")}
    coco = {
        "images": [{"id": 1, "file_name": "one.png", "height": 4, "width": 4}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "segmentation": mask, "bbox": [0, 0, 4, 4], "area": 16, "iscrowd": 0}],
        "categories": [{"id": 1, "name": "cell"}],
    }
    pred = [{"image_id": 1, "category_id": 1, "segmentation": mask, "bbox": [0, 0, 4, 4], "score": 0.99}]
    with tempfile.TemporaryDirectory(prefix="r119_selftest_") as tmpdir:
        ann_path = Path(tmpdir) / "ann.json"
        pred_path = Path(tmpdir) / "pred.json"
        metrics_path = Path(tmpdir) / "metrics.json"
        ann_path.write_text(json.dumps(coco), encoding="utf-8")
        pred_path.write_text(json.dumps(pred), encoding="utf-8")
        metrics_path.write_text(
            json.dumps({"bbox_AP": 1.0, "bbox_AP50": 1.0, "bbox_AP75": 1.0, "segm_AP": 1.0, "segm_AP50": 1.0, "segm_AP75": 1.0}),
            encoding="utf-8",
        )
        test_args = argparse.Namespace(bootstrap_samples=20, seed=1, tolerance=1e-9)
        payload = _diagnose_one("selftest", {"ann": str(ann_path), "pred": str(pred_path), "metrics": str(metrics_path)}, args=test_args)
    segm_ap = payload["overall"]["segm"]["AP"]
    proxy_ap = payload["proxy_bootstrap"]["segm_ap_proxy"]["mean"]
    if abs(segm_ap - 1.0) > 1e-9 or abs(proxy_ap - 1.0) > 1e-9:
        raise DiagnosisError(f"self-test failed: segm_ap={segm_ap}, proxy_ap={proxy_ap}")
    print(json.dumps({"self_test": "passed", "segm_ap": segm_ap, "proxy_ap": proxy_ap}, sort_keys=True))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="output/diagnostics/r119_protocol_stability_20260518", type=Path)
    parser.add_argument("--bootstrap-samples", default=500, type=int)
    parser.add_argument("--seed", default=119, type=int)
    parser.add_argument("--tolerance", default=1e-9, type=float)
    parser.add_argument("--run", choices=sorted(DEFAULT_RUNS), action="append")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        return self_test()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
