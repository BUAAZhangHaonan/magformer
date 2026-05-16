#!/usr/bin/env python3
"""Diagnose per-run and union prediction oracle bounds for COCO masks."""

import argparse
import copy
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from scipy.optimize import linear_sum_assignment


DENSITY_BUCKETS = [
    ("25-30", lambda n: 25 <= n <= 30),
    ("46-60", lambda n: 46 <= n <= 60),
    (">90", lambda n: n > 90),
]
AREA_BUCKETS = [
    ("<=256", lambda a: a <= 256),
    ("257-452", lambda a: 257 <= a <= 452),
    ("453-579", lambda a: 453 <= a <= 579),
    (">579", lambda a: a > 579),
]
THRESHOLDS = (0.50, 0.75, 0.90)


def _safe_label(label):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_") or "run"


def _pct(values, qs=(10, 50, 90)):
    if not values:
        return {f"p{q}": None for q in qs}
    arr = np.asarray(values, dtype=np.float64)
    return {f"p{q}": float(np.percentile(arr, q)) for q in qs}


def _bucket_name(value, buckets):
    for name, pred in buckets:
        if pred(value):
            return name
    return "other"


def _rle(segmentation, height, width):
    if isinstance(segmentation, dict):
        out = dict(segmentation)
        if isinstance(out.get("counts"), str):
            out["counts"] = out["counts"].encode("ascii")
        return out
    return mask_utils.frPyObjects(segmentation, height, width)


def _compute_iou(gt_rles, pred_rles, iscrowd):
    if not gt_rles or not pred_rles:
        return np.zeros((len(gt_rles), len(pred_rles)), dtype=np.float64)
    return mask_utils.iou(pred_rles, gt_rles, iscrowd).T


def _recall_from_max(max_ious):
    total = len(max_ious)
    return {
        f"@{int(t * 100)}": float(np.sum(max_ious >= t) / total) if total else 0.0
        for t in THRESHOLDS
    }


def _one_to_one_counts(iou_matrix):
    counts = {t: 0 for t in (0.50, 0.75)}
    if iou_matrix.size == 0:
        return counts
    rows, cols = linear_sum_assignment(-iou_matrix)
    matched_ious = iou_matrix[rows, cols]
    for threshold in counts:
        counts[threshold] = int(np.sum(matched_ious >= threshold))
    return counts


def _mean_coco_metrics(eval_obj):
    def mean_precision(iou_thr=None, area="all", max_det=200):
        precision = eval_obj.eval["precision"]
        area_idx = list(eval_obj.params.areaRngLbl).index(area)
        max_det_idx = list(eval_obj.params.maxDets).index(max_det)
        if iou_thr is None:
            vals = precision[:, :, :, area_idx, max_det_idx]
        else:
            idx = np.where(np.isclose(eval_obj.params.iouThrs, iou_thr))[0]
            vals = precision[idx, :, :, area_idx, max_det_idx]
        vals = vals[vals > -1]
        return float(np.mean(vals)) if vals.size else -1.0

    def mean_recall(area="all", max_det=200):
        recall = eval_obj.eval["recall"]
        area_idx = list(eval_obj.params.areaRngLbl).index(area)
        max_det_idx = list(eval_obj.params.maxDets).index(max_det)
        vals = recall[:, :, area_idx, max_det_idx]
        vals = vals[vals > -1]
        return float(np.mean(vals)) if vals.size else -1.0

    return {
        "AP": mean_precision(max_det=200),
        "AP50": mean_precision(iou_thr=0.50, max_det=200),
        "AP75": mean_precision(iou_thr=0.75, max_det=200),
        "APs": mean_precision(area="small", max_det=200),
        "APm": mean_precision(area="medium", max_det=200),
        "APl": mean_precision(area="large", max_det=200),
        "AR1": mean_recall(max_det=1),
        "AR10": mean_recall(max_det=10),
        "AR200": mean_recall(max_det=200),
        "ARs": mean_recall(area="small", max_det=200),
        "ARm": mean_recall(area="medium", max_det=200),
        "ARl": mean_recall(area="large", max_det=200),
    }


def _coco_eval(ann_path, predictions, iou_type):
    if not predictions:
        return {}
    coco_gt = COCO(str(ann_path))
    coco_dt = coco_gt.loadRes(predictions)
    eval_obj = COCOeval(coco_gt, coco_dt, iou_type)
    eval_obj.params.maxDets = [1, 10, 200]
    eval_obj.evaluate()
    eval_obj.accumulate()
    return _mean_coco_metrics(eval_obj)


def _summarize_records(records, key, buckets):
    out = {}
    for name in [bucket[0] for bucket in buckets] + ["other"]:
        vals = np.asarray([record["max_iou"] for record in records if record[key] == name], dtype=np.float64)
        if vals.size == 0:
            out[name] = {"gt": 0, "oracle_recall": _recall_from_max(vals)}
            continue
        out[name] = {
            "gt": int(vals.size),
            "oracle_recall": _recall_from_max(vals),
            "max_iou_p10_p50_p90": _pct(vals.tolist()),
        }
    return out


def _summarize_one_to_one(bucket_counts):
    out = {}
    for name, data in bucket_counts.items():
        gt_count = data["gt"]
        out[name] = {
            "gt": int(gt_count),
            "one_to_one_recall@50": float(data["matched50"] / gt_count) if gt_count else 0.0,
            "one_to_one_recall@75": float(data["matched75"] / gt_count) if gt_count else 0.0,
            "matched@50": int(data["matched50"]),
            "matched@75": int(data["matched75"]),
        }
    return dict(sorted(out.items()))


def _load_context(ann_path):
    with Path(ann_path).open() as f:
        gt = json.load(f)
    images = {image["id"]: image for image in gt["images"]}
    anns_by_image = defaultdict(list)
    for ann in gt["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)
    image_ids = [image["id"] for image in gt["images"]]
    density = {image_id: len(anns_by_image[image_id]) for image_id in image_ids}
    gt_order = [ann["id"] for image_id in image_ids for ann in anns_by_image[image_id]]
    return {
        "gt": gt,
        "images": images,
        "anns_by_image": anns_by_image,
        "image_ids": image_ids,
        "density": density,
        "gt_order": gt_order,
    }


def _load_predictions(predictions):
    loaded = {}
    paths = {}
    for label, path in predictions:
        with Path(path).open() as f:
            loaded[label] = json.load(f)
        paths[label] = str(path)
    return loaded, paths


def evaluate_pool(label, predictions, ann_path, context, out_dir, skip_coco_eval=False):
    preds_by_image = defaultdict(list)
    for idx, pred in enumerate(predictions):
        preds_by_image[pred["image_id"]].append((idx, pred))

    max_ious_all = []
    gt_records = []
    per_gt_max_iou = {}
    one_to_one_total = {0.50: 0, 0.75: 0}
    one_to_one_by_density = defaultdict(lambda: {"gt": 0, "matched50": 0, "matched75": 0})
    one_to_one_by_area = defaultdict(lambda: {"gt": 0, "matched50": 0, "matched75": 0})
    pred_max_iou_scores = {}
    pred_matched_iou_scores = {}

    for image_id in context["image_ids"]:
        image = context["images"][image_id]
        anns = context["anns_by_image"][image_id]
        pred_items = preds_by_image[image_id]
        pred_list = [item[1] for item in pred_items]
        gt_rles = [_rle(ann["segmentation"], image["height"], image["width"]) for ann in anns]
        pred_rles = [_rle(pred["segmentation"], image["height"], image["width"]) for pred in pred_list]
        iscrowd = [int(ann.get("iscrowd", 0)) for ann in anns]
        ious = _compute_iou(gt_rles, pred_rles, iscrowd)

        if pred_list:
            if anns:
                pred_best = ious.max(axis=0)
                for col, (pred_idx, _pred) in enumerate(pred_items):
                    pred_max_iou_scores[pred_idx] = float(pred_best[col])
            else:
                for pred_idx, _pred in pred_items:
                    pred_max_iou_scores[pred_idx] = 0.0

        max_ious = ious.max(axis=1) if len(anns) and pred_list else np.zeros(len(anns), dtype=np.float64)
        max_ious_all.extend(max_ious.tolist())
        counts = _one_to_one_counts(ious)
        for threshold, count in counts.items():
            one_to_one_total[threshold] += count

        if ious.size:
            rows, cols = linear_sum_assignment(-ious)
            for row, col in zip(rows, cols):
                value = float(ious[int(row), int(col)])
                pred_matched_iou_scores[pred_items[int(col)][0]] = value
                ann = anns[int(row)]
                d_bucket = _bucket_name(context["density"][image_id], DENSITY_BUCKETS)
                a_bucket = _bucket_name(float(ann.get("area", 0)), AREA_BUCKETS)
                if value >= 0.50:
                    one_to_one_by_density[d_bucket]["matched50"] += 1
                    one_to_one_by_area[a_bucket]["matched50"] += 1
                if value >= 0.75:
                    one_to_one_by_density[d_bucket]["matched75"] += 1
                    one_to_one_by_area[a_bucket]["matched75"] += 1

        for row_idx, ann in enumerate(anns):
            max_iou = float(max_ious[row_idx])
            per_gt_max_iou[ann["id"]] = max_iou
            d_bucket = _bucket_name(context["density"][image_id], DENSITY_BUCKETS)
            a_bucket = _bucket_name(float(ann.get("area", 0)), AREA_BUCKETS)
            gt_records.append(
                {
                    "gt_id": ann["id"],
                    "image_id": image_id,
                    "density": context["density"][image_id],
                    "density_bucket": d_bucket,
                    "area": float(ann.get("area", 0)),
                    "area_bucket": a_bucket,
                    "max_iou": max_iou,
                }
            )
            one_to_one_by_density[d_bucket]["gt"] += 1
            one_to_one_by_area[a_bucket]["gt"] += 1

    total_gt = len(context["gt"]["annotations"])
    max_arr = np.asarray(max_ious_all, dtype=np.float64)
    safe_label = _safe_label(label)
    score_max_path = Path(out_dir) / f"{safe_label}_score_max_iou_predictions.json"
    score_matched_path = Path(out_dir) / f"{safe_label}_score_matched_iou_predictions.json"

    score_max_preds = []
    score_matched_preds = []
    for idx, pred in enumerate(predictions):
        max_pred = copy.deepcopy(pred)
        max_pred["score"] = pred_max_iou_scores.get(idx, 0.0)
        score_max_preds.append(max_pred)
        matched_pred = copy.deepcopy(pred)
        matched_pred["score"] = pred_matched_iou_scores.get(idx, 0.0)
        score_matched_preds.append(matched_pred)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with score_max_path.open("w") as f:
        json.dump(score_max_preds, f)
    with score_matched_path.open("w") as f:
        json.dump(score_matched_preds, f)

    oracle_score_ap = {
        "max_iou_score": {"prediction_file": str(score_max_path), "segm": None, "bbox": None},
        "matched_iou_score": {"prediction_file": str(score_matched_path), "segm": None, "bbox": None},
    }
    if not skip_coco_eval:
        oracle_score_ap["max_iou_score"]["segm"] = _coco_eval(ann_path, score_max_preds, "segm")
        oracle_score_ap["max_iou_score"]["bbox"] = _coco_eval(ann_path, score_max_preds, "bbox")
        oracle_score_ap["matched_iou_score"]["segm"] = _coco_eval(ann_path, score_matched_preds, "segm")
        oracle_score_ap["matched_iou_score"]["bbox"] = _coco_eval(ann_path, score_matched_preds, "bbox")

    result = {
        "label": label,
        "images": len(context["image_ids"]),
        "gt": total_gt,
        "predictions": len(predictions),
        "oracle_recall": _recall_from_max(max_arr),
        "max_iou_p10_p50_p90": _pct(max_ious_all),
        "one_to_one_oracle_recall": {
            "@50": float(one_to_one_total[0.50] / total_gt) if total_gt else 0.0,
            "@75": float(one_to_one_total[0.75] / total_gt) if total_gt else 0.0,
            "matched@50": int(one_to_one_total[0.50]),
            "matched@75": int(one_to_one_total[0.75]),
        },
        "by_density_bucket": _summarize_records(gt_records, "density_bucket", DENSITY_BUCKETS),
        "one_to_one_by_density_bucket": _summarize_one_to_one(one_to_one_by_density),
        "by_area_bucket": _summarize_records(gt_records, "area_bucket", AREA_BUCKETS),
        "one_to_one_by_area_bucket": _summarize_one_to_one(one_to_one_by_area),
        "oracle_score_ap": oracle_score_ap,
        "_per_gt_max_iou": per_gt_max_iou,
    }

    out_path = Path(out_dir) / f"{safe_label}_oracle_bound_summary.json"
    public_result = {key: value for key, value in result.items() if not key.startswith("_")}
    with out_path.open("w") as f:
        json.dump(public_result, f, indent=2)
    return result


def _coverage_overlap(run_results, gt_order, threshold=0.75):
    labels = list(run_results)
    signatures = defaultdict(int)
    run_only = {label: 0 for label in labels}
    shared_misses = 0
    covered_by_any = 0
    covered_by_all = 0

    for gt_id in gt_order:
        covered = tuple(
            label for label in labels if run_results[label]["_per_gt_max_iou"].get(gt_id, 0.0) >= threshold
        )
        if not covered:
            shared_misses += 1
        else:
            covered_by_any += 1
        if len(covered) == len(labels):
            covered_by_all += 1
        if len(covered) == 1:
            run_only[covered[0]] += 1
        signatures["+".join(covered) if covered else "missed_by_all"] += 1

    return {
        "threshold": threshold,
        "gt": len(gt_order),
        "covered_by_any@75": covered_by_any,
        "covered_by_all@75": covered_by_all,
        "shared_misses@75": shared_misses,
        "run_only_covered@75": run_only,
        "coverage_signature_counts@75": dict(sorted(signatures.items())),
    }


def run_diagnostic(ann_path, predictions, out_dir, skip_coco_eval=False):
    ann_path = Path(ann_path)
    out_dir = Path(out_dir)
    context = _load_context(ann_path)
    loaded_predictions, pred_paths = _load_predictions(predictions)

    run_results = {}
    for label, preds in loaded_predictions.items():
        run_results[label] = evaluate_pool(label, preds, ann_path, context, out_dir, skip_coco_eval)

    union_predictions = []
    for label, preds in loaded_predictions.items():
        for pred in preds:
            union_pred = copy.deepcopy(pred)
            union_pred["source_run"] = label
            union_predictions.append(union_pred)
    union_result = evaluate_pool("union", union_predictions, ann_path, context, out_dir, skip_coco_eval)

    public_runs = {
        label: {key: value for key, value in result.items() if not key.startswith("_")}
        for label, result in run_results.items()
    }
    summary = {
        "annotation_path": str(ann_path),
        "prediction_paths": pred_paths,
        "skip_coco_eval": bool(skip_coco_eval),
        "runs": public_runs,
        "union": {key: value for key, value in union_result.items() if not key.startswith("_")},
        "coverage_overlap": _coverage_overlap(run_results, context["gt_order"], threshold=0.75),
        "notes": {
            "oracle_score_ap": "Scores are replaced with GT-derived mask IoU; masks and boxes are unchanged.",
            "union": "Union concatenates all input prediction JSON files and adds source_run metadata only.",
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "r53_prediction_union_oracle_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ann", required=True, help="COCO GT annotation JSON")
    parser.add_argument(
        "--pred",
        action="append",
        nargs=2,
        metavar=("LABEL", "PATH"),
        required=True,
        help="Prediction label and COCO result JSON. Repeat for multiple runs.",
    )
    parser.add_argument("--out-dir", required=True, help="Output directory for summaries and oracle-score predictions")
    parser.add_argument(
        "--skip-coco-eval",
        action="store_true",
        help="Skip oracle-score COCOeval. Intended only for lightweight synthetic tests.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    summary = run_diagnostic(
        ann_path=args.ann,
        predictions=[(label, Path(path)) for label, path in args.pred],
        out_dir=args.out_dir,
        skip_coco_eval=args.skip_coco_eval,
    )
    compact = {
        "runs": {
            label: {
                "oracle_r75": result["oracle_recall"]["@75"],
                "matched_oracle_score_segm_ap": (
                    result["oracle_score_ap"]["matched_iou_score"]["segm"] or {}
                ).get("AP"),
            }
            for label, result in summary["runs"].items()
        },
        "union": {
            "oracle_r75": summary["union"]["oracle_recall"]["@75"],
            "matched_oracle_score_segm_ap": (
                summary["union"]["oracle_score_ap"]["matched_iou_score"]["segm"] or {}
            ).get("AP"),
        },
        "coverage_overlap": summary["coverage_overlap"],
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
