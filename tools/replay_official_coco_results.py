#!/usr/bin/env python3
"""Replay saved official COCO predictions against source annotations."""

from __future__ import annotations

import argparse
import copy
import json
import tempfile
from pathlib import Path
from typing import Any, Iterable

from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

METRIC_NAMES = ("AP", "AP50", "AP75", "APs", "APm", "APl")


def load_coco_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a COCO annotation object")
    for key in ("images", "annotations", "categories"):
        if not isinstance(payload.get(key), list):
            raise ValueError(f"{path} is missing COCO list field: {key}")
    out = copy.deepcopy(payload)
    out.setdefault("info", {})
    out.setdefault("licenses", [])
    return out


def _load_result_rows(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"{path} must contain a COCO result list")
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{path}[{idx}] must be an object")
    return rows


def _rle_for_eval(segmentation: dict[str, Any]) -> dict[str, Any]:
    rle = dict(segmentation)
    counts = rle.get("counts")
    if isinstance(counts, str):
        rle["counts"] = counts.encode("ascii")
    return rle


def _bbox_from_rle(segmentation: dict[str, Any]) -> list[float]:
    return [float(value) for value in mask_utils.toBbox(_rle_for_eval(segmentation)).tolist()]


def _area_from_rle(segmentation: dict[str, Any]) -> float:
    return float(mask_utils.area(_rle_for_eval(segmentation)))


def prepare_result_rows(
    rows: Iterable[dict[str, Any]],
    *,
    drop_empty_masks: bool = True,
    recompute_bbox: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    prepared: list[dict[str, Any]] = []
    summary = {
        "input": 0,
        "kept": 0,
        "dropped_empty_masks": 0,
        "recomputed_bboxes": 0,
    }
    for row in rows:
        summary["input"] += 1
        fixed = copy.deepcopy(row)
        segmentation = fixed.get("segmentation")
        if isinstance(segmentation, dict):
            area = _area_from_rle(segmentation)
            if drop_empty_masks and area <= 0.0:
                summary["dropped_empty_masks"] += 1
                continue
            if recompute_bbox:
                bbox = _bbox_from_rle(segmentation)
                old_bbox = fixed.get("bbox")
                if old_bbox != bbox:
                    summary["recomputed_bboxes"] += 1
                fixed["bbox"] = bbox
        prepared.append(fixed)
        summary["kept"] += 1
    return prepared, summary


def _zero_metric_block() -> dict[str, float]:
    return {name: 0.0 for name in METRIC_NAMES}


def _mean_precision(coco_eval: COCOeval, *, area_label: str, iou_threshold: float | None) -> float:
    precision = coco_eval.eval["precision"]
    area_idx = list(coco_eval.params.areaRngLbl).index(area_label)
    max_det_idx = list(coco_eval.params.maxDets).index(coco_eval.params.maxDets[-1])
    if iou_threshold is None:
        values = precision[:, :, :, area_idx, max_det_idx]
    else:
        iou_indices = [
            idx
            for idx, value in enumerate(coco_eval.params.iouThrs)
            if abs(float(value) - iou_threshold) < 1e-9
        ]
        if not iou_indices:
            return -100.0
        values = precision[iou_indices, :, :, area_idx, max_det_idx]
    values = values[values > -1]
    if values.size == 0:
        return -100.0
    return float(values.mean()) * 100.0


def _metric_block(coco_eval: COCOeval) -> dict[str, float]:
    return {
        "AP": _mean_precision(coco_eval, area_label="all", iou_threshold=None),
        "AP50": _mean_precision(coco_eval, area_label="all", iou_threshold=0.50),
        "AP75": _mean_precision(coco_eval, area_label="all", iou_threshold=0.75),
        "APs": _mean_precision(coco_eval, area_label="small", iou_threshold=None),
        "APm": _mean_precision(coco_eval, area_label="medium", iou_threshold=None),
        "APl": _mean_precision(coco_eval, area_label="large", iou_threshold=None),
    }


def _eval_one(coco_gt: COCO, rows: list[dict[str, Any]], iou_type: str, max_det: int) -> dict[str, float]:
    if not rows:
        return _zero_metric_block()
    coco_dt = coco_gt.loadRes(rows)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType=iou_type)
    coco_eval.params.maxDets = [1, 10, int(max_det)]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return _metric_block(coco_eval)


def run_replay(
    *,
    ann_path: Path,
    results_path: Path,
    output_metrics: Path,
    drop_empty_masks: bool = True,
    recompute_bbox: bool = True,
    max_dets: Iterable[int] = (100, 200),
) -> dict[str, Any]:
    payload = load_coco_payload(ann_path)
    rows, counts = prepare_result_rows(
        _load_result_rows(results_path),
        drop_empty_masks=drop_empty_masks,
        recompute_bbox=recompute_bbox,
    )
    max_det_values = [int(value) for value in max_dets]
    metrics: dict[str, Any] = {
        "inputs": {
            "annotations": str(ann_path),
            "results": str(results_path),
        },
        "options": {
            "drop_empty_masks": bool(drop_empty_masks),
            "recompute_bbox": bool(recompute_bbox),
            "max_dets": max_det_values,
        },
        "result_counts": counts,
        "bbox": {},
        "segm": {},
    }

    with tempfile.TemporaryDirectory(prefix="official_coco_replay_") as tmpdir:
        tmp_ann_path = Path(tmpdir) / "annotations.with_metadata.json"
        tmp_ann_path.write_text(json.dumps(payload), encoding="utf-8")
        coco_gt = COCO(str(tmp_ann_path))
        for max_det in max_det_values:
            key = f"maxDets{max_det}"
            metrics["bbox"][key] = _eval_one(coco_gt, rows, "bbox", max_det)
            metrics["segm"][key] = _eval_one(coco_gt, rows, "segm", max_det)

    output_metrics.parent.mkdir(parents=True, exist_ok=True)
    output_metrics.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ann-json", "--ann-file", dest="ann_path", type=Path, required=True)
    parser.add_argument("--results-json", type=Path, required=True)
    parser.add_argument("--output-metrics", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-dets", type=int, nargs="+", default=[100, 200])
    parser.add_argument("--keep-empty-masks", action="store_true")
    parser.add_argument("--no-recompute-bbox", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_metrics = args.output_metrics
    if output_metrics is None:
        output_dir = args.output_dir if args.output_dir is not None else args.results_json.parent
        output_metrics = output_dir / "metrics.cocoeval.json"
    metrics = run_replay(
        ann_path=args.ann_path,
        results_path=args.results_json,
        output_metrics=output_metrics,
        drop_empty_masks=not args.keep_empty_masks,
        recompute_bbox=not args.no_recompute_bbox,
        max_dets=args.max_dets,
    )
    print(f"[replay] wrote: {output_metrics}")
    print("[replay] metrics_json=" + json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
