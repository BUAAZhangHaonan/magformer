#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def _eval_one(coco_gt: COCO, coco_dt: COCO, iou_type: str) -> Dict[str, float]:
    coco_eval = COCOeval(coco_gt, coco_dt, iouType=iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Stats are in [0,1]; align with detectron2 json output (0..100).
    stats = [float(x) * 100.0 for x in coco_eval.stats]
    return {
        f"{iou_type}/AP": stats[0],
        f"{iou_type}/AP50": stats[1],
        f"{iou_type}/AP75": stats[2],
        f"{iou_type}/APs": stats[3],
        f"{iou_type}/APm": stats[4],
        f"{iou_type}/APl": stats[5],
    }


def _zero_metrics(iou_type: str) -> Dict[str, float]:
    return {
        f"{iou_type}/AP": 0.0,
        f"{iou_type}/AP50": 0.0,
        f"{iou_type}/AP75": 0.0,
        f"{iou_type}/APs": 0.0,
        f"{iou_type}/APm": 0.0,
        f"{iou_type}/APl": 0.0,
    }


def evaluate_coco_results(
    ann_file: Path,
    results_json: Path,
    iteration: int = -1,
) -> Dict[str, Any]:
    coco_gt = COCO(str(ann_file))

    rows: List[Dict[str, Any]] = json.loads(
        results_json.read_text(encoding="utf-8"))
    metrics: Dict[str, Any] = {"iteration": int(iteration)}
    if len(rows) == 0:
        metrics.update(_zero_metrics("bbox"))
        metrics.update(_zero_metrics("segm"))
        return metrics

    coco_dt = coco_gt.loadRes(str(results_json))
    metrics.update(_eval_one(coco_gt, coco_dt, "bbox"))
    metrics.update(_eval_one(coco_gt, coco_dt, "segm"))
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann-file", type=str, required=True)
    ap.add_argument("--results-json", type=str, required=True)
    ap.add_argument("--output-metrics", type=str, required=True)
    ap.add_argument("--iteration", type=int, default=-1)
    args = ap.parse_args()

    ann_file = Path(args.ann_file)
    results_json = Path(args.results_json)
    output_metrics = Path(args.output_metrics)
    metrics = evaluate_coco_results(
        ann_file=ann_file,
        results_json=results_json,
        iteration=int(args.iteration),
    )

    output_metrics.parent.mkdir(parents=True, exist_ok=True)
    output_metrics.write_text(json.dumps(
        metrics, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[coco-eval] wrote: {output_metrics}")


if __name__ == "__main__":
    main()
