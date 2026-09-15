#!/usr/bin/env python3
"""Rescore a COCO detections JSON under different maxDets and box-NMS settings.

Measures how much of AP_s is ranking-limited: score the same detections with
COCOeval maxDets=100 vs 200/300, and after per-image box-IoU NMS (which
promotes small TPs by removing duplicate large dets above them).

Usage:
  python rescore_dets.py --dets FILE [--nms-iou 0.7 0.85] [--save OUT.json]
"""
import argparse
import copy
import json

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

GT_DEFAULT = (
    "/home/hdd3/zhanghaonan/magformer/magformer_datasets/20260318_1K_32254/"
    "annotations/instances_val.validated.json"
)


def box_nms(rows, iou_thr):
    """Per-image greedy box NMS, keep higher score. Returns kept row indices."""
    by_img = {}
    for i, r in enumerate(rows):
        by_img.setdefault(r["image_id"], []).append(i)
    keep = []
    for _, idxs in by_img.items():
        idxs.sort(key=lambda i: -rows[i]["score"])
        kept = []
        for i in idxs:
            x, y, w, h = rows[i]["bbox"]
            ok = True
            for j in kept:
                x2, y2, w2, h2 = rows[j]["bbox"]
                ix = max(0, min(x + w, x2 + w2) - max(x, x2))
                iy = max(0, min(y + h, y2 + h2) - max(y, y2))
                inter = ix * iy
                union = w * h + w2 * h2 - inter
                if union > 0 and inter / union > iou_thr:
                    ok = False
                    break
            if ok:
                kept.append(i)
        keep.extend(kept)
    return keep


def run_eval(coco_gt, rows, iou_type, max_dets):
    coco_dt = coco_gt.loadRes(copy.deepcopy(rows))
    e = COCOeval(coco_gt, coco_dt, iouType=iou_type)
    e.params.maxDets = [1, 10, max_dets]
    e.evaluate()
    e.accumulate()
    e.summarize()
    p = e.stats
    return {
        "AP": float(p[0]), "AP50": float(p[1]), "AP75": float(p[2]),
        "AP_small": float(p[3]), "AP_medium": float(p[4]), "AP_large": float(p[5]),
        f"AR@{max_dets}": float(p[8]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dets", required=True)
    ap.add_argument("--gt", default=GT_DEFAULT)
    ap.add_argument("--nms-iou", nargs="*", type=float, default=[],
                    help="box-NMS thresholds to sweep (e.g., 0.7 0.85)")
    ap.add_argument("--max-dets", nargs="*", type=int, default=[100, 200])
    ap.add_argument("--save", default=None)
    args = ap.parse_args()

    with open(args.dets) as f:
        rows = json.load(f)
    print(f"dets={len(rows)} file={args.dets}")
    coco_gt = COCO(args.gt)

    results = {}
    variants = {"raw": rows}
    for thr in args.nms_iou:
        keep = box_nms(rows, thr)
        variants[f"nms_{thr}"] = [rows[i] for i in keep]
        print(f"NMS iou={thr}: kept {len(keep)}/{len(rows)}")

    for name, rws in variants.items():
        results[name] = {}
        for iou in ("segm", "bbox"):
            results[name][iou] = {}
            for md in args.max_dets:
                print(f"\n===== {name} iou={iou} maxDets={md} =====")
                results[name][iou][f"md{md}"] = run_eval(coco_gt, rws, iou, md)

    if args.save:
        with open(args.save, "w") as f:
            json.dump({"file": args.dets, "results": results}, f, indent=1)
        print(f"\nwrote {args.save}")


if __name__ == "__main__":
    main()
