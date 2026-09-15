#!/usr/bin/env python3
"""GT-oracle AP experiments for AP_s diagnosis.

Answers: if detection and scoring were PERFECT, what AP/AP_s does the current
mask-grid quantization allow? Each "oracle grid" simulates a perfect mask head
that outputs logits on a GxG grid (G = 1024/stride): GT mask -> INTER_AREA
down to GxG (per-cell coverage fraction = best achievable logit) -> bilinear
up to 1024 -> threshold 0.5 -> RLE. This is the information-theoretic ceiling
of a stride-s mask head with perfect learning.

Oracles:
  full  = GT masks untouched (protocol sanity, expect AP=1.0)
  256   = stride-4 mask grid  (= current c0 mask head, common_stride=4)
  512   = stride-2 mask grid  (hypothetical high-res head)
  128   = stride-8 grid
  64    = stride-16 grid

Also scores an existing detections JSON (e.g., the c0 fullval results) with
the identical COCOeval setup as an anchor against the official numbers.

Usage:
  python gt_oracle_ap.py --grids full 256 512 128 64 \
      [--detections /path/to/coco_instances_results.json] \
      [--workers 24] [--out DIR]
"""
import argparse
import json
import os
import time
from multiprocessing import Pool

import cv2
import numpy as np
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

GT_DEFAULT = (
    "/home/hdd3/zhanghaonan/magformer/magformer_datasets/20260318_1K_32254/"
    "annotations/instances_val.validated.json"
)


def _one_image(payload):
    img_id, h, w, anns, grids = payload
    out = {g: [] for g in grids}
    for ann_id, cat_id, seg, gt_bbox in anns:
        rles = maskUtils.frPyObjects(seg, h, w)
        rle = maskUtils.merge(rles)
        m = maskUtils.decode(rle)
        if "full" in grids:
            out["full"].append(
                {
                    "image_id": img_id,
                    "category_id": cat_id,
                    "segmentation": {"size": rle["size"], "counts": rle["counts"]},
                    "score": 1.0,
                    "bbox": gt_bbox,
                }
            )
        for g in grids:
            if g == "full":
                continue
            small = cv2.resize(m.astype(np.float32), (g, g), interpolation=cv2.INTER_AREA)
            up = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
            q = (up >= 0.5).astype(np.uint8)
            if q.sum() == 0:
                continue  # object vanished under quantization: a miss by construction
            qrle = maskUtils.encode(np.asfortranarray(q))
            out[g].append(
                {
                    "image_id": img_id,
                    "category_id": cat_id,
                    "segmentation": {"size": qrle["size"], "counts": qrle["counts"]},
                    "score": 1.0,
                    "bbox": maskUtils.toBbox(qrle).tolist(),
                }
            )
    return out


def evaluate_rows(coco_gt, rows, tag, iou_type):
    if not rows:
        print(f"[{tag}] no rows")
        return None
    coco_dt = coco_gt.loadRes(list(rows))
    e = COCOeval(coco_gt, coco_dt, iouType=iou_type)
    e.evaluate()
    e.accumulate()
    e.summarize()
    p = e.stats
    return {
        "AP": float(p[0]),
        "AP50": float(p[1]),
        "AP75": float(p[2]),
        "AP_small": float(p[3]),
        "AP_medium": float(p[4]),
        "AP_large": float(p[5]),
        "AR100": float(p[8]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default=GT_DEFAULT)
    ap.add_argument("--grids", nargs="+", default=["full", "256", "512", "128", "64"])
    ap.add_argument("--detections", default=None, help="anchor: score existing det JSON")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out", default="/home/hdd3/zhanghaonan/magformer/output/aps_20260913/oracle")
    args = ap.parse_args()
    args.grids = [g if g == "full" else int(g) for g in args.grids]
    os.makedirs(args.out, exist_ok=True)

    print("loading GT ...", flush=True)
    with open(args.gt) as f:
        data = json.load(f)
    coco_gt = COCO(args.gt)
    by_img = {}
    for a in data["annotations"]:
        by_img.setdefault(a["image_id"], []).append(
            (a["id"], a["category_id"], a["segmentation"], a["bbox"])
        )
    hw = {im["id"]: (im["height"], im["width"]) for im in data["images"]}
    payloads = []
    for img_id, anns in by_img.items():
        h, w = hw[img_id]
        payloads.append((img_id, h, w, anns, args.grids))

    n_small = sum(1 for a in data["annotations"] if a["area"] < 1024)
    print(
        f"images={len(payloads)} instances={len(data['annotations'])} small={n_small}",
        flush=True,
    )

    t0 = time.time()
    with Pool(args.workers) as pool:
        results = pool.map(_one_image, payloads, chunksize=8)
    print(f"quantized in {time.time() - t0:.1f}s", flush=True)

    merged = {g: [] for g in args.grids}
    for r in results:
        for g in args.grids:
            merged[g].extend(r[g])

    report = {"gt": args.gt, "grids": {}}
    for g in args.grids:
        rows = merged[g]
        vanished = len(by_img and merged["full"]) - len(rows) if g != "full" else 0
        entry = {"n_dets": len(rows)}
        for iou in ("segm", "bbox"):
            print(f"\n===== oracle grid={g} iou={iou} (n={len(rows)}) =====", flush=True)
            entry[iou] = evaluate_rows(coco_gt, rows, f"{g}-{iou}", iou)
        report["grids"][g] = entry

    if args.detections:
        print("\n===== anchor: existing detections =====", flush=True)
        with open(args.detections) as f:
            dets = json.load(f)
        report["anchor"] = {"file": args.detections, "n": len(dets)}
        for iou in ("segm", "bbox"):
            print(f"--- detections iou={iou} ---", flush=True)
            report["anchor"][iou] = evaluate_rows(coco_gt, dets, f"det-{iou}", iou)

    out_path = os.path.join(args.out, "oracle_results.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=1)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
