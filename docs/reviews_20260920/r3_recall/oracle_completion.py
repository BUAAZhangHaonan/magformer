#!/usr/bin/env python3
"""Oracle recall completion: quantify how many AP_s points the pure
"never detected at all" dimension is worth.

Variants:
  A: for every small GT (area<1024) with best-det IoU<0.5, inject its own GT
     mask as a det with score 0.99 (perfect recall completion).
  B: same but score 0.45 (realistic small-TP score level, tests ranking slot
     competition inside maxDets=100).
  C: inject GT mask for EVERY small GT at 0.99 (recall + mask-quality oracle
     for the small class; upper bound of everything except scoring).

Also prints AP50_small for the baseline (PR-curve readout).
Outputs oracle_completion.json.
"""
import contextlib
import io
import json
import os
import time
from collections import defaultdict

import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from pycocotools.cocoeval import COCOeval

HERE = os.path.dirname(os.path.abspath(__file__))
DETS = ("/home/hdd3/zhanghaonan/magformer/output/aps_20260913/p5_runs/"
        "f1_full_design_256k/coco_instances_results.json")
GT = ("/home/hdd3/zhanghaonan/magformer/magformer_datasets/20260318_1K_32254/"
      "annotations/instances_val.validated.json")


def gt_rle(a, h, w):
    seg = a["segmentation"]
    if isinstance(seg, list):
        rles = maskUtils.frPyObjects(seg, h, w)
        return maskUtils.merge(rles)
    if isinstance(seg.get("counts"), list):
        return maskUtils.frPyObjects(seg, h, w)
    return seg


def det_rle(d):
    s = d["segmentation"]
    c = s["counts"]
    return {"size": s["size"], "counts": c.encode() if isinstance(c, str) else c}


t0 = time.time()
print("loading ...", flush=True)
with contextlib.redirect_stdout(io.StringIO()):
    coco = COCO(GT)
dets = json.load(open(DETS))
img_hw = {im["id"]: (im["height"], im["width"]) for im in coco.dataset["images"]}
gt_by_img = defaultdict(list)
for a in coco.dataset["annotations"]:
    gt_by_img[a["image_id"]].append(a)
dets_by_img = defaultdict(list)
for d in dets:
    dets_by_img[d["image_id"]].append(d)
print(f"loaded in {time.time()-t0:.0f}s", flush=True)

# find never-hit small GTs
never_hit = []   # ann dicts
all_small = []
for img_id, anns in gt_by_img.items():
    dds = dets_by_img.get(img_id, [])
    small = [a for a in anns if 0 < a["area"] < 1024]
    all_small.extend(small)
    if not small:
        continue
    h, w = img_hw[img_id]
    dr = [det_rle(d) for d in dds]
    gr = [gt_rle(a, h, w) for a in small]
    if dr:
        iou = maskUtils.iou(dr, gr, [0] * len(gr))
        best = iou.max(axis=0)
    else:
        best = np.zeros(len(gr))
    never_hit.extend([a for a, b in zip(small, best) if b < 0.5])
print(f"never-hit small GT: {len(never_hit)}/{len(all_small)}", flush=True)


def rle_counts_bytes(r):
    return {"size": r["size"],
            "counts": r["counts"].encode() if isinstance(r["counts"], str) else r["counts"]}


def inject(anns, score):
    rows = []
    for a in anns:
        h, w = img_hw[a["image_id"]]
        r = gt_rle(a, h, w)
        rows.append({"image_id": a["image_id"], "category_id": 1,
                     "score": score, "segmentation": rle_counts_bytes(r),
                     "bbox": list(a.get("bbox", [0, 0, 1, 1]))})
    return rows


def run(dd, tag):
    coco_dt = coco.loadRes(dd)
    e = COCOeval(coco, coco_dt, iouType="segm")
    e.params.maxDets = [1, 10, 100]
    with contextlib.redirect_stdout(io.StringIO()):
        e.evaluate()
        e.accumulate()
        e.summarize()
    p = e.stats
    prc = e.eval["precision"]
    # AP50 small: mean of defined entries at iou .5, area small, md100
    v = prc[0, :, 0, 1, 2]
    v = v[v > -1]
    r = e.eval["recall"]
    out = {"AP": float(p[0]), "AP_small": float(p[3]), "AP50_small": float(v.mean()),
           "AR_small@100": float(np.nanmean(r[:, :, 1, 2]))}
    print(tag, json.dumps(out), flush=True)
    return out


out = {}
out["baseline"] = run(dets, "baseline")
out["A_neverhit_0.99"] = run(dets + inject(never_hit, 0.99), "A")
out["B_neverhit_0.45"] = run(dets + inject(never_hit, 0.45), "B")
out["C_allsmall_0.99"] = run(dets + inject(all_small, 0.99), "C")
with open(os.path.join(HERE, "oracle_completion.json"), "w") as f:
    json.dump(out, f, indent=1)
print("DONE")
