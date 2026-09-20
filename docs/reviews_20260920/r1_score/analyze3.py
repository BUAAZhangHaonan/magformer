#!/usr/bin/env python3
"""R1 final bound: keepers ranked by IoU (theft-minimal perfect scoring).

score = 1 + IoU*1e-4 for per-GT best-score keepers (ranked by quality, all
above every FP); non-keeper matched dets -> -1 (bottom, they are redundant
for every GT they could match); unmatched unchanged. This is the practical
upper bound of ANY rescoring of the current detection set.
"""
import contextlib
import io
import json
import os
import time

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = "/home/hdd3/zhanghaonan/magformer/output/aps_20260913/p5_runs/f1_full_design_256k"
GT = ("/home/hdd3/zhanghaonan/magformer/magformer_datasets/20260318_1K_32254/"
      "annotations/instances_val.validated.json")
DET = os.path.join(RUN, "eval_snapshots/2001_0920/dets.json")
AREA_RNG = [[0.0, 1e10], [0.0, 1024.0], [0.0, 64.0], [64.0, 256.0], [256.0, 1024.0]]
AREA_LBL = ["all", "small", "<64", "64-256", "256-1024"]


def rle_of(d):
    s = d["segmentation"]
    c = s["counts"]
    if isinstance(c, str):
        c = c.encode()
    return {"size": s["size"], "counts": c}


def main():
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(GT)
    dets = json.load(open(DET))
    gt_by_img = {}
    for a in coco_gt.dataset["annotations"]:
        gt_by_img.setdefault(a["image_id"], []).append(a)
    dets_by_img = {}
    for i, d in enumerate(dets):
        dets_by_img.setdefault(d["image_id"], []).append(i)
    det_max_iou = np.zeros(len(dets), dtype=np.float32)
    gt_matches = {}
    for img_id, gts in gt_by_img.items():
        dids = dets_by_img.get(img_id, [])
        if not dids or not gts:
            continue
        g_rles = [maskUtils.merge(maskUtils.frPyObjects(g["segmentation"], 1024, 1024))
                  for g in gts]
        d_rles = [rle_of(dets[i]) for i in dids]
        iou = maskUtils.iou(d_rles, g_rles, [0] * len(g_rles))
        for li, gi in enumerate(dids):
            det_max_iou[gi] = iou[li].max()
        for lj, g in enumerate(gts):
            col = iou[:, lj]
            hits = [(dids[k], float(col[k])) for k in np.nonzero(col >= 0.5)[0]]
            if hits:
                gt_matches.setdefault(g["id"], []).extend(hits)
    src = np.array([d["score"] for d in dets])
    matched = det_max_iou >= 0.5
    keeper_of = {g: max(h, key=lambda h: src[h[0]])[0] for g, h in gt_matches.items()}
    keepers = set(keeper_of.values())

    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(dets)
    ann_by_det = list(coco_dt.dataset["annotations"])
    orig = [a["score"] for a in ann_by_det]
    for i, a in enumerate(ann_by_det):
        if i in keepers:
            a["score"] = 1.0 + float(det_max_iou[i]) * 1e-4
        elif matched[i]:
            a["score"] = -1.0
    e = COCOeval(coco_gt, coco_dt, "segm")
    e.params.areaRng = [list(r) for r in AREA_RNG]
    e.params.areaRngLbl = list(AREA_LBL)
    t0 = time.time()
    e.evaluate()
    e.accumulate()
    prec = e.eval["precision"]
    rec = e.eval["recall"]
    aidx = {l: i for i, l in enumerate(AREA_LBL)}
    midx = len(e.params.maxDets) - 1
    out = {}
    for lbl in AREA_LBL:
        v = prec[:, :, :, aidx[lbl], midx]
        v = v[v > -1]
        out[f"AP_{lbl}"] = float(v.mean()) if v.size else -1.0
    for lbl in ("small", "<64", "64-256", "256-1024"):
        v = rec[:, :, aidx[lbl], midx]
        v = v[v > -1]
        out[f"AR100_{lbl}"] = float(v.mean()) if v.size else -1.0
    print(f"a_plus_dedup_ranked ({time.time()-t0:.0f}s): "
          f"APs={out['AP_small']:.4f} AP<64={out['AP_<64']:.4f} "
          f"AP64-256={out['AP_64-256']:.4f} AP256-1k={out['AP_256-1024']:.4f} "
          f"AP={out['AP_all']:.4f}", flush=True)
    with open(os.path.join(HERE, "results3.json"), "w") as f:
        json.dump({"a_plus_dedup_ranked": out}, f, indent=1)


if __name__ == "__main__":
    main()
