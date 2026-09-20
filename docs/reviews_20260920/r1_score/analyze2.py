#!/usr/bin/env python3
"""R1 follow-up: corrected dedup semantics + clean upper bounds + FP/rank stats.

Fixes vs analyze.py:
  - dedup keeper set is per-DET (a det survives if it is the best-score IoU>=0.5
    det for AT LEAST ONE GT); analyze.py killed dets that were non-keeper for any
    single GT even if keeper for another (over-kill in dense stacks).
  - b_perfect: matched -> IoU, unmatched -> 0 (true "perfect ranking" upper bound;
    analyze.py's b kept unmatched at original high scores, inverting the top).
  - a_plus_dedup_fixed: keepers -> 1.0, other matched -> -1.
  - c_matched_only: quantile-recalibrate ONLY matched det-area<64 dets onto the
    matched det-area 256-1024 score distribution (oracle scale calibration).
Extra stats:
  - distinct dets per hit GT bucket (no shared-det overcount)
  - per-IoU-thr recall slices (0.5 / 0.75 / 0.95) for base
  - score-separation AUC (matched vs hard-FP) per det bucket
  - rank of the best small-GT det within its image's score-ranked det list
"""
import contextlib
import glob
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


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def rle_of(d):
    s = d["segmentation"]
    c = s["counts"]
    if isinstance(c, str):
        c = c.encode()
    return {"size": s["size"], "counts": c}


def bucket_of_area(a):
    return "<64" if a < 64 else "64-256" if a < 256 else "256-1024" if a < 1024 else ">=1024"


def eval_variant(coco_gt, coco_dt, name, per_thr=False):
    t0 = time.time()
    e = COCOeval(coco_gt, coco_dt, "segm")
    e.params.areaRng = [list(r) for r in AREA_RNG]
    e.params.areaRngLbl = list(AREA_LBL)
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
    for lbl in ("small", "<64", "64-256", "256-1024", "all"):
        v = rec[:, :, aidx[lbl], midx]
        v = v[v > -1]
        out[f"AR100_{lbl}"] = float(v.mean()) if v.size else -1.0
    if per_thr:
        for ti, thr in enumerate(e.params.iouThrs):
            for lbl in ("<64", "64-256", "256-1024", "small"):
                v = rec[ti, :, aidx[lbl], midx]
                v = v[v > -1]
                out[f"R@{thr:.2f}_{lbl}"] = float(v.mean()) if v.size else -1.0
    log(f"  {name}: APs={out['AP_small']:.4f} AP<64={out['AP_<64']:.4f} "
        f"AP64-256={out['AP_64-256']:.4f} AP256-1k={out['AP_256-1024']:.4f} "
        f"AP={out['AP_all']:.4f} ({time.time()-t0:.0f}s)")
    return out


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

    t0 = time.time()
    det_max_iou = np.zeros(len(dets), dtype=np.float32)
    det_arg_gt = np.full(len(dets), -1, dtype=np.int64)
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
            row = iou[li]
            j = int(np.argmax(row))
            det_max_iou[gi] = row[j]
            det_arg_gt[gi] = gts[j]["id"]
        for lj, g in enumerate(gts):
            col = iou[:, lj]
            hits = [(dids[k], float(col[k])) for k in np.nonzero(col >= 0.5)[0]]
            if hits:
                gt_matches.setdefault(g["id"], []).extend(hits)
    log(f"matching done ({time.time()-t0:.0f}s)")

    det_area = np.array([float(maskUtils.area(rle_of(d))) for d in dets])
    det_bucket = np.array([bucket_of_area(a) for a in det_area])
    matched = det_max_iou >= 0.5
    src = np.array([d["score"] for d in dets])
    ann_by_id = {a["id"]: a for a in coco_gt.dataset["annotations"]}

    # keepers: best-score det per GT; a det survives if keeper of >=1 GT
    keeper_of = {}
    for gid, hits in gt_matches.items():
        keeper_of[gid] = max(hits, key=lambda h: src[h[0]])[0]
    keepers = set(keeper_of.values())
    RESULTS = {"n_keepers": len(keepers)}

    # distinct dets per hit GT bucket (no shared-det overcount)
    per_gt = {}
    for b in ("<64", "64-256", "256-1024"):
        gids = [g for g, a in ann_by_id.items() if bucket_of_area(a["area"]) == b]
        hit = [g for g in gids if g in gt_matches]
        distinct = [len(set(di for di, _ in gt_matches[g])) for g in hit]
        # dets whose ARGMAX gt is in this bucket (each det counted once)
        arg_in = [int(((det_arg_gt == g) & matched).sum()) for g in []]  # placeholder
        per_gt[b] = {
            "n_gt": len(gids), "n_hit": len(hit),
            "mean_distinct_dets_per_hit_gt": float(np.mean(distinct)) if distinct else 0.0,
            "frac_hit_gt_ge2_distinct": float(np.mean(np.array(distinct) >= 2)) if distinct else 0.0,
        }
    RESULTS["per_gt_distinct"] = per_gt

    # AUC: score separating matched vs hard-FP within det buckets
    def auc(pos, neg):
        if len(pos) == 0 or len(neg) == 0:
            return None
        # rank-based AUC
        allv = np.concatenate([pos, neg])
        order = np.argsort(allv)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(allv) + 1)
        r_pos = ranks[:len(pos)].sum()
        return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))

    RESULTS["auc"] = {
        b: auc(src[(det_bucket == b) & matched], src[(det_bucket == b) & (det_max_iou < 0.3)])
        for b in ("<64", "64-256", "256-1024")
    }
    # AUC per GT bucket: score of best-score hit det vs... skip (det-bucket version suffices)

    # rank of best small-GT det within its image's score-ranked det list
    rank_stats = {}
    for b in ("<64", "64-256", "256-1024"):
        ranks = []
        for gid, k in keeper_of.items():
            a = ann_by_id[gid]
            if bucket_of_area(a["area"]) != b:
                continue
            img = a["image_id"]
            sc_img = src[dets_by_img[img]]
            r = int((sc_img > src[k]).sum())  # 0-based rank by score
            ranks.append(r)
        ranks = np.array(ranks)
        rank_stats[b] = {
            "median": float(np.median(ranks)) if len(ranks) else None,
            "p90": float(np.percentile(ranks, 90)) if len(ranks) else None,
            "frac_ge_90": float((ranks >= 90).mean()) if len(ranks) else None,
            "n": int(len(ranks)),
        }
    RESULTS["keeper_rank_in_image"] = rank_stats

    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(dets)
    ann_by_det = list(coco_dt.dataset["annotations"])
    orig = [a["score"] for a in ann_by_det]

    def run(name, fn, per_thr=False):
        for i, a in enumerate(ann_by_det):
            a["score"] = fn(i, orig[i])
        RESULTS[name] = eval_variant(coco_gt, coco_dt, name, per_thr=per_thr)
        for a, s in zip(ann_by_det, orig):
            a["score"] = s

    RESULTS.setdefault("variants", {})
    log("== base (per-thr recall) ==")
    run("base_thr", lambda i, s: s, per_thr=True)
    log("== dedup_fixed ==")
    run("dedup_fixed", lambda i, s: s if i in keepers else -1.0)
    log("== a_plus_dedup_fixed ==")
    run("a_plus_dedup_fixed",
        lambda i, s: 1.0 if i in keepers else (-1.0 if matched[i] else s))
    log("== b_perfect (matched->IoU, unmatched->0) ==")
    run("b_perfect", lambda i, s: float(det_max_iou[i]) if matched[i] else 0.0)
    log("== a_separate (matched just above FPs, order kept) ==")
    run("a_separate", lambda i, s: 0.999 + 1e-5 * s if matched[i] else 0.998 * s)
    log("== c_matched_only ==")
    mref = np.sort(src[(det_bucket == "256-1024") & matched])
    idxs = np.nonzero((det_bucket == "<64") & matched)[0]
    order = idxs[np.argsort(src[idxs], kind="stable")]
    cmap = {}
    for r, i in enumerate(order):
        cmap[i] = float(np.quantile(mref, (r + 0.5) / len(order)))
    run("c_matched_only", lambda i, s: cmap.get(i, s))

    with open(os.path.join(HERE, "results2.json"), "w") as f:
        json.dump(RESULTS, f, indent=1)
    log("wrote results2.json")
    print(json.dumps({k: RESULTS[k] for k in
                      ("n_keepers", "per_gt_distinct", "auc", "keeper_rank_in_image")}, indent=1))


if __name__ == "__main__":
    main()
