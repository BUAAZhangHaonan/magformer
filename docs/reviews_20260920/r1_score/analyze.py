#!/usr/bin/env python3
"""R1 review: score formation & ranking quality for AP_s (F1 full model).

Stages (all CPU, single process, dets loaded once):
  1. det<->GT IoU matching per image (pycocotools maskUtils.iou, single class)
  2. baseline COCOeval (segm) with sub-bucket area ranges: <64 / 64-256 / 256-1024
  3. counterfactual rescore variants (scores mutated in place on ONE coco_dt):
       a_matched_to_1   matched (IoU>=0.5 w/ any GT) -> 1.0
       b_score_iou      matched -> its segm IoU with best GT
       c_recalib_lt64   det-area<64 scores quantile-mapped onto det-area 256-1024 dist
       c_recalib_all    all three small det-buckets mapped onto 256-1024 dist
       dedup            per GT keep best-score IoU>=0.5 det, others -> -1
       a_plus_dedup     a_matched_to_1 then per GT keep max-IoU det
       fp_sanitize_lt64 det-area<64 & maxIoU<0.3 -> -1
  4. FP structure & duplicate stats from the matching
  5. score-maturity curve over eval_snapshots history + linear extrapolation

Writes results.json next to this script. Usage:
  CUDA_VISIBLE_DEVICES= python analyze.py
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
DET = os.path.join(RUN, "eval_snapshots/2001_0920/dets.json")  # latest = 112K EMA eval
RESULTS = {}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def bucket_of_area(a):
    if a < 64:
        return "<64"
    if a < 256:
        return "64-256"
    if a < 1024:
        return "256-1024"
    return ">=1024"


AREA_RNG = [
    [0.0, 1e10],
    [0.0, 1024.0],          # COCO small (= union of the three sub-buckets)
    [0.0, 64.0],
    [64.0, 256.0],
    [256.0, 1024.0],
]
AREA_LBL = ["all", "small", "<64", "64-256", "256-1024"]


def rle_of(d):
    s = d["segmentation"]
    c = s["counts"]
    if isinstance(c, str):
        c = c.encode()
    return {"size": s["size"], "counts": c}


def eval_variant(coco_gt, coco_dt, name):
    """Run COCOeval segm on current coco_dt scores; extract AP slices + ARs."""
    t0 = time.time()
    e = COCOeval(coco_gt, coco_dt, "segm")
    e.params.areaRng = [list(r) for r in AREA_RNG]
    e.params.areaRngLbl = list(AREA_LBL)
    e.evaluate()
    e.accumulate()
    prec = e.eval["precision"]  # (T,R,K,A,M)
    rec = e.eval["recall"]      # (T,K,A,M)
    aidx = {lbl: i for i, lbl in enumerate(AREA_LBL)}
    midx = len(e.params.maxDets) - 1  # maxDets=100
    out = {}
    for lbl in AREA_LBL:
        v = prec[:, :, :, aidx[lbl], midx]
        v = v[v > -1]
        out[f"AP_{lbl}"] = float(v.mean()) if v.size else -1.0
    for thr, key in ((0, "AP50"), (5, "AP75")):
        v = prec[thr, :, :, aidx["all"], midx]
        v = v[v > -1]
        out[key] = float(v.mean()) if v.size else -1.0
    for lbl in ("small", "<64", "64-256", "256-1024"):
        v = rec[:, :, aidx[lbl], midx]
        v = v[v > -1]
        out[f"AR100_{lbl}"] = float(v.mean()) if v.size else -1.0
    v = rec[:, :, aidx["all"], midx]
    v = v[v > -1]
    out["AR100_all"] = float(v.mean()) if v.size else -1.0
    log(f"  eval {name}: APs={out['AP_small']:.4f} AP<64={out['AP_<64']:.4f} "
        f"AP64-256={out['AP_64-256']:.4f} AP256-1k={out['AP_256-1024']:.4f} "
        f"AP={out['AP_all']:.4f} ({time.time()-t0:.0f}s)")
    return out


def main():
    log("loading GT + dets ...")
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(GT)
    dets = json.load(open(DET))
    log(f"n_dets={len(dets)}")

    gt_by_img = {}
    for a in coco_gt.dataset["annotations"]:
        gt_by_img.setdefault(a["image_id"], []).append(a)

    # ---- stage 1: matching ----
    t0 = time.time()
    det_max_iou = np.zeros(len(dets), dtype=np.float32)
    det_arg_gt = np.full(len(dets), -1, dtype=np.int64)   # global ann id of best GT
    det_area = np.zeros(len(dets), dtype=np.float64)
    # per GT (global ann id): list of (det_global_idx, iou) with iou>=0.5
    gt_matches = {}
    dets_by_img = {}
    for i, d in enumerate(dets):
        dets_by_img.setdefault(d["image_id"], []).append(i)

    for img_id, gts in gt_by_img.items():
        g_rles = [maskUtils.merge(maskUtils.frPyObjects(g["segmentation"], 1024, 1024))
                  for g in gts]
        dids = dets_by_img.get(img_id, [])
        if not dids or not gts:
            continue
        d_rles = [rle_of(dets[i]) for i in dids]
        iou = maskUtils.iou(d_rles, g_rles, [0] * len(g_rles))  # (n_dt, n_gt)
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

    # det mask areas
    for i, d in enumerate(dets):
        det_area[i] = float(maskUtils.area(rle_of(d)))

    matched = det_max_iou >= 0.5
    RESULTS["match_stats"] = {
        "n_dets": int(len(dets)),
        "n_matched_iou50": int(matched.sum()),
        "det_bucket_counts": {b: int((np.array([bucket_of_area(a) for a in det_area]) == b).sum())
                               for b in ("<64", "64-256", "256-1024", ">=1024")},
    }

    # ---- build ONE coco_dt; keep handle to per-det ann dict ----
    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(dets)
    ann_by_det = list(coco_dt.dataset["annotations"])  # input order == dets order
    assert len(ann_by_det) == len(dets)
    orig_scores = [a["score"] for a in ann_by_det]

    def set_scores(fn):
        for i, a in enumerate(ann_by_det):
            a["score"] = fn(i, a["score"])

    def restore():
        for a, s in zip(ann_by_det, orig_scores):
            a["score"] = s

    # ---- stage 2+3: variants ----
    RESULTS["variants"] = {}
    log("== baseline ==")
    RESULTS["variants"]["base"] = eval_variant(coco_gt, coco_dt, "base")

    log("== (a) matched -> 1.0 ==")
    set_scores(lambda i, s: 1.0 if matched[i] else s)
    RESULTS["variants"]["a_matched_to_1"] = eval_variant(coco_gt, coco_dt, "a")
    restore()

    log("== (b) matched -> IoU ==")
    set_scores(lambda i, s: float(det_max_iou[i]) if matched[i] else s)
    RESULTS["variants"]["b_score_iou"] = eval_variant(coco_gt, coco_dt, "b")
    restore()

    log("== (c) quantile recalibration ==")
    det_bucket = np.array([bucket_of_area(a) for a in det_area])
    src_scores = np.array(orig_scores)
    ref = np.sort(src_scores[det_bucket == "256-1024"])

    def quantile_map(mask):
        idxs = np.nonzero(mask)[0]
        order = idxs[np.argsort(src_scores[idxs], kind="stable")]
        n = len(order)
        new = np.empty(n)
        for r, i in enumerate(order):
            f = (r + 0.5) / n
            new[r] = np.quantile(ref, f)
        return dict(zip(order.tolist(), new.tolist()))

    m64 = quantile_map(det_bucket == "<64")
    set_scores(lambda i, s: m64.get(i, s))
    RESULTS["variants"]["c_recalib_lt64"] = eval_variant(coco_gt, coco_dt, "c_lt64")
    restore()

    mall = quantile_map(det_bucket == "<64")
    mall.update(quantile_map(det_bucket == "64-256"))
    set_scores(lambda i, s: mall.get(i, s))
    RESULTS["variants"]["c_recalib_all"] = eval_variant(coco_gt, coco_dt, "c_all")
    restore()

    log("== oracle dedup (per GT keep best-score IoU>=0.5 det) ==")
    kill = np.zeros(len(dets), dtype=bool)
    dup_counts = []
    for gid, hits in gt_matches.items():
        keep = max(hits, key=lambda h: src_scores[h[0]])[0]
        for di, _ in hits:
            if di != keep:
                kill[di] = True
        dup_counts.append(len(hits))
    RESULTS["dup_stats"] = {
        "n_gt_with_hits": len(gt_matches),
        "mean_dets_per_hit_gt": float(np.mean(dup_counts)) if dup_counts else 0.0,
        "median_dets_per_hit_gt": float(np.median(dup_counts)) if dup_counts else 0.0,
        "frac_gt_multi_det": float(np.mean(np.array(dup_counts) >= 2)) if dup_counts else 0.0,
    }
    set_scores(lambda i, s: -1.0 if kill[i] else s)
    RESULTS["variants"]["dedup"] = eval_variant(coco_gt, coco_dt, "dedup")
    restore()

    log("== (a) + oracle dedup (per GT keep max-IoU det at 1.0) ==")
    keep_map = {}
    for gid, hits in gt_matches.items():
        keep = max(hits, key=lambda h: h[1])[0]
        keep_map[keep] = True

    def a_dedup_fn(i, s):
        if i in keep_map:
            return 1.0
        if matched[i]:
            return -1.0
        return s
    set_scores(a_dedup_fn)
    RESULTS["variants"]["a_plus_dedup"] = eval_variant(coco_gt, coco_dt, "a+dedup")
    restore()

    log("== fp sanitize <64 (maxIoU<0.3 -> kill) ==")
    set_scores(lambda i, s: -1.0 if (det_bucket[i] == "<64" and det_max_iou[i] < 0.3) else s)
    RESULTS["variants"]["fp_sanitize_lt64"] = eval_variant(coco_gt, coco_dt, "fp_sanitize")
    restore()

    # ---- stage 4: FP structure ----
    fp_stats = {}
    for b in ("<64", "64-256", "256-1024"):
        m = det_bucket == b
        tp = m & matched
        fp_hard = m & (det_max_iou < 0.3)
        fp_soft = m & ~matched & ~fp_hard  # 0.3<=IoU<0.5
        sc = src_scores
        fp_stats[b] = {
            "n_dets": int(m.sum()),
            "frac_hard_fp": float(fp_hard.mean() if m.any() else 0),
            "frac_tp": float(tp.mean() if m.any() else 0),
            "tp_score_median": float(np.median(sc[tp])) if tp.any() else None,
            "tp_score_p25": float(np.percentile(sc[tp], 25)) if tp.any() else None,
            "tp_score_p75": float(np.percentile(sc[tp], 75)) if tp.any() else None,
            "hard_fp_score_median": float(np.median(sc[fp_hard])) if fp_hard.any() else None,
            "hard_fp_score_p90": float(np.percentile(sc[fp_hard], 90)) if fp_hard.any() else None,
            "n_hard_fp_above_0.5": int((sc[fp_hard] > 0.5).sum()) if fp_hard.any() else 0,
            "n_tp_below_0.5": int((sc[tp] <= 0.5).sum()) if tp.any() else 0,
            "frac_hard_fp_score_gt_tp_median": (
                float((sc[fp_hard] > np.median(sc[tp])).mean()) if fp_hard.any() and tp.any() else None),
            "soft_fp_frac": float(fp_soft.mean() if m.any() else 0),
        }
    RESULTS["fp_structure"] = fp_stats

    # per-GT-bucket stats: dup + tp score (two definitions)
    ann_by_id = {a["id"]: a for a in coco_gt.dataset["annotations"]}
    per_gt = {}
    for b in ("<64", "64-256", "256-1024"):
        gids = [gid for gid, a in ann_by_id.items() if bucket_of_area(a["area"]) == b]
        hit = [g for g in gids if g in gt_matches]
        dups = [len(gt_matches[g]) for g in hit]
        # TP score: best-IoU det among hits (subbucket.py style uses best-IoU unused)
        best_iou_sc, best_score_sc = [], []
        for g in hit:
            hs = gt_matches[g]
            best_iou_sc.append(src_scores[max(hs, key=lambda h: h[1])[0]])
            best_score_sc.append(src_scores[max(hs, key=lambda h: src_scores[h[0]])[0]])
        per_gt[b] = {
            "n_gt": len(gids),
            "n_gt_with_iou50_det": len(hit),
            "recall_any": float(len(hit) / max(len(gids), 1)),
            "mean_dets_per_hit_gt": float(np.mean(dups)) if dups else 0.0,
            "frac_gt_ge2_dets": float(np.mean(np.array(dups) >= 2)) if dups else 0.0,
            "tp_score_median_bestIoU": float(np.median(best_iou_sc)) if best_iou_sc else None,
            "tp_score_median_bestScore": float(np.median(best_score_sc)) if best_score_sc else None,
            "n_hit_gt_best_det_score_lt_0.5": int(np.sum(np.array(best_iou_sc) < 0.5)) if best_iou_sc else 0,
        }
    RESULTS["per_gt_bucket"] = per_gt

    # hard-FP score distribution detail for <64
    m = (det_bucket == "<64") & (det_max_iou < 0.3)
    if m.any():
        RESULTS["fp_structure"]["<64"]["hard_fp_score_percentiles"] = {
            str(q): float(np.percentile(src_scores[m], q)) for q in (50, 75, 90, 95, 99)
        }

    # ---- stage 5: maturity curve ----
    val_rows = []
    for line in open(os.path.join(RUN, "metrics_log.jsonl")):
        d = json.loads(line)
        if d.get("phase") == "val":
            val_rows.append(d)
    snaps = []
    for sd in sorted(glob.glob(os.path.join(RUN, "eval_snapshots", "*"))):
        dj = os.path.join(sd, "dets.json")
        if not os.path.exists(dj):
            continue
        mt = os.path.getmtime(dj)
        # snapshot dets.json is a copy made shortly AFTER the eval that produced it;
        # val row wall_time = eval end. Take nearest val row by |wall_time - mtime|
        # (copy lag ~minutes; eval spacing ~3.9h, so unambiguous).
        step = min(val_rows, key=lambda d: abs(d["wall_time"] - mt))["iter"]
        snaps.append({"dir": os.path.basename(sd), "mtime": mt, "step": step})
    maturity = []
    for s in snaps:
        sb = os.path.join(RUN, "eval_snapshots", s["dir"], "subbucket.json")
        if not os.path.exists(sb):
            continue
        b = json.load(open(sb)).get("buckets", {})
        maturity.append({
            "step": s["step"], "tag": s["dir"],
            **{k: b.get(k, {}).get("tp_score_median") for k in ("<64", "64-256", "256-1024")},
            **{f"recall_{k}": b.get(k, {}).get("recall@0.5") for k in ("<64", "64-256", "256-1024")},
        })
    maturity.sort(key=lambda r: (r["step"] is None, r["step"]))
    RESULTS["maturity"] = maturity

    # linear fits: full history and last 5 evals, extrapolate to 128K/256K
    fits = {}
    for key in ("<64", "64-256", "256-1024"):
        pts = [(r["step"], r[key]) for r in maturity if r["step"] and r[key] is not None]
        if len(pts) < 3:
            continue
        x = np.array([p[0] for p in pts], dtype=float)
        y = np.array([p[1] for p in pts])
        full = np.polyfit(x, y, 1)
        last5 = np.polyfit(x[-5:], y[-5:], 1)
        fits[key] = {
            "n_pts": len(pts),
            "slope_per_8k_full": float(full[0] * 8000),
            "slope_per_8k_last5": float(last5[0] * 8000),
            "extrap_128k_full": float(np.polyval(full, 128000)),
            "extrap_256k_full": float(np.polyval(full, 256000)),
            "extrap_128k_last5": float(np.polyval(last5, 128000)),
            "extrap_256k_last5": float(np.polyval(last5, 256000)),
        }
    RESULTS["maturity_fits"] = fits

    # AP_s trend from trainer log (contract offset is constant; deltas valid)
    aps_trend = [(d["iter"], d["val/segm_APs"]) for d in val_rows]
    RESULTS["aps_trend"] = aps_trend
    if len(aps_trend) >= 6:
        x = np.array([a[0] for a in aps_trend], dtype=float)
        y = np.array([a[1] for a in aps_trend])
        f_all = np.polyfit(x, y, 1)
        f_l5 = np.polyfit(x[-5:], y[-5:], 1)
        RESULTS["aps_trend_fits"] = {
            "delta_per_8k_full": float(f_all[0] * 8000),
            "delta_per_8k_last5": float(f_l5[0] * 8000),
            "extrap_256k_full": float(np.polyval(f_all, 256000)),
            "extrap_256k_last5": float(np.polyval(f_l5, 256000)),
            "extrap_128k_last5": float(np.polyval(f_l5, 128000)),
        }

    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(RESULTS, f, indent=1)
    log("wrote results.json")
    print(json.dumps({k: RESULTS[k] for k in ("dup_stats", "maturity_fits", "aps_trend_fits")
                      if k in RESULTS}, indent=1))


if __name__ == "__main__":
    main()
