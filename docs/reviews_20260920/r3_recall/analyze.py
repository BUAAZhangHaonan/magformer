#!/usr/bin/env python3
"""R3 review analysis: recall structure / query capacity / dedup for AP_s.

Run (CPU only):
  CUDA_VISIBLE_DEVICES= /home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python analyze.py

Inputs (read-once):
  DETS: latest eval dets (coco_instances_results.json of the live F1 run)
  GT:   instances_val.validated.json (3276 images)

Outputs: results.json next to this script (written incrementally).
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
OUT = os.path.join(HERE, "results.json")

results = {}


def save():
    with open(OUT, "w") as f:
        json.dump(results, f, indent=1, default=float)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


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


# ---------------------------------------------------------------- GT load
log("loading GT ...")
t0 = time.time()
with contextlib.redirect_stdout(io.StringIO()):
    coco = COCO(GT)
log(f"GT loaded in {time.time()-t0:.0f}s")

img_hw = {im["id"]: (im["height"], im["width"]) for im in coco.dataset["images"]}
hs = sorted(set(h for h, _ in img_hw.values()))
ws = sorted(set(w for _, w in img_hw.values()))

gt_by_img = defaultdict(list)
for a in coco.dataset["annotations"]:
    gt_by_img[a["image_id"]].append(a)   # annotation order == matcher order
n_imgs = len(coco.dataset["images"])
results["gt_basic"] = {
    "n_images": n_imgs, "n_annotations": len(coco.dataset["annotations"]),
    "image_heights": hs, "image_widths": ws,
}
log(f"{n_imgs} images, {len(coco.dataset['annotations'])} anns")

# per-image small-GT count distributions -------------------------------
per_img_small1024, per_img_aim4096, per_img_gt_all = [], [], []
for img_id, anns in gt_by_img.items():
    per_img_small1024.append(sum(1 for a in anns if 0 < a["area"] < 1024))
    per_img_aim4096.append(sum(1 for a in anns if 0 < a["area"] < 4096))
    per_img_gt_all.append(len(anns))


def pct(x, q):
    return float(np.percentile(np.asarray(x), q)) if len(x) else None


results["per_img_counts"] = {
    "small<1024": {"mean": float(np.mean(per_img_small1024)),
                   "p50": pct(per_img_small1024, 50),
                   "p90": pct(per_img_small1024, 90),
                   "p99": pct(per_img_small1024, 99),
                   "max": int(np.max(per_img_small1024)),
                   "n_img_ge1": int(np.sum(np.asarray(per_img_small1024) >= 1)),
                   "n_img_ge20": int(np.sum(np.asarray(per_img_small1024) >= 20))},
    "aim<4096": {"mean": float(np.mean(per_img_aim4096)),
                 "p50": pct(per_img_aim4096, 50),
                 "p90": pct(per_img_aim4096, 90),
                 "p99": pct(per_img_aim4096, 99),
                 "max": int(np.max(per_img_aim4096)),
                 "n_img_ge1": int(np.sum(np.asarray(per_img_aim4096) >= 1))},
    "gt_all": {"mean": float(np.mean(per_img_gt_all)),
               "p50": pct(per_img_gt_all, 50),
               "p90": pct(per_img_gt_all, 90),
               "max": int(np.max(per_img_gt_all))},
}
save()

# AIM sel[:16] coverage --------------------------------------------------
aim_total = aim_capped = imgs_with_cap = 0
small_total = small_beyond_cap = 0
small_beyond_cap_sub = {"<64": 0, "<32": 0}
for img_id, anns in gt_by_img.items():
    elig = [a for a in anns if 0 < a["area"] < 4096]
    aim_total += len(elig)
    if len(elig) > 16:
        imgs_with_cap += 1
        aim_capped += len(elig) - 16
        for a in elig[16:]:
            if a["area"] < 1024:
                small_beyond_cap += 1
                if a["area"] < 64:
                    small_beyond_cap_sub["<64"] += 1
                if a["area"] < 32:
                    small_beyond_cap_sub["<32"] += 1
    small_total += sum(1 for a in anns if 0 < a["area"] < 1024)
results["aim_cap16"] = {
    "gt_lt4096_total": aim_total,
    "gt_lt4096_beyond_cap": aim_capped,
    "frac_lt4096_beyond_cap": aim_capped / max(aim_total, 1),
    "n_images_exceeding_cap": imgs_with_cap,
    "coco_small_total": small_total,
    "coco_small_beyond_cap": small_beyond_cap,
    "frac_coco_small_beyond_cap": small_beyond_cap / max(small_total, 1),
    "coco_small_beyond_cap_subbuckets": small_beyond_cap_sub,
}
log(f"AIM cap16: {aim_capped}/{aim_total} (<4096) beyond cap; "
    f"coco-small beyond {small_beyond_cap}/{small_total}; imgs>{16}: {imgs_with_cap}")
save()

# ---------------------------------------------------------------- load dets
log("loading dets (read once) ...")
t0 = time.time()
dets = json.load(open(DETS))
log(f"dets loaded in {time.time()-t0:.0f}s, {len(dets)} rows")
results["n_dets"] = len(dets)

dets_by_img = defaultdict(list)
for d in dets:
    dets_by_img[d["image_id"]].append(d)

img_ids_sorted = sorted(img_hw)
det_counts = [len(dets_by_img.get(i, [])) for i in img_ids_sorted]
results["per_img_dets"] = {"mean": float(np.mean(det_counts)),
                           "p50": pct(det_counts, 50),
                           "p90": pct(det_counts, 90),
                           "max": int(np.max(det_counts)),
                           "n_img_at_100cap": int(np.sum(np.asarray(det_counts) >= 100))}
save()

# ------------------------------------------------- hard recall & match stats
log("per-image IoU analysis (small GT vs all dets) ...")
t0 = time.time()
subgroups = {"<32": (0, 32), "<64": (0, 64), "64-256": (64, 256),
             "256-1024": (256, 1024), "small<1024": (0, 1024)}
best_iou = {k: [] for k in subgroups}
dets_per_gt = {k: [] for k in subgroups}
score_of_best_hit = {k: [] for k in subgroups}
gt_hit_03 = {k: 0 for k in subgroups}
n_done = 0
for img_id, anns in gt_by_img.items():
    h, w = img_hw[img_id]
    dds = dets_by_img.get(img_id, [])
    sc_arr = np.asarray([d["score"] for d in dds]) if dds else np.zeros(0)
    det_rles = [det_rle(d) for d in dds]
    gt_small = [a for a in anns if 0 < a["area"] < 1024]
    iou_mat = None
    if det_rles and gt_small:
        gt_rles = [gt_rle(a, h, w) for a in gt_small]
        iou_mat = maskUtils.iou(det_rles, gt_rles, [0] * len(gt_rles))  # (D,G)
    for j, a in enumerate(gt_small):
        col = iou_mat[:, j] if iou_mat is not None else np.zeros(0)
        if col.size:
            bi = int(np.argmax(col))
            bv = float(col[bi])
        else:
            bi, bv = -1, 0.0
        for k, (lo, hi) in subgroups.items():
            if lo <= a["area"] < hi:
                best_iou[k].append(bv)
                if bv >= 0.3:
                    gt_hit_03[k] += 1
                if bv >= 0.5 and bi >= 0:
                    score_of_best_hit[k].append(float(dds[bi]["score"]))
                if col.size:
                    thr = (col >= 0.5) & (sc_arr > 0.05)
                    dets_per_gt[k].append(int(np.sum(thr)))
                else:
                    dets_per_gt[k].append(0)
    n_done += 1
    if n_done % 800 == 0:
        log(f"  ... {n_done}/{n_imgs} images ({time.time()-t0:.0f}s)")

rec = {}
for k in subgroups:
    v = np.asarray(best_iou[k])
    dp = np.asarray(dets_per_gt[k])
    sc = score_of_best_hit[k]
    rec[k] = {
        "n": int(v.size),
        "never_hit@0.5": float(np.mean(v < 0.5)) if v.size else None,
        "never_hit@0.3": float(np.mean(v < 0.3)) if v.size else None,
        "best_iou_median": float(np.median(v)) if v.size else None,
        "best_iou_p90": pct(v, 90) if v.size else None,
        "dets_per_gt_mean": float(dp.mean()) if dp.size else None,
        "dets_per_gt_p90": pct(dp, 90) if dp.size else None,
        "dets_per_gt_max": int(dp.max()) if dp.size else None,
        "frac_gt_multi_det(>=2)": float(np.mean(dp >= 2)) if dp.size else None,
        "frac_gt_zero_det": float(np.mean(dp == 0)) if dp.size else None,
        "top1_hit_score_median": float(np.median(sc)) if sc else None,
    }
results["hard_recall"] = rec
log(f"IoU pass done in {time.time()-t0:.0f}s")
save()

# ------------------------------------------------------------- COCOeval runs
log("COCOeval full (md 1/10/100 in one pass) ...")


def eval_stats(coco_dt, img_ids=None):
    e = COCOeval(coco, coco_dt, iouType="segm")
    if img_ids is not None:
        e.params.imgIds = sorted(img_ids)
    e.params.maxDets = [1, 10, 100]
    with contextlib.redirect_stdout(io.StringIO()):
        e.evaluate()
        e.accumulate()
        e.summarize()
    p = e.stats
    r = e.eval["recall"]      # (T,K,A,M)
    prc = e.eval["precision"]  # (T,R,K,A,M)
    prec05_small = prc[0, :, 0, 1, 2]   # iou .5, cat0, area small, md100 slot
    defined = np.where(prec05_small > -1 + 1e-9)[0]
    rec_thrs = e.params.recThrs
    out = {
        "AP": float(p[0]), "AP50": float(p[1]), "AP75": float(p[2]),
        "AP_small": float(p[3]), "AP_medium": float(p[4]), "AP_large": float(p[5]),
        "AR@1_all": float(p[6]), "AR@10_all": float(p[7]), "AR@100_all": float(p[8]),
        "AR_small@1": float(np.nanmean(r[:, :, 1, 0])),
        "AR_small@10": float(np.nanmean(r[:, :, 1, 1])),
        "AR_small@100": float(np.nanmean(r[:, :, 1, 2])),
        "AR_med@100": float(np.nanmean(r[:, :, 2, 2])),
        "AR_large@100": float(np.nanmean(r[:, :, 3, 2])),
        "small_recall_ceiling@0.5(md100)": float(rec_thrs[defined.max()]) if defined.size else 0.0,
        "small_prec_curve@0.5": [float(x) for x in prec05_small[defined]],
    }
    return out


t0 = time.time()
coco_dt = coco.loadRes(dets)
results["eval_full"] = eval_stats(coco_dt)
log(f"full eval {time.time()-t0:.0f}s: {results['eval_full']['AP']:.4f}/"
    f"{results['eval_full']['AP_small']:.4f}")
save()

# dense vs sparse --------------------------------------------------------
dense_ids = [i for i, c in zip(img_ids_sorted, per_img_small1024) if c >= 20]
sparse_ids = [i for i, c in zip(img_ids_sorted, per_img_small1024) if 1 <= c < 20]
results["subsets"] = {
    "dense(n_small>=20)": {
        "n_img": len(dense_ids),
        "n_small_gt": int(sum(c for c in per_img_small1024 if c >= 20)),
        "frac_small_gt": float(sum(c for c in per_img_small1024 if c >= 20)
                                / max(sum(per_img_small1024), 1))},
    "sparse(1..19)": {
        "n_img": len(sparse_ids),
        "n_small_gt": int(sum(c for c in per_img_small1024 if 1 <= c < 20)),
        "frac_small_gt": float(sum(c for c in per_img_small1024 if 1 <= c < 20)
                                / max(sum(per_img_small1024), 1))},
}
log(f"dense imgs {len(dense_ids)} ({results['subsets']['dense(n_small>=20)']['n_small_gt']} small GT), "
    f"sparse imgs {len(sparse_ids)}")
results["eval_dense"] = eval_stats(coco_dt, dense_ids)
save()
results["eval_sparse"] = eval_stats(coco_dt, sparse_ids)
save()

# oracle dedup ------------------------------------------------------------
log("oracle dedup (per small GT keep best det) ...")
t0 = time.time()
drop_objs = set()
for img_id, anns in gt_by_img.items():
    dds = dets_by_img.get(img_id, [])
    gt_small = [a for a in anns if 0 < a["area"] < 1024]
    if not dds or not gt_small:
        continue
    h, w = img_hw[img_id]
    dr = [det_rle(d) for d in dds]
    gr = [gt_rle(a, h, w) for a in gt_small]
    iou_mat = maskUtils.iou(dr, gr, [0] * len(gr))
    best_gt = np.argmax(iou_mat, axis=1)
    best_val = iou_mat[np.arange(len(dds)), best_gt]
    matched = best_val >= 0.5
    if not matched.any():
        continue
    keep = set()
    for g in np.unique(best_gt[matched]):
        same = np.where((best_gt == g) & matched)[0]
        sc = [dds[k]["score"] for k in same]
        keep.add(int(same[int(np.argmax(sc))]))
    for j in range(len(dds)):
        if matched[j] and j not in keep:
            drop_objs.add(id(dds[j]))
log(f"dedup pass {time.time()-t0:.0f}s, dropping {len(drop_objs)} dets")
dets_dedup = [d for d in dets if id(d) not in drop_objs]
results["oracle_dedup"] = {"n_dropped": len(drop_objs), "n_kept": len(dets_dedup)}
coco_dt2 = coco.loadRes(dets_dedup)
results["oracle_dedup"]["eval"] = eval_stats(coco_dt2)
save()

# duplicate clusters (box IoU >= 0.5, score>0.05) --------------------------
log("duplicate clusters (box IoU) ...")
t0 = time.time()
cluster_sizes = []
frac_dets_dup = []
for img_id in img_ids_sorted:
    dds = [d for d in dets_by_img.get(img_id, []) if d["score"] > 0.05]
    n = len(dds)
    if n < 2:
        continue
    b = np.asarray([d["bbox"] for d in dds])
    x1, y1 = b[:, 0], b[:, 1]
    x2, y2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    ix = np.maximum(0, np.minimum(x2[:, None], x2[None, :]) - np.maximum(x1[:, None], x1[None, :]))
    iy = np.maximum(0, np.minimum(y2[:, None], y2[None, :]) - np.maximum(y1[:, None], y1[None, :]))
    inter = ix * iy
    areas = b[:, 2] * b[:, 3]
    union = areas[:, None] + areas[None, :] - inter
    iou = np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)
    np.fill_diagonal(iou, 0.0)
    dup = iou >= 0.5
    deg = dup.sum(axis=1)
    seen = np.zeros(n, dtype=bool)
    for i in range(n):
        if not seen[i]:
            stack, comp = [i], 0
            seen[i] = True
            while stack:
                u = stack.pop()
                comp += 1
                for v in np.where(dup[u] & ~seen)[0]:
                    seen[v] = True
                    stack.append(int(v))
            cluster_sizes.append(comp)
    frac_dets_dup.append(float(np.mean(deg > 0)))
results["dup_clusters"] = {
    "n_clusters": len(cluster_sizes),
    "cluster_size_mean": float(np.mean(cluster_sizes)),
    "cluster_size_p90": pct(cluster_sizes, 90),
    "cluster_size_max": int(np.max(cluster_sizes)) if cluster_sizes else None,
    "frac_clusters_ge2": float(np.mean(np.asarray(cluster_sizes) >= 2)) if cluster_sizes else None,
    "frac_dets_with_dup_overlap": float(np.mean(frac_dets_dup)) if frac_dets_dup else None,
}
log(f"cluster pass {time.time()-t0:.0f}s")
save()
log("ALL DONE")
