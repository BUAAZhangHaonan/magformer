#!/usr/bin/env python3
"""R2 review analysis: small-target mask quality & IoU-threshold structure.

Stages (all CPU, dets.json loaded exactly once):
  A. Per-bucket AP at single IoU thresholds (segm + bbox), custom areaRngs.
  B. Matched det-GT pair mask-IoU distribution per bucket (score-greedy
     matching), plus per-GT max-IoU ceiling; box-IoU on the same pairs.
  C. bbox-vs-segm per-bucket delta (from A) + matched-pair box/mask IoU gap
     (from B) -> separates "localization wrong" from "rendering blurry".
  D. Rendering-chain simulation on GT small masks: 1024 ->(area/maxpool)-> 512
     ->(bilinear)-> 1024 -> binarize at sweep of thresholds; IoU vs GT.
  E. Oracle AP_s(t) on a 1200-image subset: GT-as-detection with simulated
     masks at render threshold t in {0.4,0.45,0.5,0.55} (+maxpool@0.5).
  F. Morphological +-1px probe on production matched small masks.

Writes results.json next to this script and prints a digest.
"""
import contextlib
import io
import json
import os
import sys
import time
from collections import defaultdict

# CPU only, be polite to the training box.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import cv2
import numpy as np
import torch

cv2.setNumThreads(8)
torch.set_num_threads(8)

from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from pycocotools.cocoeval import COCOeval
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
GT_PATH = (
    "/home/hdd3/zhanghaonan/magformer/magformer_datasets/20260318_1K_32254/"
    "annotations/instances_val.validated.json"
)
DETS = (
    "/home/hdd3/zhanghaonan/magformer/output/aps_20260913/p5_runs/"
    "f1_full_design_256k/eval_snapshots/2001_0920/dets.json"
)

THRS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
SINGLE_THRS = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
AREA_RNGS = [
    [0, 1e10],
    [0, 64],
    [64, 256],
    [256, 1024],
    [0, 1024],
]
AREA_LBLS = ["all", "<64", "64-256", "256-1024", "small(0-1024)"]


def bucket_of(area):
    if area < 64:
        return "<64"
    if area < 256:
        return "64-256"
    if area < 1024:
        return "256-1024"
    return None


def as_rle(seg):
    if isinstance(seg["counts"], str):
        seg = {"size": seg["size"], "counts": seg["counts"].encode()}
    return seg


def gt_rle(ann):
    seg = ann["segmentation"]
    if isinstance(seg, list):
        return maskUtils.merge(maskUtils.frPyObjects(seg, 1024, 1024))
    return as_rle(seg)


def hist_stats(vals, edges=None):
    v = np.asarray(vals, dtype=float)
    out = {
        "n": int(v.size),
        "mean": float(v.mean()) if v.size else None,
        "p25": float(np.percentile(v, 25)) if v.size else None,
        "p50": float(np.percentile(v, 50)) if v.size else None,
        "p75": float(np.percentile(v, 75)) if v.size else None,
        "frac>=0.5": float((v >= 0.5).mean()) if v.size else None,
        "frac>=0.75": float((v >= 0.75).mean()) if v.size else None,
        "frac>=0.9": float((v >= 0.9).mean()) if v.size else None,
    }
    if edges is None:
        edges = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    h, _ = np.histogram(v, bins=edges)
    out["hist"] = {f"[{edges[i]:.1f},{edges[i+1]:.1f})": int(h[i]) for i in range(len(h))}
    return out


def iou_masks(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / union if union else 0.0


def main():
    t0 = time.time()
    results = {"dets_file": DETS, "gt_file": GT_PATH}

    print(f"[load] dets ...", flush=True)
    dets = json.load(open(DETS))
    print(f"[load] dets rows={len(dets)}  ({time.time()-t0:.0f}s)", flush=True)
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(GT_PATH)
    gt_by_img = defaultdict(list)
    for a in coco.dataset["annotations"]:
        gt_by_img[a["image_id"]].append(a)
    n_cats = len(coco.getCatIds())
    print(f"[load] gt imgs={len(gt_by_img)} cats={n_cats} anns={len(coco.dataset['annotations'])}", flush=True)

    # ---------------- Stage A: bucket x IoU-threshold AP ----------------
    def run_eval(iou_type):
        coco_dt = coco.loadRes([dict(d) for d in dets])
        e = COCOeval(coco, coco_dt, iouType=iou_type)
        e.params.iouThrs = np.array(THRS)
        e.params.areaRng = AREA_RNGS
        e.params.areaRngLbl = AREA_LBLS
        e.params.maxDets = [1, 10, 100]
        e.evaluate()
        e.accumulate()
        P = e.eval["precision"]  # [T, R, K, A, M]
        table = {}
        for ai, lbl in enumerate(AREA_LBLS):
            row = {}
            for ti, thr in enumerate(THRS):
                q = P[ti, :, :, ai, 2]
                q = q[q > -1]
                row[f"@{thr:.2f}"] = float(q.mean()) if q.size else None
            std = [row[f"@{t:.2f}"] for t in THRS]
            row["AP[.5:.95]"] = float(np.mean([x for x in std if x is not None]))
            table[lbl] = row
        return table

    print("[A] COCOeval segm ...", flush=True)
    results["A_ap_segm"] = run_eval("segm")
    print(json.dumps(results["A_ap_segm"]["small(0-1024)"], indent=1), flush=True)
    print("[A] COCOeval bbox ...", flush=True)
    results["A_ap_bbox"] = run_eval("bbox")

    # ---------------- Stage B: matched pairs ----------------
    print("[B] matched-pair IoU ...", flush=True)
    dets_by_img = defaultdict(list)
    for d in dets:
        dets_by_img[d["image_id"]].append(d)

    matched = defaultdict(list)   # bucket -> list of dict(mask_iou, box_iou, score)
    ceiling = defaultdict(list)   # bucket -> per-GT max mask IoU over all dets
    n_gt = defaultdict(int)
    for img_id, anns in gt_by_img.items():
        dds = sorted(dets_by_img.get(img_id, []), key=lambda d: -d["score"])[:200]
        if not dds:
            for a in anns:
                b = bucket_of(a["area"])
                if b:
                    n_gt[b] += 1
                    ceiling[b].append(0.0)
            continue
        det_rles = [as_rle(d["segmentation"]) for d in dds]
        gt_rles = [gt_rle(a) for a in anns]
        crowds = [int(a.get("iscrowd", 0)) for a in anns]
        iou_m = maskUtils.iou(det_rles, gt_rles, crowds)  # [n_det, n_gt]
        # box IoU matrix
        db = np.array([d["bbox"] for d in dds], dtype=float)
        gb = np.array([a["bbox"] for a in anns], dtype=float)
        ix = np.maximum(0, np.minimum(db[:, None, 0] + db[:, None, 2], gb[None, :, 0] + gb[None, :, 2]) - np.maximum(db[:, None, 0], gb[None, :, 0]))
        iy = np.maximum(0, np.minimum(db[:, None, 1] + db[:, None, 3], gb[None, :, 1] + gb[None, :, 3]) - np.maximum(db[:, None, 1], gb[None, :, 1]))
        inter = ix * iy
        union = db[:, None, 2] * db[:, None, 3] + gb[None, :, 2] * gb[None, :, 3] - inter
        box_iou_m = np.where(union > 0, inter / union, 0.0)

        used = np.zeros(len(anns), dtype=bool)
        # per-GT ceiling over all dets
        for j, a in enumerate(anns):
            b = bucket_of(a["area"])
            if b:
                n_gt[b] += 1
                ceiling[b].append(float(iou_m[:, j].max()) if iou_m.size else 0.0)
        # score-greedy matching (COCO-style)
        for i in range(len(dds)):
            cand = iou_m[i].copy()
            cand[used] = -1
            j = int(cand.argmax())
            if cand[j] >= 0.5:
                used[j] = True
                a = anns[j]
                b = bucket_of(a["area"])
                if b:
                    matched[b].append({
                        "mask_iou": float(iou_m[i, j]),
                        "box_iou": float(box_iou_m[i, j]),
                        "score": float(dds[i]["score"]),
                    })

    results["B_n_gt"] = dict(n_gt)
    results["B_matched_iou"] = {b: hist_stats([m["mask_iou"] for m in v]) for b, v in matched.items()}
    results["B_matched_box_iou"] = {b: hist_stats([m["box_iou"] for m in v]) for b, v in matched.items()}
    results["B_matched_score"] = {b: hist_stats([m["score"] for m in v]) for b, v in matched.items()}
    results["B_ceiling_iou"] = {b: hist_stats(v) for b, v in ceiling.items()}
    # matched pairs kept for stage F
    results["B_matched_recall_at_0.5"] = {
        b: len(v) / max(n_gt[b], 1) for b, v in matched.items()
    }

    # ---------------- Stage D: rendering-chain simulation on GT ----------------
    print("[D] GT rendering simulation ...", flush=True)
    RENDER_THRS = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
    caps = {"<32": 600, "32-64": 600, "64-128": 400, "128-256": 400, "256-1024": 400}
    counts = defaultdict(int)
    sim = defaultdict(lambda: defaultdict(list))  # subbucket -> method@thr -> [iou]
    sim_areas = defaultdict(list)
    for img_id, anns in gt_by_img.items():
        for a in anns:
            ar = a["area"]
            if ar >= 1024:
                continue
            sb = ("<32" if ar < 32 else "32-64" if ar < 64 else "64-128" if ar < 128 else "128-256" if ar < 256 else "256-1024")
            if counts[sb] >= caps[sb]:
                continue
            g = maskUtils.decode(gt_rle(a)).astype(np.float32)
            # canvas 512 via area interpolation (== E3 "optimal logit")
            lo_area = cv2.resize(g, (512, 512), interpolation=cv2.INTER_AREA) / 4.0
            lo_max = g.reshape(512, 2, 512, 2).max(axis=(1, 3))
            for name, lo in (("area", lo_area), ("maxpool", lo_max.astype(np.float32))):
                up = cv2.resize(lo, (1024, 1024), interpolation=cv2.INTER_LINEAR)
                for t in RENDER_THRS:
                    pred = up > t
                    sim[sb][f"{name}@{t:.2f}"].append(iou_masks(pred, g > 0.5))
            counts[sb] += 1
            sim_areas[sb].append(float(ar))
            if all(counts[k] >= caps[k] for k in caps):
                break
        if all(counts[k] >= caps[k] for k in caps):
            break
    results["D_counts"] = dict(counts)
    results["D_sim_iou"] = {sb: {k: hist_stats(v, edges=[0, 0.3, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.01]) for k, v in d.items()} for sb, d in sim.items()}
    results["D_subgroup_merge"] = {
        "<128": {k: hist_stats(sim["<32"][k] + sim["32-64"][k] + sim["64-128"][k], edges=[0, 0.3, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.01]) for k in sim["<32"]},
        "<64": {k: hist_stats(sim["<32"][k] + sim["32-64"][k], edges=[0, 0.3, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.01]) for k in sim["<32"]},
    }

    # ---------------- Stage E: oracle AP_s(t) on subset ----------------
    print("[E] oracle AP_s(t) subset-1200 ...", flush=True)
    ORACLE_THRS = [0.4, 0.45, 0.5, 0.55]
    img_ids = list(gt_by_img.keys())[:1200]
    struct = np.ones((3, 3), dtype=bool)

    def oracle_eval(rows):
        coco_dt = coco.loadRes(rows)
        e = COCOeval(coco, coco_dt, iouType="segm")
        e.params.imgIds = img_ids
        e.params.areaRng = AREA_RNGS
        e.params.areaRngLbl = AREA_LBLS
        e.params.maxDets = [1, 10, 100]
        e.evaluate()
        e.accumulate()
        P = e.eval["precision"]
        out = {}
        for ai, lbl in enumerate(AREA_LBLS):
            if lbl == "all":
                continue
            q = P[:, :, :, ai, 2]
            q = q[q > -1]
            out[lbl] = float(q.mean()) if q.size else None
        return out

    oracle = {}
    for t in ORACLE_THRS:
        rows = []
        for img_id in img_ids:
            for a in gt_by_img[img_id]:
                g = maskUtils.decode(gt_rle(a)).astype(np.float32)
                lo = cv2.resize(g, (512, 512), interpolation=cv2.INTER_AREA) / 4.0
                up = cv2.resize(lo, (1024, 1024), interpolation=cv2.INTER_LINEAR)
                rle = maskUtils.encode(np.asfortranarray((up > t).astype(np.uint8)))
                rows.append({
                    "image_id": img_id,
                    "category_id": a["category_id"],
                    "bbox": list(a["bbox"]),
                    "score": 0.99,
                    "segmentation": {"size": rle["size"], "counts": rle["counts"].decode("latin-1")},
                })
        oracle[f"area@{t:.2f}"] = oracle_eval(rows)
        print(f"[E] area@{t:.2f}: {json.dumps(oracle[f'area@{t:.2f}'])}", flush=True)
        del rows
    # maxpool variant at 0.5 only
    rows = []
    for img_id in img_ids:
        for a in gt_by_img[img_id]:
            g = maskUtils.decode(gt_rle(a)).astype(np.float32)
            lo = g.reshape(512, 2, 512, 2).max(axis=(1, 3)).astype(np.float32)
            up = cv2.resize(lo, (1024, 1024), interpolation=cv2.INTER_LINEAR)
            rle = maskUtils.encode(np.asfortranarray((up > 0.5).astype(np.uint8)))
            rows.append({
                "image_id": img_id,
                "category_id": a["category_id"],
                "bbox": list(a["bbox"]),
                "score": 0.99,
                "segmentation": {"size": rle["size"], "counts": rle["counts"].decode("latin-1")},
            })
    oracle["maxpool@0.50"] = oracle_eval(rows)
    print(f"[E] maxpool@0.50: {json.dumps(oracle['maxpool@0.50'])}", flush=True)
    results["E_oracle_ap"] = oracle

    # ---------------- Stage F: +-1px morphology probe on production masks ----------------
    print("[F] morphology probe ...", flush=True)
    morp = defaultdict(lambda: defaultdict(list))
    pair_store = defaultdict(list)  # bucket -> (img_id, ann_idx_global, det_idx)
    # redo matching cheaply storing indices
    dets_by_img2 = dets_by_img
    for img_id, anns in gt_by_img.items():
        dds = sorted(dets_by_img2.get(img_id, []), key=lambda d: -d["score"])[:200]
        if not dds:
            continue
        det_rles = [as_rle(d["segmentation"]) for d in dds]
        gt_rl = [gt_rle(a) for a in anns]
        crowds = [int(a.get("iscrowd", 0)) for a in anns]
        iou_m = maskUtils.iou(det_rles, gt_rl, crowds)
        used = np.zeros(len(anns), dtype=bool)
        for i in range(len(dds)):
            cand = iou_m[i].copy()
            cand[used] = -1
            j = int(cand.argmax())
            if cand[j] >= 0.5:
                used[j] = True
                b = bucket_of(anns[j]["area"])
                if b:
                    pair_store[b].append((dds[i], anns[j]))
    for b, pairs in pair_store.items():
        cap = 1500 if b != "256-1024" else 1000
        for d, a in pairs[:cap]:
            dm = maskUtils.decode(as_rle(d["segmentation"])).astype(bool)
            gm = maskUtils.decode(gt_rle(a)).astype(bool)
            base = iou_masks(dm, gm)
            dil = iou_masks(ndimage.binary_dilation(dm, structure=struct), gm)
            ero = iou_masks(ndimage.binary_erosion(dm, structure=struct, border_value=0), gm)
            morp[b]["base"].append(base)
            morp[b]["dilate1"].append(dil)
            morp[b]["erode1"].append(ero)
    results["F_morph"] = {b: {k: hist_stats(v, edges=[0, 0.3, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.01]) for k, v in d.items()} for b, d in morp.items()}
    results["F_morph_mean"] = {b: {k: float(np.mean(v)) for k, v in d.items()} for b, d in morp.items()}

    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(results, f, indent=1)
    print(f"[done] {time.time()-t0:.0f}s -> results.json", flush=True)


if __name__ == "__main__":
    main()
