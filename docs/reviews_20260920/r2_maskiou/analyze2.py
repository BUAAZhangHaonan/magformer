#!/usr/bin/env python3
"""R2-v2: small-target mask quality & IoU-threshold structure ("paintable" condition).

Usage: analyze2.py <stage>   with stage in
  Asegm | Abbox | B | D | F | CF1 | CF2 | digest

Stages (all CPU; dets.json read exactly once per process; each stage writes
results_<stage>.json immediately so an interrupt loses nothing):

  Asegm/Abbox  COCOeval per-bucket (<64 / 64-256 / 256-1k / COCO-small) AP at
               each IoU threshold .50-.95 (segm and bbox separately).
  B            Matched-pair mask IoU per bucket: for every GT, the highest-
               score det with segm IoU>=0.5 (as briefed); plus per-GT max-IoU
               ceiling (AIM-arena analog, area<4096 aggregate) and box IoU of
               the matched pairs.
  D            GT rendering-chain simulation: 1024 GT --INTER_AREA--> 512
               (coverage fraction, NO /4) --bilinear--> 1024, binarize at
               {0.4,0.45,0.5,0.55}; per-mask IoU vs GT. Sub-buckets + <128
               subgroup + population-weighted COCO-small aggregate.
               (maxpool@0.5 kept as reference; the old analyze.py divided the
               INTER_AREA result by 4, which zeroed every threshold >= 0.25 —
               that was the all-zero stage-E bug.)
  F            +-1px morphological probe on production matched small masks.
  CF1/CF2      Counterfactual rescore: replace the mask of every det matched
               (score-greedy, segm IoU>=0.5) to a small GT (<1024 px^2) with
               (CF1) the GT mask rendered through the exact production chain
               (512 canvas area-interp, bilinear x2, >0.5) or (CF2) the exact
               GT mask; everything else (scores, dets set, other masks)
               unchanged. Full-val COCOeval segm incl. per-threshold curves.
               CF1 = "paintable-through-chain" ceiling given this detection
               set; CF2 = mask-side upper bound given this detection set.
  digest       merge all results_*.json into results.json + print tables.

Hard rule: CUDA_VISIBLE_DEVICES="" (GPUs 4-7 are training).
"""
import contextlib
import io
import json
import os
import sys
import time
from collections import defaultdict

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import cv2
import numpy as np

cv2.setNumThreads(8)
np.set_printoptions(suppress=True)

from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from pycocotools.cocoeval import COCOeval

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
AREA_RNGS = [[0, 1e10], [0, 64], [64, 256], [256, 1024], [0, 1024]]
AREA_LBLS = ["all", "<64", "64-256", "256-1024", "small(0-1024)"]
POP_COUNTS = {"<64": 980, "64-256": 1872, "256-1024": 6679}  # EVIDENCE.md sec.2


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


def iou_masks(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / union if union else 0.0


def render_chain_mask(gt_r):
    """Production chain applied to a GT mask: 512 canvas (INTER_AREA coverage),
    bilinear x2 back to 1024, binarize > 0.5. Returns bool array."""
    g = maskUtils.decode(gt_r).astype(np.float32)
    lo = cv2.resize(g, (512, 512), interpolation=cv2.INTER_AREA)
    up = cv2.resize(lo, (1024, 1024), interpolation=cv2.INTER_LINEAR)
    return up > 0.5


def hist_stats(vals, edges=None, fracs=(0.5, 0.6, 0.7, 0.75, 0.8, 0.9)):
    v = np.asarray(vals, dtype=float)
    out = {"n": int(v.size)}
    if v.size == 0:
        return out
    out.update({
        "mean": float(v.mean()),
        "p25": float(np.percentile(v, 25)),
        "p50": float(np.percentile(v, 50)),
        "p75": float(np.percentile(v, 75)),
    })
    for f in fracs:
        out[f"frac>={f}"] = float((v >= f - 1e-9).mean())
    if edges is None:
        edges = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    h, _ = np.histogram(v, bins=edges)
    out["hist"] = {f"[{edges[i]:.2f},{edges[i+1]:.2f})": int(h[i]) for i in range(len(h))}
    return out


def load_all():
    t0 = time.time()
    dets = json.load(open(DETS))
    print(f"[load] dets rows={len(dets)} ({time.time()-t0:.0f}s)", flush=True)
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(GT_PATH)
    gt_by_img = defaultdict(list)
    for a in coco.dataset["annotations"]:
        gt_by_img[a["image_id"]].append(a)
    print(f"[load] gt imgs={len(gt_by_img)} anns={len(coco.dataset['annotations'])}", flush=True)
    return dets, coco, gt_by_img


def run_eval(coco, dets_rows, iou_type, img_ids=None):
    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco.loadRes(dets_rows)
        e = COCOeval(coco, coco_dt, iouType=iou_type)
        if img_ids is not None:
            e.params.imgIds = img_ids
        e.params.iouThrs = np.array(THRS)
        e.params.areaRng = AREA_RNGS
        e.params.areaRngLbl = AREA_LBLS
        e.params.maxDets = [100]
        e.evaluate()
        e.accumulate()
    P = e.eval["precision"]  # [T, R, K, A, M]
    table = {}
    for ai, lbl in enumerate(AREA_LBLS):
        row = {}
        for ti, thr in enumerate(THRS):
            q = P[ti, :, :, ai, 0]
            q = q[q > -1]
            row[f"@{thr:.2f}"] = float(q.mean()) if q.size else None
        vals = [row[f"@{t:.2f}"] for t in THRS]
        row["AP[.5:.95]"] = float(np.mean([x for x in vals if x is not None]))
        table[lbl] = row
    return table


def save(stage, payload):
    p = os.path.join(HERE, f"results_{stage}.json")
    with open(p, "w") as f:
        json.dump(payload, f, indent=1)
    print(f"[save] {p}", flush=True)


# ----------------------------------------------------------------------------
def stage_a(iou_type):
    dets, coco, _ = load_all()
    t0 = time.time()
    table = run_eval(coco, [dict(d) for d in dets], iou_type)
    print(f"[A:{iou_type}] {time.time()-t0:.0f}s", flush=True)
    save(f"A_{iou_type}", {"iou_type": iou_type, "table": table})
    for lbl in AREA_LBLS:
        print(lbl, json.dumps(table[lbl]), flush=True)


# ----------------------------------------------------------------------------
def match_iter(dets_by_img, gt_by_img, det_cap=200):
    """Yield (img_id, dds, anns, iou_m) once; iou_m[i, j] = mask IoU."""
    for img_id, anns in gt_by_img.items():
        dds = sorted(dets_by_img.get(img_id, []), key=lambda d: -d["score"])[:det_cap]
        if not dds or not anns:
            continue
        det_rles = [as_rle(d["segmentation"]) for d in dds]
        gt_rles = [gt_rle(a) for a in anns]
        crowds = [int(a.get("iscrowd", 0)) for a in anns]
        iou_m = maskUtils.iou(det_rles, gt_rles, crowds)  # [n_det, n_gt]
        yield img_id, dds, anns, iou_m


def stage_b():
    dets, coco, gt_by_img = load_all()
    dets_by_img = defaultdict(list)
    for d in dets:
        dets_by_img[d["image_id"]].append(d)

    # box IoU helper
    def box_iou(d, a):
        dx1, dy1, dw, dh = d["bbox"]
        gx1, gy1, gw, gh = a["bbox"]
        ix = max(0.0, min(dx1 + dw, gx1 + gw) - max(dx1, gx1))
        iy = max(0.0, min(dy1 + dh, gy1 + gh) - max(dy1, gy1))
        inter = ix * iy
        u = dw * dh + gw * gh - inter
        return inter / u if u > 0 else 0.0

    matched = defaultdict(list)   # bucket -> dicts (per-GT best-score det with IoU>=0.5)
    ceiling = defaultdict(list)   # bucket -> per-GT max IoU over dets (incl. 0)
    n_gt = defaultdict(int)
    t0 = time.time()
    n_img = 0
    for img_id, dds, anns, iou_m in match_iter(dets_by_img, gt_by_img):
        n_img += 1
        scores = np.array([d["score"] for d in dds])
        for j, a in enumerate(anns):
            b = bucket_of(a["area"])
            keys = [b] if b else []
            if a["area"] < 4096:
                keys.append("<4096(AIM)")
            col = iou_m[:, j]
            mx = float(col.max()) if col.size else 0.0
            for k in keys:
                n_gt[k] += 1
                ceiling[k].append(mx)
            if not keys:
                continue
            ok = np.where(col >= 0.5)[0]
            if ok.size:
                i = int(ok[np.argmax(scores[ok])])  # highest-score det with IoU>=0.5
                rec = {
                    "mask_iou": float(col[i]),
                    "box_iou": box_iou(dds[i], a),
                    "score": float(scores[i]),
                    "gt_area": float(a["area"]),
                    "det_area": float(maskUtils.area(as_rle(dds[i]["segmentation"]))),
                }
                for k in ([b] if b else []):
                    matched[k].append(rec)
        if n_img % 500 == 0:
            print(f"[B] {n_img}/{len(gt_by_img)} imgs ({time.time()-t0:.0f}s)", flush=True)
    out = {
        "n_gt": dict(n_gt),
        "matched_n": {b: len(v) for b, v in matched.items()},
        "matched_recall_at_0.5": {b: len(v) / max(n_gt[b], 1) for b, v in matched.items()},
        "matched_mask_iou": {b: hist_stats([m["mask_iou"] for m in v]) for b, v in matched.items()},
        "matched_box_iou": {b: hist_stats([m["box_iou"] for m in v]) for b, v in matched.items()},
        "matched_score": {b: hist_stats([m["score"] for m in v], edges=[0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.01]) for b, v in matched.items()},
        "matched_box_minus_mask": {b: hist_stats([m["box_iou"] - m["mask_iou"] for m in v], edges=[-1, -.3, -.2, -.1, -.05, 0, .05, .1, .2, .3, 1.01]) for b, v in matched.items()},
        "ceiling_iou": {b: hist_stats(v) for b, v in ceiling.items()},
        "aim_reference": {"arena_matched_iou_area<4096_frozen_c0": 0.4134, "arena_oracle": 0.4152},
    }
    print(f"[B] done {time.time()-t0:.0f}s", flush=True)
    for b in ["<64", "64-256", "256-1024", "<4096(AIM)"]:
        if b in out["matched_mask_iou"]:
            m = out["matched_mask_iou"][b]
            c = out["ceiling_iou"][b]
            print(f"[B] {b}: n_gt={n_gt[b]} matched={out['matched_n'][b]} "
                  f"rec@.5={out['matched_recall_at_0.5'][b]:.3f} "
                  f"matchedIoU p50={m['p50']:.3f} frac>=.75={m['frac>=0.75']:.3f} | "
                  f"ceiling mean={c['mean']:.3f} p50={c['p50']:.3f}", flush=True)
    save("B", out)


# ----------------------------------------------------------------------------
def stage_d():
    dets, coco, gt_by_img = load_all()  # dets unused; kept for one-file loading
    del dets
    RENDER_THRS = [0.4, 0.45, 0.5, 0.55]
    caps = {"<32": 400, "32-64": 400, "64-128": 400, "128-256": 400, "256-1024": 400}
    counts = defaultdict(int)
    sim = defaultdict(lambda: defaultdict(list))   # subbucket -> variant -> [iou]
    areas = defaultdict(list)
    t0 = time.time()
    for img_id, anns in gt_by_img.items():
        done = True
        for a in anns:
            ar = a["area"]
            if ar >= 1024:
                continue
            sb = ("<32" if ar < 32 else
                  "32-64" if ar < 64 else
                  "64-128" if ar < 128 else
                  "128-256" if ar < 256 else
                  "256-1024")
            if counts[sb] >= caps[sb]:
                continue
            g = maskUtils.decode(gt_rle(a)).astype(np.float32)
            gb = g > 0.5
            lo_area = cv2.resize(g, (512, 512), interpolation=cv2.INTER_AREA)  # coverage in [0,1]
            lo_max = g.reshape(512, 2, 512, 2).max(axis=(1, 3)).astype(np.float32)
            for name, lo in (("area", lo_area), ("maxpool", lo_max)):
                up = cv2.resize(lo, (1024, 1024), interpolation=cv2.INTER_LINEAR)
                for t in RENDER_THRS:
                    sim[sb][f"{name}@{t:.2f}"].append(iou_masks(up > t, gb))
            counts[sb] += 1
            areas[sb].append(float(ar))
        if all(counts[k] >= caps[k] for k in caps):
            break
    E = [0, 0.3, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.01]
    out = {
        "counts": dict(counts),
        "render_thrs": RENDER_THRS,
        "sim": {sb: {k: hist_stats(v, edges=E) for k, v in d.items()} for sb, d in sim.items()},
        "sim_raw": {sb: {k: [round(x, 4) for x in v] for k, v in d.items()} for sb, d in sim.items()},
    }
    # merges
    def merge(keys):
        return {k: hist_stats(sum((sim[s][k] for s in keys), []), edges=E) for k in sim["<32"]}

    out["merge_<64"] = merge(["<32", "32-64"])
    out["merge_<128"] = merge(["<32", "32-64", "64-128"])
    out["merge_small(0-1024)"] = merge(["<32", "32-64", "64-128", "128-256", "256-1024"])
    # population-weighted COCO-small aggregate frac(IoU>=t) per render threshold
    pop = {"<64": POP_COUNTS["<64"], "64-256": POP_COUNTS["64-256"], "256-1024": POP_COUNTS["256-1024"]}
    agg = {}
    for k in sim["<32"]:
        if not k.startswith("area@"):
            continue
        per = []
        tot = 0
        for b, w in pop.items():
            v = np.asarray(sim_key_merge(sim, b, k), dtype=float)
            per.append(v)
            tot += w
        allv = np.concatenate(per)
        wvec = np.concatenate([np.full(len(p), pop[b]) for p, b in zip(per, pop)])
        fr = {}
        for t in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
            fr[f">={t}"] = float(np.average(allv >= t - 1e-9, weights=wvec))
        agg[k] = fr
    out["popweighted_small_frac"] = agg
    print(f"[D] done {time.time()-t0:.0f}s counts={dict(counts)}", flush=True)
    for k in [f"area@{t:.2f}" for t in RENDER_THRS]:
        r = out["merge_small(0-1024)"][k]
        print(f"[D] small(0-1024) {k}: p50={r['p50']:.3f} frac>=.5={r['frac>=0.5']:.3f} "
              f"frac>=.75={r['frac>=0.75']:.3f}", flush=True)
    for k in [f"area@{t:.2f}" for t in RENDER_THRS]:
        r = out["merge_<128"][k]
        print(f"[D] <128 {k}: p50={r['p50']:.3f} frac>=.5={r['frac>=0.5']:.3f} "
              f"frac>=.75={r['frac>=0.75']:.3f}", flush=True)
    save("D", out)


def sim_key_merge(sim, bucket, key):
    if bucket == "<64":
        return sim["<32"][key] + sim["32-64"][key]
    if bucket == "64-256":
        return sim["64-128"][key] + sim["128-256"][key]
    return sim["256-1024"][key]


# ----------------------------------------------------------------------------
def stage_cf(which):
    """CF1: matched small dets get the GT mask through the render chain;
    CF2: matched small dets get the exact GT mask."""
    dets, coco, gt_by_img = load_all()
    dets_by_img = defaultdict(list)
    for d in dets:
        dets_by_img[d["image_id"]].append(d)

    t0 = time.time()
    rows = [dict(d) for d in dets]  # preserve original det order & scores
    # index map: for each image, sorted dds -> their id in rows (use object id via
    # a per-image index built from the original list order is fragile; instead we
    # rebuild replacement map keyed by (image_id, score, mask area) -> risky.
    # Simpler: rebuild rows per image from dets_by_img preserving nothing; the
    # global det list order does not matter for COCOeval.
    n_repl = 0
    repl_areas = []
    out_rows = []
    for img_id, dds, anns, iou_m in match_iter(dets_by_img, gt_by_img):
        small_j = [j for j, a in enumerate(anns) if a["area"] < 1024]
        assign = {}   # det idx -> gt idx (score-greedy COCO-style, small GTs only)
        if small_j:
            used = set()
            order = np.argsort([-d["score"] for d in dds])
            for i in order:
                cand = iou_m[i].copy()
                for j in range(len(anns)):
                    if j in used or j not in small_j:
                        cand[j] = -1
                j = int(cand.argmax())
                if cand[j] >= 0.5:
                    used.add(j)
                    assign[i] = j
        repl = {}
        for i, j in assign.items():
            gr = gt_rle(anns[j])
            if which == "CF2":
                rle = gr
            else:
                m = render_chain_mask(gr).astype(np.uint8)
                rle = maskUtils.encode(np.asfortranarray(m))
            repl[i] = {
                "segmentation": {"size": rle["size"], "counts": rle["counts"]},
            }
            repl_areas.append(float(anns[j]["area"]))
        for i, d in enumerate(dds):
            r = dict(d)
            if i in repl:
                r.update(repl[i])
                n_repl += 1
            out_rows.append(r)
    # dets on images with no GT anns / no dds kept as-is
    seen_imgs = set(gt_by_img.keys())
    for img_id, dl in dets_by_img.items():
        if img_id not in seen_imgs:
            out_rows.extend(dict(d) for d in dl)
    print(f"[{which}] replaced masks on {n_repl} small-matched dets ({time.time()-t0:.0f}s)", flush=True)

    table = run_eval(coco, out_rows, "segm")
    save(which, {
        "which": which,
        "n_replaced": n_repl,
        "replaced_gt_area_hist": hist_stats(repl_areas, edges=[0, 32, 64, 128, 256, 512, 1024]),
        "table": table,
    })
    for lbl in AREA_LBLS:
        print(lbl, json.dumps(table[lbl]), flush=True)


# ----------------------------------------------------------------------------
def stage_f():
    from scipy import ndimage
    dets, coco, gt_by_img = load_all()
    dets_by_img = defaultdict(list)
    for d in dets:
        dets_by_img[d["image_id"]].append(d)
    struct = np.ones((3, 3), dtype=bool)
    caps = {"<64": 800, "64-256": 800, "256-1024": 400}
    got = defaultdict(int)
    morp = defaultdict(lambda: defaultdict(list))
    t0 = time.time()
    for img_id, dds, anns, iou_m in match_iter(dets_by_img, gt_by_img):
        used = np.zeros(len(anns), dtype=bool)
        order = np.argsort([-d["score"] for d in dds])
        for i in order:
            cand = iou_m[i].copy()
            cand[used] = -1
            j = int(cand.argmax())
            if cand[j] >= 0.5:
                used[j] = True
                b = bucket_of(anns[j]["area"])
                if b and got[b] < caps[b]:
                    got[b] += 1
                    dm = maskUtils.decode(as_rle(dds[i]["segmentation"])).astype(bool)
                    gm = maskUtils.decode(gt_rle(anns[j])).astype(bool)
                    morp[b]["base"].append(iou_masks(dm, gm))
                    morp[b]["dilate1"].append(iou_masks(ndimage.binary_dilation(dm, structure=struct), gm))
                    morp[b]["erode1"].append(iou_masks(ndimage.binary_erosion(dm, structure=struct, border_value=0), gm))
        if all(got[k] >= caps[k] for k in caps):
            break
    out = {b: {k: hist_stats(v, edges=[0, 0.3, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.01]) for k, v in d.items()}
           for b, d in morp.items()}
    out["_mean"] = {b: {k: float(np.mean(v)) for k, v in d.items()} for b, d in morp.items()}
    out["_n"] = dict(got)
    print(f"[F] done {time.time()-t0:.0f}s n={dict(got)}", flush=True)
    print("[F] means:", json.dumps(out["_mean"]), flush=True)
    save("F", out)


# ----------------------------------------------------------------------------
def stage_digest():
    merged = {}
    for fn in sorted(os.listdir(HERE)):
        if fn.startswith("results_") and fn.endswith(".json") and fn != "results.json":
            key = fn[len("results_"):-len(".json")]
            try:
                merged[key] = json.load(open(os.path.join(HERE, fn)))
            except Exception as e:
                merged[key] = f"<unreadable: {e}>"
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(merged, f, indent=1)
    print("[digest] keys:", list(merged), flush=True)
    # key comparison table: segm vs bbox per bucket per threshold
    if "A_segm" in merged and "A_bbox" in merged:
        ts = merged["A_segm"]["table"]
        tb = merged["A_bbox"]["table"]
        for lbl in AREA_LBLS:
            print(f"--- {lbl}: thr segm bbox delta")
            for t in THRS + ["AP[.5:.95]"]:
                k = f"@{t:.2f}" if t != "AP[.5:.95]" else t
                s, b = ts[lbl][k], tb[lbl][k]
                if s is not None and b is not None:
                    print(f"{k} {s:.4f} {b:.4f} {b-s:+.4f}")


def main():
    stage = sys.argv[1]
    if stage == "Asegm":
        stage_a("segm")
    elif stage == "Abbox":
        stage_a("bbox")
    elif stage == "B":
        stage_b()
    elif stage == "D":
        stage_d()
    elif stage == "F":
        stage_f()
    elif stage == "CF1":
        stage_cf("CF1")
    elif stage == "CF2":
        stage_cf("CF2")
    elif stage == "digest":
        stage_digest()
    else:
        raise SystemExit(f"unknown stage {stage}")


if __name__ == "__main__":
    main()
