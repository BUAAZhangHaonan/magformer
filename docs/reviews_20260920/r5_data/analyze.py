#!/usr/bin/env python3
"""R5 independent review: data density, augmentation, copy-paste channel.

CPU-only. Streams the 2.9 GB train annotation JSON with a hand-rolled
raw_decode streamer (ijson unavailable in the env, pip install is out of
scope for the write sandbox).

Parts:
  A. train/val annotation stats: GT per image, sub-bucket counts (buckets
     match scripts/subbucket_scores.py: area px^2, <64 / 64-256 / 256-1024),
     small share, pixel share, paste-source eligibility.
  B. augmentation accounting: scale ~ U(0.5, 2.0) (ResizeScale), analytic
     effective-size distribution incl. FixedSizeCrop visibility (1/s^2 for
     s>1, 1 for s<=1).  Net densify-vs-crush numbers.
  C. depth channel: 200 sampled val images (among images with >=1 small GT),
     normalized-depth contrast inside small masks vs local ring, SNR vs the
     gaussian noise (std 0.01 raw -> 0.025 normalized after (d-0.3)/0.4).
"""
import json
import codecs
import random
import sys
import time
from collections import defaultdict

import numpy as np

TRAIN_ANN = (
    "/home/hdd3/zhanghaonan/magformer/magformer_datasets/20260318_1K_32254/"
    "annotations/instances_train.validated.json"
)
VAL_ANN = (
    "/home/hdd3/zhanghaonan/magformer/magformer_datasets/20260318_1K_32254/"
    "annotations/instances_val.validated.json"
)
DEPTH_VAL_DIR = (
    "/home/hdd3/zhanghaonan/magformer/magformer_datasets/20260318_1K_32254/"
    "depth/depth_npy/val"
)
IMG_VAL_DIR = (
    "/home/hdd3/zhanghaonan/magformer/magformer_datasets/20260318_1K_32254/"
    "images/val"
)

BATCH_IMGS = 4  # F1: 4 GPUs x ims_per_batch 1
MIN_SCALE, MAX_SCALE = 0.5, 2.0  # F1 config
IMAGE_SIZE = 1024
RAW_NOISE_STD = 0.01  # depth gaussian noise, raw units (applied pre-normalize)
CLIP_MIN, CLIP_MAX = 0.3, 0.7  # depth clip, raw units
NORM_GAIN = 1.0 / (CLIP_MAX - CLIP_MIN)  # -> 2.5 ; normalized noise = 0.025
SEED = 20260920
SMALL = 1024  # small threshold, area px^2 (== side < 32 px)


# ---------------------------------------------------------------------------
# streaming COCO parser
# ---------------------------------------------------------------------------
def stream_top_array(path, key, chunk_bytes=1 << 24):
    """Yield dicts of top-level array `key` from a (large) COCO JSON.

    Cursor-based: never reslices the buffer per item (avoids O(n^2) copies);
    compacts only when refilling.
    """
    dec = json.JSONDecoder()
    inc = codecs.getincrementaldecoder("utf-8")()
    buf = ""
    cur = 0
    started = False
    eof = False
    key_needle = '"%s"' % key
    with open(path, "rb") as f:
        while True:
            if not eof:
                raw = f.read(chunk_bytes)
                if not raw:
                    eof = True
                    try:
                        tail = inc.decode(b"", final=True)
                    except UnicodeDecodeError:
                        tail = ""
                    buf += tail
                else:
                    buf += inc.decode(raw)
                    if not started:
                        pos = buf.find(key_needle)
                        if pos == -1:
                            buf = buf[-(len(key_needle) + 8):]
                            continue
                        br = buf.find("[", pos + len(key_needle))
                        if br == -1:
                            buf = buf[pos:]
                            continue
                        buf = buf[br + 1:]
                        started = True
                # compact consumed prefix before/after read
                if cur > 0 and cur < len(buf):
                    buf = buf[cur:]
                    cur = 0
                elif cur >= len(buf):
                    buf = ""
                    cur = 0
            # try to parse items at cursor
            progressed = True
            while progressed:
                progressed = False
                i = cur
                n = len(buf)
                while i < n and buf[i] in " \t\r\n":
                    i += 1
                if i >= n:
                    cur = i
                    if eof:
                        return
                    break
                if buf[i] == "]":
                    return
                if buf[i] == ",":
                    cur = i + 1
                    progressed = True
                    continue
                if buf[i] not in "{[\"0123456789tfn-":
                    # unexpected token: cannot make progress
                    if eof:
                        raise ValueError(f"stuck at {buf[i:i+40]!r}")
                    break
                try:
                    obj, end = dec.raw_decode(buf, i)
                except json.JSONDecodeError:
                    if eof:
                        raise
                    cur = i
                    break
                yield obj
                cur = end
                progressed = True
            if not eof and not progressed:
                continue
            if eof and not progressed:
                return


def pct(a, q):
    if len(a) == 0:
        return None
    return float(np.percentile(np.asarray(a), q))


# ---------------------------------------------------------------------------
# Part A: annotation statistics
# ---------------------------------------------------------------------------
def ann_stats(split, path):
    t0 = time.time()
    images = {}  # id -> (h, w, file_name)
    for im in stream_top_array(path, "images"):
        images[im["id"]] = (im.get("height"), im.get("width"), im.get("file_name"))
    n_img = len(images)
    shapes = defaultdict(int)
    for (h, w, _) in images.values():
        shapes[(h, w)] += 1

    img_area = {iid: h * w for iid, (h, w, _) in images.items()}
    per_img = defaultdict(lambda: np.zeros(5, dtype=np.int64))  # gt, <64, 64-256, 256-1k, small_sum_idx
    sums = defaultdict(float)
    counts = defaultdict(int)
    # per-ann compact arrays for part B / paste eligibility
    areas_l, imgid_l = [], []
    boxes_l = []
    crowd = 0
    for a in stream_top_array(path, "annotations"):
        if a.get("iscrowd", 0):
            crowd += 1
            continue
        iid = a["image_id"]
        ar = float(a["area"])
        x, y, w, h = a["bbox"]
        areas_l.append(ar)
        imgid_l.append(iid)
        boxes_l.append((x, y, x + w, y + h))
        row = per_img[iid]
        row[0] += 1
        if ar < 64:
            row[1] += 1
            counts["<64"] += 1
        elif ar < 256:
            row[2] += 1
            counts["64-256"] += 1
        elif ar < 1024:
            row[3] += 1
            counts["256-1024"] += 1
        if ar < SMALL:
            row[4] += 1
        sums["all"] += ar
        if ar < SMALL:
            sums["small"] += ar
        if ar < 64:
            sums["<64"] += ar

    areas = np.asarray(areas_l, dtype=np.float64)
    boxes = np.asarray(boxes_l, dtype=np.float64)
    imgids = np.asarray(imgid_l, dtype=np.int64)
    gt_per_img = np.array([per_img[i][0] for i in images], dtype=np.int64)
    small_per_img = np.array([per_img[i][4] for i in images], dtype=np.int64)
    b64_per_img = np.array([per_img[i][1] for i in images], dtype=np.int64)
    n_gt = int(gt_per_img.sum())
    n_small = int(small_per_img.sum())

    # paste-source eligibility (non-crowd, complete bbox inside image)
    ihw = np.array([[images[int(i)][0], images[int(i)][1]] for i in imgids])
    ih, iw = ihw[:, 0][:, None], ihw[:, 1][:, None]
    x1, y1, x2, y2 = boxes[:, 0:1], boxes[:, 1:2], boxes[:, 2:3], boxes[:, 3:4]
    inside0 = (x1 >= 0) & (y1 >= 0) & (x2 <= iw) & (y2 <= ih)
    inside2 = (x1 >= 2) & (y1 >= 2) & (x2 <= iw - 2) & (y2 <= ih - 2)
    small_m = areas < SMALL
    paste_src = {
        "noncrowd_total": n_gt,
        "crowd": crowd,
        "small_noncrowd": int(small_m.sum()),
        "small_area_ge4": int((small_m & (areas >= 4)).sum()),
        "small_area_ge16": int((small_m & (areas >= 16)).sum()),
        "small_bbox_inside_margin0": int((small_m & inside0[:, 0]).sum()),
        "small_bbox_inside_margin2": int((small_m & inside2[:, 0]).sum()),
        "small_ge16_inside_margin2": int(
            (small_m & (areas >= 16) & inside2[:, 0]).sum()
        ),
        "small_subhist": {
            "<64": counts["<64"],
            "64-256": counts["64-256"],
            "256-1024": counts["256-1024"],
        },
    }

    out = {
        "n_images": n_img,
        "image_shapes": {f"{h}x{w}": c for (h, w), c in shapes.items()},
        "n_gt_noncrowd": n_gt,
        "n_crowd": crowd,
        "n_small(<1024)": n_small,
        "small_share_of_gt": n_small / max(n_gt, 1),
        "buckets": dict(counts),
        "gt_per_img": {
            "mean": float(gt_per_img.mean()),
            "p50": pct(gt_per_img, 50),
            "p90": pct(gt_per_img, 90),
            "max": int(gt_per_img.max()),
        },
        "small_per_img_all": {
            "mean": float(small_per_img.mean()),
            "p50": pct(small_per_img, 50),
            "p90": pct(small_per_img, 90),
            "max": int(small_per_img.max()),
            "frac_imgs_with_ge1_small": float((small_per_img > 0).mean()),
        },
        "small_per_img_cond": {
            "mean": float(small_per_img[small_per_img > 0].mean()),
            "p50": pct(small_per_img[small_per_img > 0], 50),
            "p90": pct(small_per_img[small_per_img > 0], 90),
            "max": int(small_per_img.max()),
        },
        "lt64_per_img_all": {
            "mean": float(b64_per_img.mean()),
            "p90": pct(b64_per_img, 90),
            "max": int(b64_per_img.max()),
            "frac_imgs_with_ge1": float((b64_per_img > 0).mean()),
        },
        "expected_small_per_batch4": float(BATCH_IMGS * small_per_img.mean()),
        "expected_lt64_per_batch4": float(BATCH_IMGS * b64_per_img.mean()),
        "pixel_share": {
            "small_px_per_img": sums["small"] / max(n_img, 1),
            "small_pixel_share": sums["small"] / max(sum(img_area.values()), 1),
            "lt64_pixel_share": sums["<64"] / max(sum(img_area.values()), 1),
            "median_small_area": float(np.median(areas[small_m])) if small_m.any() else None,
        },
        "paste_sources": paste_src,
    }
    print(f"[A:{split}] done in {time.time()-t0:.0f}s: {json.dumps({k: out[k] for k in ('n_images','n_gt_noncrowd','n_small(<1024)','buckets')}, default=str)}")
    return out, areas, imgids, images


# ---------------------------------------------------------------------------
# Part B: augmentation accounting (analytic)
# ---------------------------------------------------------------------------
def p_eff_below(area, thr):
    """P(a*s^2 < thr), s ~ U(0.5, 2)."""
    x = np.sqrt(thr / np.maximum(area, 1e-9))
    return np.clip((x - MIN_SCALE) / (MAX_SCALE - MIN_SCALE), 0.0, 1.0)


def e_visible_below(area, thr):
    """E[vis(s) * 1{a*s^2 < thr}], vis(s)=1 (s<=1), 1/s^2 (s>1).

    Closed form: let x = sqrt(thr/a) capped at 2.
      x <= 0.5        -> 0
      0.5 < x <= 1    -> (x - 0.5) / 1.5
      1 < x <= 2      -> (0.5 + (1 - 1/x)) / 1.5
    """
    x = np.minimum(np.sqrt(thr / np.maximum(area, 1e-9)), MAX_SCALE)
    span = MAX_SCALE - MIN_SCALE
    out = np.zeros_like(x)
    m1 = (x > MIN_SCALE) & (x <= 1.0)
    out[m1] = (x[m1] - MIN_SCALE) / span
    m2 = x > 1.0
    out[m2] = (0.5 + (1.0 - 1.0 / x[m2])) / span
    return out


def aug_accounting(areas):
    n = len(areas)
    res = {}
    for thr in (4, 16, 32, 64, 256, 1024):
        res[f"area<{thr}"] = {
            "orig_count": int((areas < thr).sum()),
            "orig_share": float((areas < thr).mean()),
            "E_paste_prob": float(p_eff_below(areas, thr).mean()),
            "E_visible_count": float(e_visible_below(areas, thr).sum()),
        }
    # overall visibility: E[vis] = (0.5 + 0.5)/1.5 = 2/3 for thr=inf
    e_vis_all = (0.5 + 0.5) / (MAX_SCALE - MIN_SCALE)
    n_vis_all = e_vis_all * n

    # effective visible composition
    eff = {}
    for name, lo, hi in (("<64", 0, 64), ("64-256", 64, 256), ("256-1024", 256, 1024)):
        orig = ((areas >= lo) & (areas < hi)).sum()
        vis_lo = e_visible_below(areas, lo).sum()
        vis_hi = e_visible_below(areas, hi).sum()
        eff[name] = {
            "orig": int(orig),
            "E_visible": float(vis_hi - vis_lo),
        }
    eff["all_visible"] = {"E_visible": float(n_vis_all)}
    eff["visible_share"] = {
        k: eff[k]["E_visible"] / n_vis_all for k in ("<64", "64-256", "256-1024")
    }
    eff["orig_share"] = {
        "<64": float((areas < 64).mean()),
        "64-256": float(((areas >= 64) & (areas < 256)).mean()),
        "256-1024": float(((areas >= 256) & (areas < 1024)).mean()),
    }

    # crush metrics: among small GT (a<1024), P(effective side < 8px / <4px / <2px)
    small = areas[areas < SMALL]
    side = np.sqrt(small)
    crush = {
        "n_small": int(small.size),
        "P_eff_side<8px (=area<64)": float(p_eff_below(small, 64).mean()),
        "P_eff_side<5.7px(=area<32)": float(p_eff_below(small, 32).mean()),
        "P_eff_side<4px (=area<16)": float(p_eff_below(small, 16).mean()),
        "P_eff_side<2px (=area<4)": float(p_eff_below(small, 4).mean()),
        "side_pct": {
            "p10": float(np.percentile(side, 10)),
            "p50": float(np.percentile(side, 50)),
            "p90": float(np.percentile(side, 90)),
        },
        # probability small GT gets upscaled to >=8px side (densify direction)
        "P_eff_side>=8px": float((1.0 - p_eff_below(small, 64)).mean()),
        "P_scale>=1.25x": float(np.clip((MAX_SCALE - 1.25) / (MAX_SCALE - MIN_SCALE), 0, 1)),
    }
    # per-batch expectations
    per_img_visible_lt64 = eff["<64"]["E_visible"] / 25654.0
    per_img_visible_small = (
        eff["256-1024"]["E_visible"] + eff["64-256"]["E_visible"] + eff["<64"]["E_visible"]
    ) / 25654.0
    batch = {
        "E_visible_small_per_img": per_img_visible_small,
        "E_visible_lt64_per_img": per_img_visible_lt64,
        "E_visible_small_per_batch4": BATCH_IMGS * per_img_visible_small,
        "E_visible_lt64_per_batch4": BATCH_IMGS * per_img_visible_lt64,
    }
    return {"thresholds": res, "buckets": eff, "crush": crush, "batch": batch,
            "note": "visibility approx: random 1024^2 crop of (1024s)^2 canvas keeps 1/s^2 for s>1 (FixedSizeCrop min-objects guard biases slightly upward); pad keeps all for s<=1; clipped-at-edge shrinkage ignored"}


# ---------------------------------------------------------------------------
# Part C: depth contrast sampling (200 val images)
# ---------------------------------------------------------------------------
def _decode_mask(seg, h, w):
    from pycocotools import mask as maskUtils

    if isinstance(seg, list):
        rle = maskUtils.merge(maskUtils.frPyObjects(seg, h, w))
    elif isinstance(seg, dict):
        if isinstance(seg.get("counts"), str):
            rle = maskUtils.frPyObjects([seg], h, w)[0]
        else:
            rle = seg  # compressed RLE
    else:
        return None
    if isinstance(rle, list):
        rle = rle[0]
    rle = {"size": rle["size"], "counts": rle["counts"]}
    return maskUtils.decode(rle).astype(bool)


def depth_stats(images_val, n_sample=200):
    import cv2

    rng = random.Random(SEED)
    # pass 1: which images have small GT
    small_by_img = defaultdict(list)
    for a in stream_top_array(VAL_ANN, "annotations"):
        if a.get("iscrowd", 0):
            continue
        if a["area"] < SMALL:
            small_by_img[a["image_id"]].append(a)
    cand = sorted(small_by_img.keys())
    sample = rng.sample(cand, min(n_sample, len(cand)))
    sample_set = set(sample)
    print(f"[C] images with >=1 small GT: {len(cand)}/{len(images_val)}; sampling {len(sample)}")

    # pass 2: all anns of the sampled images (for ring-minus-other-instances)
    all_by_img = defaultdict(list)
    for a in stream_top_array(VAL_ANN, "annotations"):
        if a["image_id"] in sample_set and not a.get("iscrowd", 0):
            all_by_img[a["image_id"]].append(a)

    recs = []  # (bucket, area, delta_norm, delta_norm_bg, snr, in_std_norm)
    img_sat = []
    n_img_ok = 0
    noise_norm = RAW_NOISE_STD * NORM_GAIN
    for iid in sample:
        anns = small_by_img[iid]
        fn = images_val[iid][2]
        stem = fn.rsplit(".", 1)[0]
        dpath = f"{DEPTH_VAL_DIR}/{stem}.npy"
        try:
            raw = np.load(dpath)
        except FileNotFoundError:
            continue
        valid = np.isfinite(raw) & (raw > 0)
        sat = float(((raw < CLIP_MIN) | (raw > CLIP_MAX)).mean())
        img_sat.append(sat)
        norm = np.where(valid, (np.clip(raw, CLIP_MIN, CLIP_MAX) - CLIP_MIN) * NORM_GAIN, 0.0)
        h, w = raw.shape
        # full-instance occupancy (bool) for this image
        occ = np.zeros((h, w), dtype=bool)
        for a in all_by_img.get(iid, []):
            m0 = _decode_mask(a.get("segmentation"), h, w)
            if m0 is not None and m0.any():
                occ |= m0
        n_img_ok += 1
        for a in anns:
            ar = float(a["area"])
            m = _decode_mask(a.get("segmentation"), h, w)
            if m is None or not m.any():
                continue
            side = float(np.sqrt(ar))
            r = max(2, int(round(side / 2)))
            ys, xs = np.where(m)
            y1, y2 = max(0, ys.min() - 2 * r), min(h, ys.max() + 2 * r + 1)
            x1, x2 = max(0, xs.min() - 2 * r), min(w, xs.max() + 2 * r + 1)
            sub = m[y1:y2, x1:x2]
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
            dil = cv2.dilate(sub.astype(np.uint8), k).astype(bool)
            ring = dil & ~sub
            ring_bg = ring & ~occ[y1:y2, x1:x2]  # true background only
            if sub.sum() < 4 or ring.sum() < 8:
                continue
            din = norm[y1:y2, x1:x2][sub]
            dring = norm[y1:y2, x1:x2][ring]
            delta = float(din.mean() - dring.mean())
            delta_bg = None
            if ring_bg.sum() >= 8:
                delta_bg = float(din.mean() - norm[y1:y2, x1:x2][ring_bg].mean())
            bucket = "<64" if ar < 64 else "64-256" if ar < 256 else "256-1024"
            recs.append((bucket, ar, delta, delta_bg, abs(delta) / noise_norm,
                         float(din.std())))

    out = {"n_images_used": n_img_ok, "n_small_gt_measured": len(recs),
           "noise_std_normalized": noise_norm,
           "mean_clip_saturation_frac": float(np.mean(img_sat)) if img_sat else None}
    for b in ("<64", "64-256", "256-1024"):
        d = [r for r in recs if r[0] == b]
        if not d:
            out[b] = {"n": 0}
            continue
        ad = np.array([abs(r[2]) for r in d])
        adbg = np.array([abs(r[3]) for r in d if r[3] is not None])
        snr = np.array([r[4] for r in d])
        out[b] = {
            "n": len(d),
            "abs_delta_norm_p50": float(np.percentile(ad, 50)),
            "abs_delta_norm_p25": float(np.percentile(ad, 25)),
            "abs_delta_norm_p75": float(np.percentile(ad, 75)),
            "abs_delta_normBG_p50": float(np.percentile(adbg, 50)) if adbg.size else None,
            "snr_p50": float(np.percentile(snr, 50)),
            "frac_absDelta>0.05(2sigma)": float((ad > 0.05).mean()),
            "frac_absDelta>0.10(4sigma)": float((ad > 0.10).mean()),
        }
    d = [r for r in recs if r[1] < 32]
    if d:
        ad = np.array([abs(r[2]) for r in d])
        out["<32px2"] = {
            "n": len(d),
            "abs_delta_norm_p50": float(np.percentile(ad, 50)),
            "frac_absDelta>0.05": float((ad > 0.05).mean()),
        }
    return out


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    out = {}

    print("=== Part A: train ===")
    tr, tr_areas, tr_imgids, _ = ann_stats("train", TRAIN_ANN)
    out["train"] = tr

    print("=== Part A: val ===")
    va, va_areas, _, va_images = ann_stats("val", VAL_ANN)
    out["val"] = va

    print("=== Part B: augmentation accounting (train areas) ===")
    out["aug"] = aug_accounting(tr_areas)

    print("=== Part C: depth contrast (200 val imgs) ===")
    out["depth"] = depth_stats(va_images)

    # loss point-share estimate (uniform part of the 12544-point sampler)
    ps = tr["pixel_share"]["small_pixel_share"]
    out["loss_point_share"] = {
        "K_points": 12544,
        "uniform_part(25pct)_expected_small_fg_points": 0.25 * 12544 * ps,
        "importance_part(75pct)_max_all_fg_points": 0.75 * 12544,
        "small_fg_points_upper_bound": 12544 * min(1.0, ps * 3.0),
        "note": "MaskFormer point sampler: oversample_ratio 3.0, importance_sample_ratio 0.75 -> fg point share capped at ~3x uniform pixel share",
    }

    with open("/home/hdd3/zhanghaonan/magformer/output/aps_20260913/reviews_20260920/r5_data/stats.json", "w") as f:
        json.dump(out, f, indent=1, default=str)
    print("written stats.json")
    print(json.dumps(out["aug"]["batch"], indent=1))
    print(json.dumps(out["aug"]["crush"], indent=1))


if __name__ == "__main__":
    main()
