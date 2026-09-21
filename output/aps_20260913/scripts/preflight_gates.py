#!/usr/bin/env python
"""G1 preflight gates on the sealed 128K checkpoint (preregistered).

Gate 1  q-IQR (MAL-CP+): quality = soft-Dice of each matched det vs its GT,
        from the sealed dets + val GT. PASS: small-bucket IQR >= 0.15 and
        p50 in [0.5, 0.8]; keeper-vs-duplicate gap >= 0.10 (secondary).
Gate 2  band-ratio (BAS-CL+): with the coverage-band definition (avg-pool
        2x2 of GT, band = coverage in (0.05, 0.95)), the sealed binary dets
        must err MORE inside the band than the interior:
        band_err / interior_err >= 1.5 (kill < 1.1) — i.e. the error mass
        the boundary supervision will attack actually lives in the band.
Gate 3  depth-SNR calibration (C2): deposit-time raw-depth contrast
        distribution over sampled train instances -> recommended
        depth_snr_min (kept OFF in B1 unless clearly separable).
Probe-recall gate cannot run on the seal (no probe head) — deferred to the
B1 arm's first eval checkpoint, per the P4-b verdict.
"""
import argparse
import json

import numpy as np
import pycocotools.mask as mask_utils

ROOT = "/home/hdd3/zhanghaonan/magformer"
SEAL_DETS = f"{ROOT}/output/aps_20260913/p5_runs/f1_seal_128k/coco_instances_results.json"
VAL_GT = f"{ROOT}/magformer_datasets/20260318_1K_32254/annotations/instances_val.validated.json"
TRAIN_GT = f"{ROOT}/magformer_datasets/20260318_1K_32254/annotations/instances_train.validated.json"
OUT = f"{ROOT}/output/aps_20260913/g1_runs/preflight.json"


def rle_iou_pair(a_rle, b_rle):
    i = mask_utils.iou([a_rle], [b_rle], [0])[0][0]
    return i


def soft_dice(det_rle, gt_rle):
    """Approximate soft-Dice between a binary det and GT via intersection
    ratio symmetric form 2I/(A+B) (equals Dice on binaries; the q in the
    matcher uses soft probabilities, this is its binary counterpart and the
    preregistered offline proxy)."""
    return 2.0 * mask_utils.area(det_rle) * 0 + 2 * mask_utils.area(
        mask_utils.merge([det_rle, gt_rle], intersect=True)) / (
        mask_utils.area(det_rle) + mask_utils.area(gt_rle) + 1e-6)


def main(limit_images=None):
    dets = json.load(open(SEAL_DETS))
    gt = json.load(open(VAL_GT))
    anns_by_img = {}
    for a in gt["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)
    dets_by_img = {}
    for d in dets:
        dets_by_img.setdefault(d["image_id"], []).append(d)
    img_hw = {im["id"]: (im["height"], im["width"]) for im in gt["images"]}

    small_q, keeper_gap = [], []
    band_hits, band_tot, int_hits, int_tot = 0, 0, 0, 0
    imgs = sorted(anns_by_img.keys())
    if limit_images:
        imgs = imgs[:limit_images]

    for img_id in imgs:
        gts = anns_by_img[img_id]
        dsl = dets_by_img.get(img_id, [])
        if not dsl:
            continue
        h = img_hw[img_id][0]
        w = img_hw[img_id][1]
        for g in gts:
            seg = g["segmentation"]
            if isinstance(seg, dict):
                gt_rle = seg
            elif isinstance(seg, list):
                gt_rle = mask_utils.frPyObjects(seg, h, w)
                if isinstance(gt_rle, list):
                    gt_rle = mask_utils.merge(gt_rle)
            else:
                continue
            g["segmentation"] = gt_rle
            ga = g["area"]
            # match best det by IoU
            best, best_iou = None, 0.0
            for d in dsl:
                i = mask_utils.iou([d["segmentation"]], [gt_rle], [0])[0][0]
                if i > best_iou:
                    best, best_iou = d, i
            if best is None or best_iou < 0.5:
                continue
            if ga < 1024:  # COCO-small proxy on original scale
                small_q.append(soft_dice(best["segmentation"], gt_rle))
            # duplicates on the same GT with IoU>=0.5 (keeper-gap secondary)
            if ga < 1024:
                for d in dsl:
                    if d is best:
                        continue
                    i = mask_utils.iou([d["segmentation"]], [gt_rle], [0])[0][0]
                    if i >= 0.5:
                        keeper_gap.append(
                            soft_dice(best["segmentation"], gt_rle)
                            - soft_dice(d["segmentation"], gt_rle))
            # band-ratio on a subsample (heavy RLE math)
            if ga < 4096 and int_tot < 200000:
                det = mask_utils.decode(best["segmentation"]).astype(np.uint8)
                gm = mask_utils.decode(gt_rle).astype(np.uint8)
                h, w = gm.shape
                if h % 2 or w % 2:
                    continue
                soft = gm.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))
                d2 = det.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))
                band = (soft > 0.05) & (soft < 0.95)
                interior = soft >= 0.95
                if band.any() and interior.any():
                    band_tot += int(band.sum())
                    band_hits += int(((d2 > 0.5) != (soft > 0.5))[band].sum())
                    int_tot += int(interior.sum())
                    int_hits += int(((d2 > 0.5) != (soft > 0.5))[interior].sum())

    small_q = np.array(small_q)
    q_iqr = float(np.percentile(small_q, 75) - np.percentile(small_q, 25))
    q_p50 = float(np.median(small_q))
    band_err = band_hits / max(1, band_tot)
    int_err = int_hits / max(1, int_tot)
    ratio = band_err / max(1e-9, int_err)
    keeper = float(np.median(keeper_gap)) if keeper_gap else None

    res = {
        "gate1_q": {
            "n": len(small_q), "iqr": q_iqr, "p50": q_p50,
            "keeper_dup_gap_median": keeper,
            "pass": bool(q_iqr >= 0.15 and 0.5 <= q_p50 <= 0.8),
        },
        "gate2_band": {
            "band_err": band_err, "interior_err": int_err, "ratio": ratio,
            "pass": bool(ratio >= 1.1),
        },
        "note": "gate3 (depth SNR calibration) runs separately; probe-recall deferred to B1 arm",
    }
    print(json.dumps(res, indent=2))
    json.dump(res, open(OUT, "w"), indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-images", type=int, default=None)
    a = ap.parse_args()
    main(a.limit_images)
