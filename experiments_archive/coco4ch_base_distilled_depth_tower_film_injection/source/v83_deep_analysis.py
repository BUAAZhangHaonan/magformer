#!/usr/bin/env python3
"""
Deeper failure mode analysis: quantified AP loss decomposition,
small object deep dive, and localization error analysis.
"""
import json
import os
import numpy as np
from collections import defaultdict, Counter
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

PRED_PATH = "/home/g203-4028/magformer/output/experiments/20260517_1k_finetune_full_1024_v83_32k_from_v60_dice20/coco_instances_results.json"
GT_PATH = "/home/g203-4028/magformer/magformer_datasets/20260318_1K_32254/annotations/instances_val.json"

# Load data
with open(PRED_PATH) as f:
    preds = json.load(f)
coco_gt = COCO(GT_PATH)
all_gt_anns = coco_gt.loadAnns(coco_gt.getAnnIds())
gt_by_img = defaultdict(list)
for a in all_gt_anns:
    gt_by_img[a['image_id']].append(a)

preds_by_img = defaultdict(list)
for p in preds:
    preds_by_img[p['image_id']].append(p)
for img_id in preds_by_img:
    preds_by_img[img_id].sort(key=lambda x: -x['score'])

def bbox_iou(b1, b2):
    x1,y1,w1,h1 = b1
    x2,y2,w2,h2 = b2
    xi1, yi1 = max(x1,x2), max(y1,y2)
    xi2, yi2 = min(x1+w1,x2+w2), min(y1+h1,y2+h2)
    inter = max(0, xi2-xi1)*max(0, yi2-yi1)
    union = w1*h1 + w2*h2 - inter
    return inter/union if union > 0 else 0.0

def get_size(area):
    if area < 1024: return 'small'
    elif area < 9216: return 'medium'
    else: return 'large'

# ============================================================
# DEEP DIVE: Small object analysis
# ============================================================
print("="*60)
print("DEEP DIVE 1: SMALL OBJECT ANALYSIS")
print("="*60)

small_gt = [a for a in all_gt_anns if a['area'] < 1024]
print(f"Total small GT objects: {len(small_gt)} ({len(small_gt)/len(all_gt_anns)*100:.1f}%)")

# Size distribution within small
small_areas = [a['area'] for a in small_gt]
print(f"\nSmall object area distribution:")
print(f"  Min area: {min(small_areas)}")
print(f"  Max area: {max(small_areas)}")
print(f"  Mean area: {np.mean(small_areas):.0f}")
for thresh in [100, 200, 300, 400, 500, 700, 1024]:
    n = sum(1 for a in small_areas if a < thresh)
    print(f"  Area < {thresh}: {n} ({n/len(small_areas)*100:.1f}%)")

# Bbox dimensions of small objects
small_bboxes = [(a['bbox'][2], a['bbox'][3]) for a in small_gt]
small_sides = [min(w,h) for w,h in small_bboxes]
print(f"\nSmall object min side distribution:")
for thresh in [8, 16, 20, 24, 28, 32]:
    n = sum(1 for s in small_sides if s < thresh)
    print(f"  Min side < {thresh}px: {n} ({n/len(small_sides)*100:.1f}%)")

# How many small objects are detected?
small_matched = 0
small_ious = []
for a in small_gt:
    best_iou = max((bbox_iou(a['bbox'], p['bbox']) for p in preds_by_img.get(a['image_id'], [])), default=0)
    small_ious.append(best_iou)
    if best_iou >= 0.5:
        small_matched += 1

print(f"\nSmall object detection rate: {small_matched}/{len(small_gt)} = {small_matched/len(small_gt)*100:.1f}%")
print(f"Small objects by detection quality:")
for lo, hi, label in [(0, 0.1, "no overlap"), (0.1, 0.3, "very weak"), (0.3, 0.5, "weak"), (0.5, 0.75, "ok"), (0.75, 1.0, "good")]:
    n = sum(1 for iou in small_ious if lo <= iou < hi)
    print(f"  {label} ({lo:.1f}-{hi:.1f}): {n} ({n/len(small_ious)*100:.1f}%)")

# Small objects per image and detection correlation
small_gt_per_img = defaultdict(list)
for a in small_gt:
    small_gt_per_img[a['image_id']].append(a)

print(f"\nImages with small objects: {len(small_gt_per_img)} / {len(gt_by_img)}")
img_small_recall = []
for img_id, small_anns in small_gt_per_img.items():
    n_detected = sum(1 for a in small_anns if max(
        (bbox_iou(a['bbox'], p['bbox']) for p in preds_by_img.get(img_id, [])), default=0
    ) >= 0.5)
    img_small_recall.append(n_detected / len(small_anns))

print(f"Per-image small object recall: mean={np.mean(img_small_recall):.3f}")
print(f"Images with 0% small object recall: {sum(1 for r in img_small_recall if r == 0)}")
print(f"Images with 100% small object recall: {sum(1 for r in img_small_recall if r >= 1.0)}")

# ============================================================
# DEEP DIVE 2: MEDIUM OBJECT DETECTION FAILURES
# ============================================================
print("\n" + "="*60)
print("DEEP DIVE 2: MEDIUM OBJECT ANALYSIS")
print("="*60)

medium_gt = [a for a in all_gt_anns if 1024 <= a['area'] < 9216]
medium_unmatched = [a for a in medium_gt if max(
    (bbox_iou(a['bbox'], p['bbox']) for p in preds_by_img.get(a['image_id'], [])), default=0
) < 0.5]

print(f"Total medium GT: {len(medium_gt)}")
print(f"Medium misses: {len(medium_unmatched)} ({len(medium_unmatched)/len(medium_gt)*100:.2f}%)")

# Area distribution of missed medium objects
missed_areas = [a['area'] for a in medium_unmatched]
detected_areas = [a['area'] for a in medium_gt if a not in medium_unmatched]
print(f"\nMissed medium objects area: mean={np.mean(missed_areas):.0f}, median={np.median(missed_areas):.0f}")
print(f"Detected medium objects area: mean={np.mean(detected_areas):.0f}, median={np.median(detected_areas):.0f}")

# Missed medium objects by area quartile
area_q = np.percentile([a['area'] for a in medium_gt], [25, 50, 75])
print(f"\nMedium area quartiles: Q25={area_q[0]:.0f}, Q50={area_q[1]:.0f}, Q75={area_q[2]:.0f}")

# ============================================================
# DEEP DIVE 3: LOCALIZATION ERROR ANALYSIS
# ============================================================
print("\n" + "="*60)
print("DEEP DIVE 3: LOCALIZATION ERROR ANALYSIS")
print("="*60)

# Objects detected at IoU>=0.5 but <0.9 - these hurt AP@75, AP@90
weak_det = []
for a in all_gt_anns:
    best_iou = max((bbox_iou(a['bbox'], p['bbox']) for p in preds_by_img.get(a['image_id'], [])), default=0)
    if 0.5 <= best_iou < 0.9:
        weak_det.append((a, best_iou))

print(f"Objects with 0.5 <= IoU < 0.9: {len(weak_det)} ({len(weak_det)/len(all_gt_anns)*100:.1f}%)")

# Size breakdown of weak detections
weak_by_size = defaultdict(list)
for a, iou in weak_det:
    weak_by_size[get_size(a['area'])].append(iou)

for sz in ['small', 'medium', 'large']:
    if sz in weak_by_size:
        ious = weak_by_size[sz]
        print(f"  {sz}: {len(ious)} objects, mean IoU={np.mean(ious):.3f}")

# ============================================================
# DEEP DIVE 4: FALSE POSITIVE ANALYSIS
# ============================================================
print("\n" + "="*60)
print("DEEP DIVE 4: FALSE POSITIVE DEEP DIVE")
print("="*60)

# Count FP by score bucket
fp_by_score = defaultdict(int)
tp_by_score = defaultdict(int)
for img_id in preds_by_img:
    img_preds = preds_by_img[img_id]
    img_gts = gt_by_img.get(img_id, [])
    gt_matched = set()

    for p in sorted(img_preds, key=lambda x: -x['score']):
        best_iou = 0
        best_gt_idx = -1
        for i, gt in enumerate(img_gts):
            if i in gt_matched:
                continue
            iou = bbox_iou(gt['bbox'], p['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i

        score_bucket = int(p['score'] * 20) / 20  # 0.05 buckets
        if best_iou >= 0.5:
            tp_by_score[score_bucket] += 1
            gt_matched.add(best_gt_idx)
        else:
            fp_by_score[score_bucket] += 1

print("FP vs TP by score bucket:")
for bucket in sorted(set(list(fp_by_score.keys()) + list(tp_by_score.keys()))):
    fp = fp_by_score.get(bucket, 0)
    tp = tp_by_score.get(bucket, 0)
    total = fp + tp
    if total > 0:
        print(f"  [{bucket:.2f}, {bucket+0.05:.2f}): TP={tp}, FP={fp}, FP%={fp/total*100:.1f}%")

# ============================================================
# DEEP DIVE 5: AP LOSS QUANTIFICATION
# ============================================================
print("\n" + "="*60)
print("DEEP DIVE 5: QUANTIFIED AP LOSS DECOMPOSITION")
print("="*60)

# COCO AP is computed over IoU thresholds 0.5:0.95 in 0.05 steps
# For each threshold, we can compute what fraction of GT is detected
iou_thresholds = np.arange(0.5, 1.0, 0.05)
all_best_ious = []
gt_sizes = []

for a in all_gt_anns:
    best_iou = max((bbox_iou(a['bbox'], p['bbox']) for p in preds_by_img.get(a['image_id'], [])), default=0)
    all_best_ious.append(best_iou)
    gt_sizes.append(get_size(a['area']))

all_best_ious = np.array(all_best_ious)

print("Recall at each IoU threshold:")
for t in iou_thresholds:
    recall = (all_best_ious >= t).mean()
    print(f"  IoU >= {t:.2f}: recall = {recall:.4f} ({recall*100:.1f}%)")

# Per-size recall at each threshold
print("\nPer-size recall at each IoU threshold:")
for sz in ['small', 'medium', 'large']:
    sz_mask = np.array([s == sz for s in gt_sizes])
    sz_ious = all_best_ious[sz_mask]
    print(f"\n  {sz} ({sz_mask.sum()} objects):")
    for t in iou_thresholds:
        recall = (sz_ious >= t).mean()
        print(f"    IoU >= {t:.2f}: recall = {recall:.4f}")

# ============================================================
# DEEP DIVE 6: THE "LAST 20 AP POINTS" ANALYSIS
# ============================================================
print("\n" + "="*60)
print("DEEP DIVE 6: WHERE ARE THE MISSING ~21 AP POINTS?")
print("="*60)

# Current AP = 79.1 (bbox). Gap to 100 = 20.9 points.
# Decompose by: detection misses, localization errors, false positives

# The AP loss comes from 3 sources:
# 1. Missed detections (recall gap)
# 2. Poor localization (detected but low IoU)
# 3. False positives (precision penalty)

# Approximate the AP contribution:
# AP = integral of precision over recall curve
# At each IoU threshold t, precision-recall curve is computed

# Simple upper bound: if every GT had a perfect match (IoU=1.0) and no FPs
# AP would be 1.0. The gap from 1.0 to current AP tells us total loss.
# The gap from AR to AP tells us the FP penalty.
# The gap from 1.0 to AR tells us the recall + localization penalty.

bbox_ar = 0.814  # AR@100
bbox_ap = 0.791  # AP@[0.5:0.95]

print(f"Current bbox AP: {bbox_ap:.3f}")
print(f"Current bbox AR@100: {bbox_ar:.3f}")
print(f"AP-AR gap (FP penalty): {bbox_ar - bbox_ap:.3f} ({(bbox_ar-bbox_ap)/bbox_ap*100:.1f}% of AP)")
print(f"AR gap to 1.0 (detection+localization): {1.0-bbox_ar:.3f}")

# The AR@100 gap tells us about recall loss
# But AR is computed over IoU 0.5:0.95, so it also includes localization
# Let's separate: AR@50 = pure detection recall, AR@[0.5:0.95] includes localization
ar_50_recall = (all_best_ious >= 0.5).mean()
ar_75_recall = (all_best_ious >= 0.75).mean()
ar_90_recall = (all_best_ious >= 0.9).mean()
ar_95_recall = (all_best_ious >= 0.95).mean()

print(f"\nDetection-only recall (IoU>=0.50): {ar_50_recall:.4f}")
print(f"Good localization recall (IoU>=0.75): {ar_75_recall:.4f}")
print(f"High precision recall (IoU>=0.90): {ar_90_recall:.4f}")
print(f"Very high precision recall (IoU>=0.95): {ar_95_recall:.4f}")

# AP@50 ≈ detection-only performance
# AP@75 includes localization
# The gap AP@50 - AP@[0.5:0.95] is mostly localization
# The gap 1.0 - AP@50 is mostly detection misses
ap_50 = 0.916  # from eval output
ap_75 = 0.818
ap_all = 0.791

print(f"\nAP decomposition:")
print(f"  AP@50 = {ap_50:.3f} -> {(1-ap_50)*100:.1f}% loss from detection misses")
print(f"  AP@75 = {ap_75:.3f} -> gap from AP@50 = {(ap_50-ap_75)*100:.1f}% from localization@75")
print(f"  AP@[.5:.95] = {ap_all:.3f} -> gap from AP@50 = {(ap_50-ap_all)*100:.1f}% from localization across thresholds")
print(f"  AP@75 - AP@[.5:.95] = {(ap_75-ap_all)*100:.1f}% from localization@90+")

# ============================================================
# DEEP DIVE 7: MASK QUALITY ANALYSIS
# ============================================================
print("\n" + "="*60)
print("DEEP DIVE 7: BBOX vs MASK AP GAP")
print("="*60)

# Run segm eval too
print("Running segmentation evaluation...")
coco_dt_segm = coco_gt.loadRes(preds)
coco_eval_segm = COCOeval(coco_gt, coco_dt_segm, 'segm')
coco_eval_segm.evaluate()
coco_eval_segm.accumulate()
coco_eval_segm.summarize()

print(f"\nBbox AP: {ap_all:.3f}")
print(f"Segm AP: {coco_eval_segm.stats[0]:.3f}")
print(f"Mask quality penalty: {ap_all - coco_eval_segm.stats[0]:.3f}")

print("\nSegm per-size AP:")
print(f"  APs: {coco_eval_segm.stats[3]:.3f} (bbox: 0.167)")
print(f"  APm: {coco_eval_segm.stats[4]:.3f} (bbox: 0.770)")
print(f"  APl: {coco_eval_segm.stats[5]:.3f} (bbox: 0.940)")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "="*60)
print("FINAL SUMMARY: AP LOSS DECOMPOSITION")
print("="*60)
print(f"""
Total AP loss from 100: {100 - ap_all*100:.1f} points (bbox)

1. DETECTION MISSES (no match at IoU>=0.5):
   - {len([a for a in all_gt_anns if max((bbox_iou(a['bbox'],p['bbox']) for p in preds_by_img.get(a['image_id'],[])),default=0) < 0.5])} / {len(all_gt_anns)} GT objects missed
   - Impact: ~{(1-ap_50)*100:.1f} AP points lost at AP@50 level
   - Small objects: {sum(1 for a in all_gt_anns if a['area']<1024 and max((bbox_iou(a['bbox'],p['bbox']) for p in preds_by_img.get(a['image_id'],[])),default=0) < 0.5)} missed / {len([a for a in all_gt_anns if a['area']<1024])} total
   - Medium objects: {sum(1 for a in all_gt_anns if 1024<=a['area']<9216 and max((bbox_iou(a['bbox'],p['bbox']) for p in preds_by_img.get(a['image_id'],[])),default=0) < 0.5)} missed / {len([a for a in all_gt_anns if 1024<=a['area']<9216])} total

2. LOCALIZATION ERROR (detected but IoU < 0.9):
   - {sum(1 for iou in all_best_ious if 0.5 <= iou < 0.9)} objects detected but poorly localized
   - Impact: ~{(ap_50-ap_all)*100:.1f} AP points (AP@50 -> AP@[0.5:0.95] gap)

3. FALSE POSITIVES:
   - 32633 FP predictions out of 200318 total
   - Impact: ~{(bbox_ar-bbox_ap)*100:.1f} AP points (AR-AP gap)

4. MASK QUALITY:
   - Bbox-segm gap: ~{ap_all - coco_eval_segm.stats[0]:.1f} AP points

PRIORITY RANKING:
   a) Small object detection: 62.4% of small objects missed -> dominant AP loss source for APs
   b) Localization (IoU<0.9 on medium objects): largest volume of error objects
   c) False positives: 16.3% of predictions are FP
   d) Mask quality: relatively small (~1.2 AP points)
""")
