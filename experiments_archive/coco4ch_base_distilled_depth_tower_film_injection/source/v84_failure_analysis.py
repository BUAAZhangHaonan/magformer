#!/usr/bin/env python3
"""
Deep failure mode analysis of v83 model.
Analyzes COCO predictions against ground truth to decompose AP loss.
"""
import json
import os
import sys
import numpy as np
from collections import Counter, defaultdict
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

# Paths
PRED_PATH = "/home/g203-4028/magformer/output/experiments/20260519_1k_finetune_full_1024_v84_pruned_32k_from_v60_dice20/coco_instances_results.json"
GT_PATH = "/home/g203-4028/magformer/magformer_datasets/20260318_1K_32254/annotations/instances_val.json"

print(f"Loading predictions from {PRED_PATH}")
print(f"Loading GT from {GT_PATH}")

# Load predictions
with open(PRED_PATH) as f:
    preds = json.load(f)
print(f"\nTotal predictions: {len(preds)}")

# Load GT
coco_gt = COCO(GT_PATH)

# Get image IDs from predictions
pred_img_ids = set(p['image_id'] for p in preds)
gt_img_ids = set(coco_gt.getImgIds())
print(f"Unique images in predictions: {len(pred_img_ids)}")
print(f"Unique images in GT: {len(gt_img_ids)}")
print(f"Overlap: {len(pred_img_ids & gt_img_ids)}")

# ============================================================
# 1. BASIC PREDICTION STATISTICS
# ============================================================
print("\n" + "="*60)
print("1. BASIC PREDICTION STATISTICS")
print("="*60)

scores = [p['score'] for p in preds]
print(f"Score range: {min(scores):.4f} - {max(scores):.4f}")
print(f"Score mean: {np.mean(scores):.4f}, median: {np.median(scores):.4f}")

# Score distribution
for thresh in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95]:
    n = sum(1 for s in scores if s >= thresh)
    print(f"  Score >= {thresh:.2f}: {n} predictions ({n/len(scores)*100:.1f}%)")

# Predictions per image
img_pred_counts = defaultdict(list)
for p in preds:
    img_pred_counts[p['image_id']].append(p)
preds_per_img = [len(v) for v in img_pred_counts.values()]
print(f"\nPredictions per image: min={min(preds_per_img)}, max={max(preds_per_img)}, "
      f"mean={np.mean(preds_per_img):.1f}, median={np.median(preds_per_img):.1f}")
print(f"Images with >= 200 predictions: {sum(1 for c in preds_per_img if c >= 200)}")

# ============================================================
# 2. GT OBJECT SIZE DISTRIBUTION
# ============================================================
print("\n" + "="*60)
print("2. GT OBJECT SIZE DISTRIBUTION")
print("="*60)

# COCO size categories based on bbox area
# small: area < 32^2 = 1024
# medium: 1024 <= area < 96^2 = 9216
# large: area >= 9216
all_gt_anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=list(gt_img_ids)))
print(f"Total GT annotations: {len(all_gt_anns)}")

size_categories = {'small': [], 'medium': [], 'large': []}
for ann in all_gt_anns:
    area = ann.get('area', 0)
    if area < 1024:
        size_categories['small'].append(ann)
    elif area < 9216:
        size_categories['medium'].append(ann)
    else:
        size_categories['large'].append(ann)

for cat, anns in size_categories.items():
    print(f"  {cat}: {len(anns)} ({len(anns)/len(all_gt_anns)*100:.1f}%)")

# GT objects per image
gt_per_img = defaultdict(list)
for ann in all_gt_anns:
    gt_per_img[ann['image_id']].append(ann)
gt_counts = [len(v) for v in gt_per_img.values()]
print(f"\nGT objects per image: min={min(gt_counts)}, max={max(gt_counts)}, "
      f"mean={np.mean(gt_counts):.1f}, median={np.median(gt_counts):.1f}")
print(f"Images with > 100 GT objects: {sum(1 for c in gt_counts if c > 100)}")
print(f"Images with > 200 GT objects: {sum(1 for c in gt_counts if c > 200)}")

# GT size distribution per image
gt_small_per_img = []
gt_med_per_img = []
gt_large_per_img = []
for img_id in gt_per_img:
    anns = gt_per_img[img_id]
    s = sum(1 for a in anns if a.get('area', 0) < 1024)
    m = sum(1 for a in anns if 1024 <= a.get('area', 0) < 9216)
    l = sum(1 for a in anns if a.get('area', 0) >= 9216)
    gt_small_per_img.append(s)
    gt_med_per_img.append(m)
    gt_large_per_img.append(l)

print(f"\nSmall GT per image: mean={np.mean(gt_small_per_img):.1f}, max={max(gt_small_per_img)}")
print(f"Medium GT per image: mean={np.mean(gt_med_per_img):.1f}, max={max(gt_med_per_img)}")
print(f"Large GT per image: mean={np.mean(gt_large_per_img):.1f}, max={max(gt_large_per_img)}")

# ============================================================
# 3. RUN COCO EVAL FOR DETAILED METRICS
# ============================================================
print("\n" + "="*60)
print("3. COCO EVALUATION - DETAILED METRICS")
print("="*60)

coco_dt = coco_gt.loadRes(preds)
coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

print("\n--- Per-category bbox AP ---")
for i, cat_id in enumerate(coco_eval.params.catIds):
    coco_eval.params.catIds = [cat_id]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

# Reset
coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
coco_eval.evaluate()
coco_eval.accumulate()

# ============================================================
# 4. PER-INSTANCE IoU ANALYSIS (MATCHING)
# ============================================================
print("\n" + "="*60)
print("4. PER-INSTANCE MATCHING ANALYSIS")
print("="*60)

# For each GT annotation, find best matching prediction
# We'll use IoU-based matching

def compute_bbox_iou(box1, box2):
    """Compute IoU between two bboxes [x,y,w,h]"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1+w1, x2+w2)
    yi2 = min(y1+h1, y2+h2)

    inter = max(0, xi2-xi1) * max(0, yi2-yi1)
    union = w1*h1 + w2*h2 - inter

    if union <= 0:
        return 0.0
    return inter / union

# Build prediction index by image
preds_by_img = defaultdict(list)
for p in preds:
    preds_by_img[p['image_id']].append(p)

# Sort predictions by score descending within each image
for img_id in preds_by_img:
    preds_by_img[img_id].sort(key=lambda x: -x['score'])

# Match GT to predictions at various IoU thresholds
all_best_ious = []
gt_matched_50 = 0
gt_matched_75 = 0
gt_matched_90 = 0
gt_matched_95 = 0
gt_unmatched = 0

# Per-size matching
size_ious = {'small': [], 'medium': [], 'large': []}
size_matched_50 = {'small': 0, 'medium': 0, 'large': 0}

for ann in all_gt_anns:
    img_id = ann['image_id']
    gt_bbox = ann['bbox']
    gt_area = ann.get('area', 0)

    # Determine size
    if gt_area < 1024:
        sz = 'small'
    elif gt_area < 9216:
        sz = 'medium'
    else:
        sz = 'large'

    # Find best matching prediction
    best_iou = 0.0
    for p in preds_by_img.get(img_id, []):
        iou = compute_bbox_iou(gt_bbox, p['bbox'])
        if iou > best_iou:
            best_iou = iou

    all_best_ious.append(best_iou)
    size_ious[sz].append(best_iou)

    if best_iou >= 0.5:
        gt_matched_50 += 1
        size_matched_50[sz] += 1
    if best_iou >= 0.75:
        gt_matched_75 += 1
    if best_iou >= 0.90:
        gt_matched_90 += 1
    if best_iou >= 0.95:
        gt_matched_95 += 1
    if best_iou < 0.5:
        gt_unmatched += 1

print(f"Total GT objects: {len(all_gt_anns)}")
print(f"Matched at IoU>=0.50: {gt_matched_50} ({gt_matched_50/len(all_gt_anns)*100:.1f}%)")
print(f"Matched at IoU>=0.75: {gt_matched_75} ({gt_matched_75/len(all_gt_anns)*100:.1f}%)")
print(f"Matched at IoU>=0.90: {gt_matched_90} ({gt_matched_90/len(all_gt_anns)*100:.1f}%)")
print(f"Matched at IoU>=0.95: {gt_matched_95} ({gt_matched_95/len(all_gt_anns)*100:.1f}%)")
print(f"Complete misses (IoU<0.50): {gt_unmatched} ({gt_unmatched/len(all_gt_anns)*100:.1f}%)")

# IoU distribution for matched objects
matched_ious = [iou for iou in all_best_ious if iou >= 0.5]
print(f"\nIoU distribution for matched objects (IoU>=0.5):")
print(f"  Count: {len(matched_ious)}")
if matched_ious:
    print(f"  Mean IoU: {np.mean(matched_ious):.4f}")
    print(f"  Median IoU: {np.median(matched_ious):.4f}")
    for thresh in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        n = sum(1 for iou in matched_ious if iou >= thresh)
        print(f"  IoU >= {thresh:.2f}: {n} ({n/len(matched_ious)*100:.1f}%)")

# IoU distribution buckets
print("\nIoU bucket distribution (all GT):")
buckets = [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
           (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
for lo, hi in buckets:
    n = sum(1 for iou in all_best_ious if lo <= iou < hi)
    print(f"  [{lo:.1f}, {hi:.1f}): {n} ({n/len(all_best_ious)*100:.1f}%)")

# Per-size matching rates
print("\nPer-size matching analysis:")
for sz in ['small', 'medium', 'large']:
    ious = size_ious[sz]
    n_total = len(ious)
    if n_total == 0:
        continue
    n_50 = sum(1 for i in ious if i >= 0.5)
    n_75 = sum(1 for i in ious if i >= 0.75)
    n_90 = sum(1 for i in ious if i >= 0.90)
    print(f"  {sz}: {n_total} objects, matched@50={n_50}({n_50/n_total*100:.1f}%), "
          f"@75={n_75}({n_75/n_total*100:.1f}%), @90={n_90}({n_90/n_total*100:.1f}%)")
    if n_50 > 0:
        matched = [i for i in ious if i >= 0.5]
        print(f"    Mean IoU (matched@50): {np.mean(matched):.4f}")

# ============================================================
# 5. PER-IMAGE RECALL ANALYSIS
# ============================================================
print("\n" + "="*60)
print("5. PER-IMAGE RECALL ANALYSIS")
print("="*60)

per_image_recall = []
per_image_recall_small = []
per_image_recall_medium = []
per_image_recall_large = []
images_zero_recall = 0

for img_id in gt_per_img:
    anns = gt_per_img[img_id]
    n_gt = len(anns)
    n_matched = sum(1 for a in anns if any(
        compute_bbox_iou(a['bbox'], p['bbox']) >= 0.5
        for p in preds_by_img.get(img_id, [])
    ))
    recall = n_matched / n_gt if n_gt > 0 else 1.0
    per_image_recall.append(recall)
    if recall == 0:
        images_zero_recall += 1

    # Per-size recall
    for sz_key, sz_thresh_low, sz_thresh_high in [('small', 0, 1024), ('medium', 1024, 9216), ('large', 9216, float('inf'))]:
        sz_anns = [a for a in anns if sz_thresh_low <= a.get('area', 0) < sz_thresh_high]
        if sz_anns:
            sz_matched = sum(1 for a in sz_anns if any(
                compute_bbox_iou(a['bbox'], p['bbox']) >= 0.5
                for p in preds_by_img.get(img_id, [])
            ))
            sz_recall = sz_matched / len(sz_anns)
            if sz_key == 'small':
                per_image_recall_small.append(sz_recall)
            elif sz_key == 'medium':
                per_image_recall_medium.append(sz_recall)
            else:
                per_image_recall_large.append(sz_recall)

print(f"Per-image recall distribution:")
print(f"  Mean: {np.mean(per_image_recall):.4f}")
print(f"  Median: {np.median(per_image_recall):.4f}")
print(f"  Min: {np.min(per_image_recall):.4f}")
print(f"  Images with 0% recall: {images_zero_recall}")
print(f"  Images with <50% recall: {sum(1 for r in per_image_recall if r < 0.5)}")
print(f"  Images with <80% recall: {sum(1 for r in per_image_recall if r < 0.8)}")
print(f"  Images with 100% recall: {sum(1 for r in per_image_recall if r >= 1.0)}")

# Recall distribution buckets
print("\nRecall distribution:")
for lo, hi in [(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 0.9), (0.9, 1.0)]:
    n = sum(1 for r in per_image_recall if lo <= r < hi)
    print(f"  [{lo:.2f}, {hi:.2f}): {n} images")

for name, recs in [('small', per_image_recall_small), ('medium', per_image_recall_medium), ('large', per_image_recall_large)]:
    if recs:
        print(f"\n{name} object recall: mean={np.mean(recs):.4f}, "
              f"images with 0% recall: {sum(1 for r in recs if r == 0)}")

# ============================================================
# 6. FALSE POSITIVE ANALYSIS
# ============================================================
print("\n" + "="*60)
print("6. FALSE POSITIVE ANALYSIS")
print("="*60)

# For each prediction, check if it matches any GT
tp_scores = []
fp_scores = []
tp_count = 0
fp_count = 0
fn_count = 0

for img_id in pred_img_ids:
    img_preds = preds_by_img[img_id]
    img_gts = gt_per_img.get(img_id, [])

    # Track which GTs are already matched
    gt_matched = set()

    # Sort predictions by score
    for p in sorted(img_preds, key=lambda x: -x['score']):
        best_iou = 0
        best_gt_idx = -1
        for i, gt in enumerate(img_gts):
            if i in gt_matched:
                continue
            iou = compute_bbox_iou(gt['bbox'], p['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i

        if best_iou >= 0.5:
            tp_scores.append(p['score'])
            tp_count += 1
            gt_matched.add(best_gt_idx)
        else:
            fp_scores.append(p['score'])
            fp_count += 1

    # Count false negatives
    fn_count += len(img_gts) - len(gt_matched)

print(f"True positives (IoU>=0.5): {tp_count}")
print(f"False positives: {fp_count}")
print(f"False negatives (missed GT): {fn_count}")
print(f"Total predictions: {tp_count + fp_count}")

if tp_scores:
    print(f"\nTP score distribution:")
    print(f"  Mean: {np.mean(tp_scores):.4f}, Median: {np.median(tp_scores):.4f}")
    print(f"  Min: {np.min(tp_scores):.4f}, Max: {np.max(tp_scores):.4f}")

if fp_scores:
    print(f"\nFP score distribution:")
    print(f"  Mean: {np.mean(fp_scores):.4f}, Median: {np.median(fp_scores):.4f}")
    print(f"  Min: {np.min(fp_scores):.4f}, Max: {np.max(fp_scores):.4f}")

    # Score threshold analysis
    for thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        n_fp = sum(1 for s in fp_scores if s >= thresh)
        n_tp = sum(1 for s in tp_scores if s >= thresh)
        print(f"  At thresh {thresh:.1f}: TP={n_tp}, FP={n_fp}, precision={n_tp/(n_tp+n_fp)*100:.1f}%")

# ============================================================
# 7. QUERY BUDGET ANALYSIS
# ============================================================
print("\n" + "="*60)
print("7. QUERY BUDGET ANALYSIS")
print("="*60)

# How often do GT objects exceed 200?
over_200 = 0
at_limit = 0
for img_id in gt_per_img:
    n_gt = len(gt_per_img[img_id])
    n_pred = len(preds_by_img.get(img_id, []))
    if n_gt > 200:
        over_200 += 1
    if n_pred >= 200:
        at_limit += 1

print(f"Images with GT objects > 200: {over_200}")
print(f"Images with 200 predictions (query budget saturated): {at_limit}")
print(f"Fraction of predictions at max: {at_limit/len(gt_per_img)*100:.1f}%")

# For images at query limit, how many GT are unmatched?
if at_limit > 0:
    unmatched_at_limit = []
    for img_id in gt_per_img:
        n_pred = len(preds_by_img.get(img_id, []))
        if n_pred >= 200:
            n_gt = len(gt_per_img[img_id])
            n_matched = 0
            gt_used = set()
            for p in sorted(preds_by_img[img_id], key=lambda x: -x['score']):
                for i, gt in enumerate(gt_per_img[img_id]):
                    if i not in gt_used:
                        if compute_bbox_iou(gt['bbox'], p['bbox']) >= 0.5:
                            n_matched += 1
                            gt_used.add(i)
                            break
            unmatched_at_limit.append(n_gt - n_matched)

    print(f"\nAt-query-limit images:")
    print(f"  Mean unmatched GT: {np.mean(unmatched_at_limit):.1f}")
    print(f"  Max unmatched GT: {max(unmatched_at_limit) if unmatched_at_limit else 0}")
    print(f"  Total unmatched GT at limit: {sum(unmatched_at_limit)}")

# ============================================================
# 8. AP LOSS DECOMPOSITION
# ============================================================
print("\n" + "="*60)
print("8. AP LOSS DECOMPOSITION")
print("="*60)

total_gt = len(all_gt_anns)
bbox_ar_100 = gt_matched_50 / total_gt  # Approximate AR@100

# AP loss sources
ap_loss_missed = gt_unmatched  # Complete detection misses
ap_loss_poor_iou = gt_matched_50 - gt_matched_75  # Detected but low IoU
ap_loss_very_poor_iou = gt_matched_75 - gt_matched_90  # Detected, medium IoU
ap_loss_excellent = gt_matched_90  # These contribute most to AP

print(f"\nDetection quality breakdown:")
print(f"  Complete misses (IoU<0.5): {ap_loss_missed} ({ap_loss_missed/total_gt*100:.1f}%)")
print(f"  Weak detections (0.5<=IoU<0.75): {ap_loss_poor_iou} ({ap_loss_poor_iou/total_gt*100:.1f}%)")
print(f"  Medium detections (0.75<=IoU<0.90): {ap_loss_very_poor_iou} ({ap_loss_very_poor_iou/total_gt*100:.1f}%)")
print(f"  Strong detections (IoU>=0.90): {ap_loss_excellent} ({ap_loss_excellent/total_gt*100:.1f}%)")

print(f"\nAP loss by source (approximate):")
print(f"  Detection misses: {ap_loss_missed/total_gt*100:.1f}% of GT -> major AP loss at all thresholds")
print(f"  Localization error: {ap_loss_poor_iou/total_gt*100:.1f}% of GT -> major AP loss at AP@75+")
print(f"  High-precision error: {(total_gt - gt_matched_90)/total_gt*100:.1f}% of GT -> AP@90/AP@95 loss")
print(f"  False positives: {fp_count} total -> precision penalty")

# Estimated AP ceiling if masks were perfect
print(f"\nEstimated AP ceilings (if mask quality were perfect):")
print(f"  Current bbox AP ~79.08, current segm AP ~77.90")
print(f"  Mask-to-bbox gap: ~1.18 AP points")
print(f"  If all detections had IoU=1.0, AP ceiling = AR = {bbox_ar_100*100:.1f}")
print(f"  Remaining gap to 100 AP = {100 - bbox_ar_100*100:.1f} points (from missed detections)")

print("\n" + "="*60)
print("SUMMARY: TOP FAILURE MODES")
print("="*60)
print(f"1. Detection misses: {ap_loss_missed/total_gt*100:.1f}% of GT objects completely missed")
print(f"2. Small objects: {len(size_ious['small'])} total, only {sum(1 for i in size_ious['small'] if i >= 0.5)} detected@50")
print(f"3. Medium objects: {len(size_ious['medium'])} total, only {sum(1 for i in size_ious['medium'] if i >= 0.5)} detected@50")
print(f"4. Localization: {ap_loss_poor_iou} objects detected but IoU<0.75")
print(f"5. FP count: {fp_count} total false positives")
print(f"6. Query budget: {at_limit} images at 200-query limit")
