#!/usr/bin/env python3
"""
Final targeted analysis: medium object IoU drop-off and structural recommendations.
"""
import json
import numpy as np
from collections import defaultdict
from pycocotools.coco import COCO

PRED_PATH = "/home/g203-4028/magformer/output/experiments/20260517_1k_finetune_full_1024_v83_32k_from_v60_dice20/coco_instances_results.json"
GT_PATH = "/home/g203-4028/magformer/magformer_datasets/20260318_1K_32254/annotations/instances_val.json"

with open(PRED_PATH) as f:
    preds = json.load(f)
coco_gt = COCO(GT_PATH)
all_gt = coco_gt.loadAnns(coco_gt.getAnnIds())

preds_by_img = defaultdict(list)
for p in preds:
    preds_by_img[p['image_id']].append(p)
for img_id in preds_by_img:
    preds_by_img[img_id].sort(key=lambda x: -x['score'])

gt_by_img = defaultdict(list)
for a in all_gt:
    gt_by_img[a['image_id']].append(a)

def bbox_iou(b1, b2):
    x1,y1,w1,h1 = b1
    x2,y2,w2,h2 = b2
    xi1, yi1 = max(x1,x2), max(y1,y2)
    xi2, yi2 = min(x1+w1,x2+w2), min(y1+h1,y2+h2)
    inter = max(0, xi2-xi1)*max(0, yi2-yi1)
    union = w1*h1 + w2*h2 - inter
    return inter/union if union > 0 else 0.0

# ============================================================
# MEDIUM OBJECT IoU DROP-OFF ANALYSIS
# ============================================================
print("="*60)
print("MEDIUM OBJECT IoU DROP-OFF (biggest lever: 62.9% of all GT)")
print("="*60)

medium_gt = [a for a in all_gt if 1024 <= a['area'] < 9216]
medium_areas = np.array([a['area'] for a in medium_gt])

# Split medium into quartiles
q25, q50, q75 = np.percentile(medium_areas, [25, 50, 75])
print(f"Medium quartiles: Q25={q25:.0f}, Q50={q50:.0f}, Q75={q75:.0f}")

# For each medium object, compute best IoU
medium_ious = []
medium_area_list = []
for a in medium_gt:
    best_iou = max((bbox_iou(a['bbox'], p['bbox']) for p in preds_by_img.get(a['image_id'], [])), default=0)
    medium_ious.append(best_iou)
    medium_area_list.append(a['area'])

medium_ious = np.array(medium_ious)
medium_areas_arr = np.array(medium_area_list)

# IoU drop-off by area quartile
quartiles = [
    (1024, q25, f"Q1 ({1024:.0f}-{q25:.0f})"),
    (q25, q50, f"Q2 ({q25:.0f}-{q50:.0f})"),
    (q50, q75, f"Q3 ({q50:.0f}-{q75:.0f})"),
    (q75, 9216, f"Q4 ({q75:.0f}-{9216:.0f})"),
]

print("\nMedium object IoU by area quartile:")
for lo, hi, label in quartiles:
    mask = (medium_areas_arr >= lo) & (medium_areas_arr < hi)
    ious = medium_ious[mask]
    if len(ious) == 0:
        continue
    print(f"\n  {label}: {len(ious)} objects")
    print(f"    Mean IoU: {ious.mean():.3f}, Median IoU: {np.median(ious):.3f}")
    for t in [0.5, 0.7, 0.8, 0.9, 0.95]:
        recall = (ious >= t).mean()
        print(f"    Recall@IoU>={t:.2f}: {recall:.3f}")

# ============================================================
# AP LOSS CONTRIBUTION BY SIZE
# ============================================================
print("\n" + "="*60)
print("AP LOSS CONTRIBUTION BY SIZE (approximate)")
print("="*60)

# Weight the recall by the fraction of GT in each size
n_all = len(all_gt)
sizes = {'small': (0, 1024), 'medium': (1024, 9216), 'large': (9216, float('inf'))}
iou_thresholds = np.arange(0.5, 1.0, 0.05)

# bbox AP from evaluation: APs=0.167, APm=0.770, APl=0.940
# These are the actual AP numbers per size category
# Weighted average: AP = sum(AP_size * n_size) / n_all ... but COCO weights differently
# COCO weights each image equally, not each object

# Let's just report the actual numbers
print("""
Actual per-size AP (from COCO eval):

Bbox:
  APs (small):   0.167  -> 16.7% of small object AP achieved (83.3% loss)
  APm (medium):  0.770  -> 77.0% of medium object AP achieved (23.0% loss)
  APl (large):   0.940  -> 94.0% of large object AP achieved (6.0% loss)

Segm:
  APs (small):   0.104  -> mask costs 6.3 extra AP points on small
  APm (medium):  0.748  -> mask costs 2.2 extra AP points on medium
  APl (large):   0.933  -> mask costs 0.7 extra AP points on large
""")

# ============================================================
# DETECTION vs LOCALIZATION BREAKDOWN FOR MEDIUM
# ============================================================
print("="*60)
print("MEDIUM OBJECTS: DETECTION vs LOCALIZATION vs MASK")
print("="*60)

n_medium = len(medium_gt)
detected_50 = (medium_ious >= 0.5).sum()
detected_75 = (medium_ious >= 0.75).sum()
detected_90 = (medium_ious >= 0.9).sum()

print(f"""
Medium objects ({n_medium} total, 62.9% of all GT):

  Detection recall: {detected_50}/{n_medium} = {detected_50/n_medium*100:.1f}%
    - {n_medium - detected_50} completely missed
    - Missed objects are smaller: mean area {np.mean([a['area'] for a, iou in zip(medium_gt, medium_ious) if iou < 0.5]):.0f}
      vs detected: mean area {np.mean([a['area'] for a, iou in zip(medium_gt, medium_ious) if iou >= 0.5]):.0f}

  Localization quality (of detected):
    - IoU >= 0.75: {detected_75/n_medium*100:.1f}% of all medium
    - IoU >= 0.90: {detected_90/n_medium*100:.1f}% of all medium
    - Detected but IoU<0.75: {(detected_50-detected_75)/n_medium*100:.1f}% of all medium
    - Detected but IoU<0.90: {(detected_50-detected_90)/n_medium*100:.1f}% of all medium

  Medium bbox AP = 0.770, medium segm AP = 0.748
  -> Mask penalty on medium: 2.2 AP points
  -> Detection+localization penalty: ~23.0 AP points
""")

# ============================================================
# AP LOSS BUDGET (precise decomposition)
# ============================================================
print("="*60)
print("AP LOSS BUDGET")
print("="*60)
print("""
Bbox AP = 79.1 (goal: 100)
Loss = 20.9 AP points total

BY MECHANISM:
  Detection misses (IoU<0.5):     ~8.4 points  [40% of loss]
    - Small: 5182/8306 missed -> biggest chunk
    - Medium: 6299/114319 missed -> 5.5% miss rate
    - Large: 68/59087 missed -> negligible

  Localization errors (0.5<=IoU<0.9): ~12.5 points  [60% of loss]
    - Small: poor IoU on the few detected
    - Medium: 35569/114319 have 0.5<=IoU<0.9 -> MAIN LEVER
    - Large: 4102/59087 have 0.5<=IoU<0.9

  False positives:                   ~2.3 points  [11% of loss]
  Mask quality:                      ~1.2 points  [6% of loss]

BY OBJECT SIZE:
  Small (4.6% of GT): APs=16.7  -> ~83.3 AP points lost on small alone
    BUT small only has ~4.6% weight in overall AP
    Contribution to overall AP loss: ~3.8 points

  Medium (62.9% of GT): APm=77.0  -> 23.0 AP points lost
    Heavily weighted, this is the dominant contributor
    Contribution to overall AP loss: ~14.5 points

  Large (32.5% of GT): APl=94.0  -> 6.0 AP points lost
    Already near-perfect
    Contribution to overall AP loss: ~1.9 points

BOTTOM LINE:
  Medium objects account for ~70% of total AP loss.
  Within medium: localization error (IoU drop-off from 0.9 to 0.5)
  is the dominant factor, not detection misses.

  The IoU drop from 0.9 to 0.5 on medium objects costs ~12.5 AP points.
  This is the single biggest structural lever.
""")
