# R116 R114 Pseudo Quality Diagnosis

Date: 2026-05-18
Branch: `feature/vc-suda-sim2real`
Mode: no training; existing prediction JSON only.

## Setup

Teacher: R114 formal target150 iter2000.

Dataset: R114 `remaining75`.

Prediction JSON:

- `output/diagnostics/r114_magformer_r113warm_target150_iter2000_remaining75_1024_backmap_topk200_20260518/coco_instances_results.json`

Ground truth for hidden quality evaluation:

- `magformer_datasets/pseudo_real_512/annotations/instances_target_unlabeled_r114_balanced_minus125.json`

Source/train COCO used for category metadata:

- `magformer_datasets/pseudo_real_512/annotations/instances_target_labeled_r114_balanced_plus125.json`

R115 is an oracle upper-bound control only. It uses full target200 GT and must not be used as the formal teacher for R116 or Stage C.

## Command

```bash
python tools/build_pseudo_bank.py \
  --predictions output/diagnostics/r114_magformer_r113warm_target150_iter2000_remaining75_1024_backmap_topk200_20260518/coco_instances_results.json \
  --source-coco magformer_datasets/pseudo_real_512/annotations/instances_target_labeled_r114_balanced_plus125.json \
  --target-coco magformer_datasets/pseudo_real_512/annotations/instances_target_unlabeled_r114_balanced_minus125.json \
  --output-json output/diagnostics/r116_r114_pseudo_quality_remaining75_20260518/pseudo_bank.json \
  --metrics-json output/diagnostics/r116_r114_pseudo_quality_remaining75_20260518/metrics.json \
  --metrics-md output/diagnostics/r116_r114_pseudo_quality_remaining75_20260518/metrics.md \
  --hidden-gt magformer_datasets/pseudo_real_512/annotations/instances_target_unlabeled_r114_balanced_minus125.json \
  --name r116_r114_pseudo_quality_remaining75 \
  --score-min 0.94 \
  --fill-min 0.55 \
  --area-min 150 \
  --overlap-nms-iou 0.95
```

No new inference was run.

## Output Summary

Output directory:

- `output/diagnostics/r116_r114_pseudo_quality_remaining75_20260518/`

Counts:

| item | value |
| --- | ---: |
| remaining75 images | 75 |
| hidden GT instances | 4364 |
| raw predictions | 7278 |
| kept pseudo instances | 3099 |
| kept per image | 41.320 |

Filters:

| filter | value |
| --- | ---: |
| score_min | 0.94 |
| fill_min | 0.55 |
| area_min | 150 |
| overlap_nms_iou | 0.95 |
| per_image_topk | none |

## Quality Metrics

Global mask quality:

| metric | value |
| --- | ---: |
| P50 | 0.895773 |
| R50 | 0.636114 |
| P75 | 0.601484 |
| R75 | 0.427131 |

Dense buckets from `build_pseudo_bank.py`:

| bucket | images | GT | raw pred | kept | P50 | R50 | P75 | R75 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0-30 | 9 | 225 | 229 | 212 | 0.985849 | 0.928889 | 0.872642 | 0.822222 |
| 31-60 | 49 | 2448 | 4173 | 1976 | 0.918522 | 0.741422 | 0.651316 | 0.525735 |
| 90-100 | 17 | 1691 | 2876 | 911 | 0.825467 | 0.444707 | 0.430296 | 0.231815 |

The dense/high gate maps to the `90-100` bucket because there are no `61-89` or `101+` images in this remaining75 split.

Supplemental read-only stats from the generated pseudo-bank JSON:

| metric | value |
| --- | ---: |
| duplicate same-image/category pair candidates | 65825 |
| duplicate pairs with mask IoU >= 0.95 | 0 |
| duplicate pair rate | 0.000000 |
| empty pseudo images | 0 |
| empty pseudo image ratio | 0.000000 |
| matched TP pred/GT mask area ratio p90 | 1.213916 |

The duplicate rate is computed over kept pseudo masks after overlap NMS. The area ratio uses `matched_tp_pred_gt_mask_area_ratio_50.p90` from `build_pseudo_bank.py`.

## Stage C Gate

R114 val28 anchor: segm AP/AP75 = `0.321831 / 0.295205`.

| gate item | required | observed | result |
| --- | ---: | ---: | --- |
| global P50 | >= 0.94 | 0.895773 | fail |
| global P75 | >= 0.60 | 0.601484 | pass |
| global R50 | >= 0.50 | 0.636114 | pass |
| global R75 | >= 0.33 | 0.427131 | pass |
| dense/high R50 | >= 0.35 | 0.444707 | pass |
| dense/high R75 | >= 0.18 | 0.231815 | pass |
| duplicate pair rate | <= 0.002 | 0.000000 | pass |
| matched TP pred/GT area ratio p90 | <= 1.40 | 1.213916 | pass |
| empty pseudo image ratio | = 0 | 0.000000 | pass |

## Conclusion

Stage C training is not allowed for R116.

The blocker is global P50: `0.895773`, below the required `0.94`. Most other gate items pass, but the pseudo bank does not meet the required precision floor. R115 remains oracle-only and cannot replace R114 as the formal teacher.
