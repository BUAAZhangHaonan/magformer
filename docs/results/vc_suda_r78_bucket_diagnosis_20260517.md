# VC-SUDA R78 no-train bucket diagnosis, 2026-05-17

R78 confirms that the current R74 target bottleneck is concentrated in tiny and dense target buckets.

The clean signal is the split between normal and hard subsets. R74 reaches normal segm AP/AP75 `0.412249 / 0.412486`, but dense is only `0.173698 / 0.117155`, dense_tiny is `0.215194 / 0.156112`, tiny `area<=256` is `0.010666 / 0.000385`, and the bottom 20% GT-area bucket is `0.003499 / 0.000098`. This is not a broad normal-image failure.

## Scope

This was a no-train diagnosis only.

- Host: `4029`.
- Project: `/home/hdd3/zhanghaonan/magformer`.
- Conda env: `/home/hdd3/zhanghaonan/anaconda3/envs/magformer`.
- R74 predictions: `output/diagnostics/r74_balanced_ce_minfg005_target_unlabeled200_20260517/coco_instances_results.json`.
- R74 metrics: `output/diagnostics/r74_balanced_ce_minfg005_target_unlabeled200_20260517/metrics.cocoeval.json`.
- R74 oracle: `output/diagnostics/r74_balanced_ce_minfg005_target_oracle_20260517/r74_oracle_bound_summary.json`.
- Target GT: `magformer_datasets/pseudo_real_512/annotations/instances_target_unlabeled.json`.
- R52 bucket rules: `output/diagnostics/r52_target_unlabeled_sampling_20260516/target_sampling_stats.json`.

No training was started. No 75+ AP research checkpoint was used.

## Bucket Definitions

- `normal`, `dense`, and `dense_tiny` use the existing R52 prediction-only stats.
- `tiny_area_le_256` uses GT mask area `<=256`.
- `bottom20_area` uses the lowest 20% of GT mask areas. This selected `2,350 / 11,750` annotations, with area threshold `207.0`.
- Annotation-area buckets keep non-target GT as ignored COCO GT, so detections on excluded objects do not become normal false positives.

## Main Table

| bucket | images | annotations | bbox AP/AP50/AP75 | segm AP/AP50/AP75 | mask oracle R@50/R@75 |
|---|---:|---:|---:|---:|---:|
| overall | 200 | 11750 | 0.387567 / 0.728979 / 0.373927 | 0.336010 / 0.656434 / 0.315963 | 0.691064 / 0.373191 |
| normal | 158 | 7634 | 0.461006 / 0.809127 / 0.473095 | 0.412249 / 0.756025 / 0.412486 | 0.784779 / 0.471706 |
| dense | 30 | 2971 | 0.234275 / 0.553791 / 0.170954 | 0.173698 / 0.436749 / 0.117155 | 0.502861 / 0.180074 |
| dense_tiny | 12 | 1145 | 0.265594 / 0.592845 / 0.205461 | 0.215194 / 0.499850 / 0.156112 | 0.554585 / 0.217467 |
| tiny_area_le_256 | 185 | 2934 | 0.080466 / 0.330162 / 0.013383 | 0.010666 / 0.056596 / 0.000385 | 0.183367 / 0.016701 |
| bottom20_area | 181 | 2350 | 0.062438 / 0.275015 / 0.008055 | 0.003499 / 0.020245 / 0.000098 | 0.124681 / 0.007660 |

## Existing Dense Audit Cross-Check

The existing R74 dense geometry audit gives the same direction.

| GT density bucket | images | gt | pred | mask TP | mask low-IoU FP | mask FN | bbox TP | bbox low-IoU FP | bbox FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0-30 | 22 | 550 | 750 | 531 | 198 | 19 | 538 | 186 | 12 |
| 31-60 | 131 | 6547 | 8321 | 5204 | 3046 | 1343 | 5541 | 2593 | 1006 |
| 90-100 | 47 | 4653 | 5110 | 2385 | 2675 | 2268 | 2861 | 2010 | 1792 |

Dense `90-100` has many missed masks and low-IoU predictions even though the prediction count is not low. This points to localization and separation quality, not just score thresholding.

## Decision

Do not jump straight to R74+32K source as the next main move.

R74 already performs well on normal target images, and the hard buckets are failing through mask geometry and oracle coverage. More source images may improve general coverage, but this table says the immediate blocker is pseudo-label weight/objective behavior on tiny and dense targets. The next step should be a focused pseudo-label weight or target objective diagnosis before spending a long run on R74+32K source.

## Commands

Diagnostic script validation:

```bash
/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python -m py_compile tools/diagnose_r78_bucket_ap.py
/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python tools/diagnose_r78_bucket_ap.py --help
```

Main diagnosis:

```bash
/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python tools/diagnose_r78_bucket_ap.py \
  --ann magformer_datasets/pseudo_real_512/annotations/instances_target_unlabeled.json \
  --pred output/diagnostics/r74_balanced_ce_minfg005_target_unlabeled200_20260517/coco_instances_results.json \
  --r52-stats output/diagnostics/r52_target_unlabeled_sampling_20260516/target_sampling_stats.json \
  --expected-metrics output/diagnostics/r74_balanced_ce_minfg005_target_unlabeled200_20260517/metrics.cocoeval.json \
  --out-json output/diagnostics/r78_bucket_diagnosis_20260517/r78_bucket_ap_summary.json \
  --out-md output/diagnostics/r78_bucket_diagnosis_20260517/r78_bucket_ap_table.md
```

Outputs:

- `output/diagnostics/r78_bucket_diagnosis_20260517/r78_bucket_ap_summary.json`
- `output/diagnostics/r78_bucket_diagnosis_20260517/r78_bucket_ap_table.md`

## Validation

- `py_compile` passed for `tools/diagnose_r78_bucket_ap.py`.
- `--help` passed.
- Main diagnosis completed all 6 buckets and matched the existing overall R74 COCO metrics within `1e-9`.
- Finite check passed for all bucket AP and oracle values.
- No training command was run.
