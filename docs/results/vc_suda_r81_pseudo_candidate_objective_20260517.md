# VC-SUDA R81 no-train pseudo candidate/objective diagnosis, 2026-05-17

R81 says the next step should be a multi-view/TTA pseudo bank.

R80 did not fail mainly because the pseudo loss was still too weak. The R80 teacher/scorer candidate pool is almost the same as R74, and tiny/bottom objects are already missing before thresholding. Dense images also expose many high-score unmatched decoder queries, but an unmatched-negative or exterior constraint cannot create the missing tiny positives.

## Scope

- Host: `4029`.
- Project: `/home/hdd3/zhanghaonan/magformer`.
- Conda env: `magformer`.
- GPU: only GPU4 via `CUDA_VISIBLE_DEVICES=4`.
- R80 config: `configs/vc_suda_stage_c_r80_unsup_schedule_1024.yaml`.
- R80 checkpoint: `output/vc_suda/stage_c_r80_unsup_schedule_1024/checkpoint_iter_0000750.pth`.
- Optional comparison: R74 `output/vc_suda/stage_c_r74_balanced_ce_minfg005_1024/checkpoint_iter_0000750.pth`.
- R78/R52 bucket stats: `output/diagnostics/r52_target_unlabeled_sampling_20260516/target_sampling_stats.json`.
- Target GT proxy: `magformer_datasets/pseudo_real_512/annotations/instances_target_unlabeled.json`.

No training was started. No training logic was changed.

## Method

I extended `tools/diagnose_vc_suda_pseudo_labels.py` only.

- `--unique-target-images` iterates the 200 `target_unlabeled` images directly in dataset order, instead of using the source-cycled training loader.
- Candidate and kept pseudo labels are still produced by the training `PseudoLabelScorer` and `filter_by_threshold`.
- Candidate/kept coverage is a bbox-IoU proxy against target GT. Training still receives no target labels.
- Kept mask area is decoder-grid hard area from pseudo mask probability `>0.5`.
- Dry-run objective audit counts raw decoder queries with foreground softmax score `>=0.3` and max hard-mask IoU to kept pseudo masks `<0.3`.
- For `tiny_area_le_256` and `bottom20_area`, candidate/kept/high-score-unmatched counts are restricted to predictions with bbox IoU `>=0.50` to a GT object in that area bucket.

Artifacts:

- R80 summary: `output/diagnostics/r81_pseudo_candidate_objective_20260517/r80_full200_summary.json`.
- R80 buckets: `output/diagnostics/r81_pseudo_candidate_objective_20260517/r80_full200_buckets.json`.
- R80 log: `output/diagnostics/r81_pseudo_candidate_objective_20260517/r80_full200.log`.
- R74 summary: `output/diagnostics/r81_pseudo_candidate_objective_20260517/r74_full200_summary.json`.
- R74 buckets: `output/diagnostics/r81_pseudo_candidate_objective_20260517/r74_full200_buckets.json`.
- R74 log: `output/diagnostics/r81_pseudo_candidate_objective_20260517/r74_full200.log`.

## R80 Full-200 Pseudo Candidate Table

| bucket | images | GT/proxy | candidates | kept | keep-rate | candidate cov@50/@75 | kept cov@50/@75 | quality mean/p10/p50/p90 | kept area mean/p10/p50/p90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 200 | 11750 | 20000 | 8502 | `0.425` | `0.555 / 0.249` | `0.367 / 0.169` | `0.093 / 0.000 / 0.069 / 0.237` | `113 / 44 / 118 / 174` |
| normal | 158 | 7634 | 15800 | 5202 | `0.329` | `0.613 / 0.306` | `0.358 / 0.191` | `0.074 / 0.000 / 0.031 / 0.226` | `120 / 48 / 128 / 177` |
| dense | 30 | 2971 | 3000 | 2284 | `0.761` | `0.440 / 0.139` | `0.367 / 0.118` | `0.158 / 0.045 / 0.171 / 0.243` | `109 / 44 / 110 / 171` |
| dense_tiny | 12 | 1145 | 1200 | 1016 | `0.847` | `0.466 / 0.156` | `0.428 / 0.154` | `0.177 / 0.081 / 0.192 / 0.247` | `89 / 35 / 83 / 156` |
| tiny_area_le_256 | 185 | 2934 | 1158 | 621 | `0.536` | `0.326 / 0.050` | `0.211 / 0.031` | `0.114 / 0.000 / 0.110 / 0.233` | `95 / 44 / 82 / 165` |
| bottom20_area | 181 | 2355 | 811 | 437 | `0.539` | `0.286 / 0.039` | `0.186 / 0.025` | `0.115 / 0.000 / 0.111 / 0.235` | `96 / 42 / 82 / 168` |

## Dry-Run Objective Audit

This table counts raw decoder queries that positive-only pseudo supervision would not bind to any kept pseudo mask under the stated definition.

| bucket | high-score unmatched mean/image | query count | score p50/p90 | mask area p50/p90 |
|---|---:|---:|---:|---:|
| overall | `29.875` | 5975 | `0.992 / 1.000` | `70 / 169` |
| normal | `22.873` | 3614 | `0.999 / 1.000` | `123 / 177` |
| dense | `52.833` | 1585 | `0.683 / 0.999` | `0 / 136` |
| dense_tiny | `64.667` | 776 | `0.597 / 0.994` | `0 / 58.5` |
| tiny_area_le_256 | `1.632` | 302 | `0.999 / 1.000` | `97 / 180` |
| bottom20_area | `1.160` | 210 | `0.999 / 1.000` | `96 / 179` |

## R74 vs R80 Candidate/Kept Coverage

| bucket | R74 candidate cov@50/@75 | R80 candidate cov@50/@75 | R74 kept cov@50/@75 | R80 kept cov@50/@75 | R74 keep-rate | R80 keep-rate |
|---|---:|---:|---:|---:|---:|---:|
| overall | `0.552 / 0.250` | `0.555 / 0.249` | `0.365 / 0.167` | `0.367 / 0.169` | `0.421` | `0.425` |
| normal | `0.607 / 0.307` | `0.613 / 0.306` | `0.355 / 0.189` | `0.358 / 0.191` | `0.327` | `0.329` |
| dense | `0.442 / 0.145` | `0.440 / 0.139` | `0.365 / 0.120` | `0.367 / 0.118` | `0.751` | `0.761` |
| dense_tiny | `0.468 / 0.147` | `0.466 / 0.156` | `0.433 / 0.141` | `0.428 / 0.154` | `0.837` | `0.847` |
| tiny_area_le_256 | `0.324 / 0.050` | `0.326 / 0.050` | `0.208 / 0.030` | `0.211 / 0.031` | `0.527` | `0.536` |
| bottom20_area | `0.287 / 0.040` | `0.286 / 0.039` | `0.183 / 0.025` | `0.186 / 0.025` | `0.530` | `0.539` |

## Conclusion

The next step is `multi-view/TTA pseudo bank`.

Reasons:

1. Tiny and bottom20 candidate coverage is already too low before thresholding. R80 candidate cov@75 is only `0.050` for `area<=256` and `0.039` for bottom20.
2. R80 barely changes the candidate pool from R74. The stronger schedule lifts keep-rate only from `0.421` to `0.425`, and hard-bucket coverage is effectively unchanged.
3. The positive-only objective does leave many high-score unmatched dense queries, but fixing unmatched negatives first would mostly regularize predictions around an incomplete pseudo target set. It would not supply the missing tiny positives.

Area-aware thresholds may recover some cov@50 from existing candidates, and unmatched negative/exterior constraints remain useful later. They should not be the next main run. The immediate blocker is candidate generation quality for tiny/bottom objects.

## Validation

- `py_compile` passed for `tools/diagnose_vc_suda_pseudo_labels.py`.
- `--help` exposes `--unique-target-images`, `--objective-score-threshold`, and `--objective-unmatched-iou`.
- 2-image GPU4 smoke passed after the bool-mask IoU fix.
- R80 full-200 GPU4 diagnosis exited `0`.
- R74 full-200 GPU4 comparison exited `0`.
- R80/R74 logs contain no `Traceback`, no `CUDA out of memory`, no `RuntimeError`, and no `non-finite`.
- R80/R74 summary and bucket JSON files contain only finite numeric values.
- No training command was run.
