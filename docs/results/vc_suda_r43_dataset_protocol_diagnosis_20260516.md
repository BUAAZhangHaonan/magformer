# VC-SUDA R43 Dataset Protocol Diagnosis - 2026-05-16

## Conclusion

`target_unlabeled200` should not be interpreted as a hand-picked AP-min teacher failure benchmark.

It is best treated as the hidden target-domain evaluation split for the pseudo_real protocol. It is still hard, but the hardness comes from the whole target annotation/scale convention, not from `target_unlabeled200` being uniquely worse than `target_labeled25` or `val28`.

## Evidence Used

New reproducible summary:

- Script: `tools/analyze_vc_suda_dataset_protocol.py`.
- Output: `output/diagnostics/r43_dataset_protocol_20260516/summary.json`.
- Markdown table: `output/diagnostics/r43_dataset_protocol_20260516/summary.md`.

Annotation JSON compared:

- `magformer_datasets/pseudo_real_512/annotations/instances_target_unlabeled.json`
- `magformer_datasets/pseudo_real_512/annotations/instances_target_labeled.json`
- `magformer_datasets/pseudo_real_512/annotations/instances_val.json`
- `magformer_datasets/pseudo_real_512/annotations/instances_source.json`
- `magformer_datasets/pseudo_real_512/annotations/target_unlabeled_train160.json`
- `magformer_datasets/pseudo_real_512/annotations/target_unlabeled_dev40.json`
- `magformer_datasets/pseudo_real_512/annotations/instances_target_labeled_r31_plus25.json`
- `magformer_datasets/pseudo_real_512/annotations/instances_target_unlabeled_r31_minus25.json`
- `magformer_datasets/pseudo_real_512/annotations/instances_target_labeled_r37_balanced_plus25.json`
- `magformer_datasets/pseudo_real_512/annotations/instances_target_unlabeled_r37_balanced_minus25.json`
- `magformer_datasets/20260318_1K_1566/annotations/instances_all.json`
- `magformer_datasets/20260318_1K_1566/annotations/instances_train.json`
- `magformer_datasets/20260318_1K_1566/annotations/instances_val.json`
- `magformer_datasets/20260318_1K_1566/annotations/instances_test.json`
- `magformer_datasets/20260318_1K_32254/annotations/instances_train.json`
- `magformer_datasets/20260318_1K_32254/annotations/instances_val.json`
- `magformer_datasets/20260318_1K_32254/annotations/instances_test.json`

Prediction JSON compared by scale only, with no matching:

- `output/diagnostics/r33_r12_ckpt499_sanity_unlabeled200_20260516/coco_instances_results.json`
- `output/diagnostics/r41_eval1536_teacher_r12_20260516/coco_instances_results.json`
- `output/experiments/vc_suda_stage_c_r42_point65536_iter0750_target_unlabeled200_1024_backmap_topk200_20260516/coco_instances_results.json`
- `output/experiments/teacher_pseudoreal_target_unlabeled200_1024_backmap_recheck_20260515/coco_instances_results.json`
- `output/experiments/teacher_pseudoreal_val_1024_backmap_corrected_config_20260515/coco_instances_results.json`
- `output/experiments/teacher8499_target_labeled25_1024_backmap_20260515/coco_instances_results.json`

Existing metrics/docs read:

- R12/R33: `output/diagnostics/r33_r12_ckpt499_sanity_unlabeled200_20260516/metrics.cocoeval.json`
- R38: `output/diagnostics/r38_oracle_score_upper_bound_20260516/combined_oracle_bound_summary.json`
- R40: `output/diagnostics/r40_oracle_miss_atlas_20260516/r40_oracle_miss_atlas_summary.json`
- R41: `output/diagnostics/r41_eval1536_teacher_r12_20260516/metrics.cocoeval.json`
- R42: `output/experiments/vc_suda_stage_c_r42_point65536_iter0750_target_unlabeled200_1024_backmap_topk200_20260516/metrics.cocoeval.json`
- R30: `docs/results/vc_suda_r30_pseudo_real_validity_audit_20260516.md`
- R36/R37: `docs/results/vc_suda_r36_promoted25_audit_20260516.md`, `docs/results/vc_suda_stage_c_r37_balanced_plus25_plan_20260516.md`
- Teacher first50: `output/experiments/teacher_first50_1024_backmap_recheck_20260515/metrics.cocoeval.json`

## Distribution Readout

| split | images | instances | inst/img p50/p90 | mask area p50 | area <= 256 | density 25-30 | density 46-60 | density >90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| target_unlabeled200 | 200 | 11,750 | 50 / 100 | 452 | 0.2497 | 22 | 131 | 47 |
| target_labeled25 | 25 | 1,697 | 50 / 100 | 401 | 0.3011 | 0 | 16 | 9 |
| val28 | 28 | 1,892 | 50 / 100 | 399 | 0.2970 | 0 | 18 | 10 |
| target_train160 | 160 | 9,383 | 50 / 100 | 457 | 0.2490 | 15 | 109 | 36 |
| target_dev40 | 40 | 2,367 | 50 / 100 | 439 | 0.2526 | 7 | 22 | 11 |
| pseudo_real source | 1,008 | 61,652 | 50 / 100 | 438 | 0.2664 | 101 | 625 | 281 |
| ordinary 1.5K all | 1,566 | 95,895 | 50 / 100 | 1,846 | 0.0547 | 147 | 987 | 432 |
| ordinary 32K train | 25,654 | 1,398,531 | 50 / 99 | 7,572 | 0.0119 | 4,145 | 15,258 | 5,192 |

The target splits have the same broad density shape: p50 is `50`, p90 is `100`, and all have many dense images. `target_unlabeled200` is not denser than `target_labeled25` or `val28`; its `>90` image share is `23.5%`, while `target_labeled25` is `36.0%` and `val28` is `35.7%`.

The real separation is target versus ordinary source-scale data. `target_unlabeled200` mask area p50 is `452`, while ordinary 1.5K all is `1,846` and ordinary 32K train is `7,572`. Small masks are also much more common on target: `area <= 256` is `24.97%` on `target_unlabeled200`, versus `5.47%` on ordinary 1.5K all and `1.19%` on ordinary 32K train.

## Prediction Scale Readout

| prediction | annotation | predictions | pred/img p50 | pred/img p50 / GT p50 | pred mask p50 / GT p50 |
|---|---|---:|---:|---:|---:|
| R33/R12 ckpt499 | target_unlabeled200 | 13,806 | 62.5 | 1.25 | 1.064 |
| R41 eval1536 | target_unlabeled200 | 17,495 | 79.5 | 1.59 | 0.982 |
| R42 point65536 | target_unlabeled200 | 13,851 | 62.0 | 1.24 | 1.069 |
| Teacher8499 direct | target_unlabeled200 | 10,954 | 56.0 | 1.12 | 8.338 |
| Teacher8499 direct | val28 | 1,509 | 55.0 | 1.10 | 8.787 |
| Teacher8499 direct | target_labeled25 | 1,397 | 58.0 | 1.16 | 8.835 |

This supports the protocol split:

- Teacher8499 is strong on original 1.5K first50: bbox AP `0.6519639237`, segm AP `0.6253031403`.
- The same Teacher8499 direct path is near zero on pseudo_real target splits: target_unlabeled200 segm AP `0.0000258774`, val28 segm AP `0.0000037152`, target_labeled25 segm AP `0.0`.
- The direct Teacher8499 target failure is a target-scale/convention mismatch across target splits, not a unique property of `target_unlabeled200`.
- R33/R12 adapted predictions are scale-aligned on `target_unlabeled200` by aggregate mask area, with pred mask p50 about `1.064x` GT p50.

## Existing AP Readout

| run | target | bbox AP/AP50/AP75 | segm AP/AP50/AP75 |
|---|---|---:|---:|
| Teacher8499 first50 | original 1.5K first50 | 0.651964 / 0.853150 / 0.717544 | 0.625303 / 0.867953 / 0.724342 |
| Teacher8499 direct | pseudo_real target_unlabeled200 | 0.0000417 / 0.000396 / 0.0 | 0.0000259 / 0.000241 / 0.0 |
| Teacher8499 direct | pseudo_real val28 | 0.00000162 / 0.0000162 / 0.0 | 0.00000372 / 0.0000186 / 0.0 |
| R33/R12 ckpt499 | pseudo_real target_unlabeled200 | 0.394476 / 0.741115 / 0.379187 | 0.320048 / 0.648363 / 0.284144 |
| R41 eval1536 | pseudo_real target_unlabeled200 | 0.356160 / 0.717496 / 0.315430 | 0.284495 / 0.609380 / 0.236851 |
| R42 point65536 | pseudo_real target_unlabeled200 | 0.394389 / 0.739593 / 0.379503 | 0.318900 / 0.648652 / 0.282661 |
| R37 balanced +25 | pseudo_real target_unlabeled200 | 0.382137 / 0.724578 / 0.359913 | 0.301567 / 0.632622 / 0.246629 |

R38/R40 explain why the R33/R12 line is hard to raise by score or eval-scale changes alone. On the fixed R15/R33 prediction pool, oracle-score segm AP only reaches about `0.361386`, global oracle R@75 is `0.347830`, dense `>90` R@75 is `0.172577`, and small `<=256` R@75 is `0.009884`.

## Interpretation

The AP target must be read as a target-domain pseudo_real benchmark target, not as the original 1.5K source-domain teacher target.

The `0.625303` Teacher8499 first50 result is a source-domain sanity line. It proves the checkpoint and 1024 backmap path can work on original 1.5K data. It does not set the current target_unlabeled200 goal because pseudo_real target objects are much smaller and use a different annotation/scale convention.

The current `0.320048` R33/R12 line is the meaningful adapted target-domain baseline. R41 and R42 failed because they did not improve the mask set or dense/small-object coverage. R31/R37 also failed, and R36 shows why: the promoted examples were hard and dense, while balanced promotion still did not fix the training path.

## What To Pause

Pause these routes until the protocol question is closed:

- More training-parameter sweeps from R41/R42.
- More inference-scale changes.
- More `+25` or label-expansion training from R31/R34/R37.
- More attempts to interpret Teacher8499 direct pseudo_real AP as a model-quality failure.

## Minimum Next Work

The smallest interpretable next step is no-training protocol accounting:

- Treat `target_unlabeled200` as the hidden target-domain eval split.
- Keep original 1.5K first50 and pseudo_real target AP in separate tables.
- Add split provenance to future docs: dataset root, annotation file, split, checkpoint, image size, and whether the checkpoint is direct source Teacher8499 or adapted R12/R33.
- If a future eval is allowed, compare the same adapted checkpoint on `target_labeled25`, `val28`, and `target_unlabeled200` under one protocol. That is the minimal way to measure target_unlabeled difficulty directly without changing training.

## Reproduction

```bash
/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python tools/analyze_vc_suda_dataset_protocol.py \
  --ann target_unlabeled200=magformer_datasets/pseudo_real_512/annotations/instances_target_unlabeled.json \
  --ann target_labeled25=magformer_datasets/pseudo_real_512/annotations/instances_target_labeled.json \
  --ann val28=magformer_datasets/pseudo_real_512/annotations/instances_val.json \
  --ann source25654=magformer_datasets/pseudo_real_512/annotations/instances_source.json \
  --ann target_train160=magformer_datasets/pseudo_real_512/annotations/target_unlabeled_train160.json \
  --ann target_dev40=magformer_datasets/pseudo_real_512/annotations/target_unlabeled_dev40.json \
  --ann r31_target_labeled50=magformer_datasets/pseudo_real_512/annotations/instances_target_labeled_r31_plus25.json \
  --ann r31_target_unlabeled175=magformer_datasets/pseudo_real_512/annotations/instances_target_unlabeled_r31_minus25.json \
  --ann r37_target_labeled50=magformer_datasets/pseudo_real_512/annotations/instances_target_labeled_r37_balanced_plus25.json \
  --ann r37_target_unlabeled175=magformer_datasets/pseudo_real_512/annotations/instances_target_unlabeled_r37_balanced_minus25.json \
  --ann ordinary_1566_all=magformer_datasets/20260318_1K_1566/annotations/instances_all.json \
  --ann ordinary_1566_train=magformer_datasets/20260318_1K_1566/annotations/instances_train.json \
  --ann ordinary_1566_val=magformer_datasets/20260318_1K_1566/annotations/instances_val.json \
  --ann ordinary_1566_test=magformer_datasets/20260318_1K_1566/annotations/instances_test.json \
  --ann ordinary_32254_train=magformer_datasets/20260318_1K_32254/annotations/instances_train.json \
  --ann ordinary_32254_val=magformer_datasets/20260318_1K_32254/annotations/instances_val.json \
  --ann ordinary_32254_test=magformer_datasets/20260318_1K_32254/annotations/instances_test.json \
  --pred r33_r12_ckpt499_tu=target_unlabeled200=output/diagnostics/r33_r12_ckpt499_sanity_unlabeled200_20260516/coco_instances_results.json \
  --pred r41_eval1536_tu=target_unlabeled200=output/diagnostics/r41_eval1536_teacher_r12_20260516/coco_instances_results.json \
  --pred r42_point65536_tu=target_unlabeled200=output/experiments/vc_suda_stage_c_r42_point65536_iter0750_target_unlabeled200_1024_backmap_topk200_20260516/coco_instances_results.json \
  --pred teacher8499_tu=target_unlabeled200=output/experiments/teacher_pseudoreal_target_unlabeled200_1024_backmap_recheck_20260515/coco_instances_results.json \
  --pred teacher8499_val=val28=output/experiments/teacher_pseudoreal_val_1024_backmap_corrected_config_20260515/coco_instances_results.json \
  --pred teacher8499_labeled=target_labeled25=output/experiments/teacher8499_target_labeled25_1024_backmap_20260515/coco_instances_results.json \
  --output-json output/diagnostics/r43_dataset_protocol_20260516/summary.json \
  --output-md output/diagnostics/r43_dataset_protocol_20260516/summary.md
```
