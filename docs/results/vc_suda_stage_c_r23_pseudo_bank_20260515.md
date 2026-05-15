# VC-SUDA Stage C R23 Pseudo Bank Tool - 2026-05-15

Conclusion: the offline 1024 pseudo bank tool is ready for reproducible quality checks, but neither bank should be connected to training yet. The high-precision filters raise precision strongly, but dense-image recall remains too low.

## Scope

- No training was started.
- No branch or worktree was created.
- Output artifacts were written under `output/diagnostics/r23_pseudo_bank_20260515/` and must stay uncommitted.
- The tool fails loudly for missing files, malformed COCO input, missing prediction fields, unknown target image ids, and unknown category ids.

## Tool

Added `tools/build_pseudo_bank.py`.

Inputs:

- COCO result prediction JSON.
- Source COCO annotation JSON for category alignment.
- Target COCO annotation JSON for pseudo-bank images.
- Optional hidden-GT COCO annotation JSON for quality metrics.

Filters:

- score threshold.
- mask fill ratio.
- mask area.
- mask-derived bbox side min/max.
- per-image top-k after filtering.
- dynamic thresholds by raw `pred_count` bucket.
- optional same-image/category mask overlap NMS, default off.

Output annotations use mask-derived `segmentation`, `bbox`, and `area` so the pseudo bank is internally consistent.

## Commands

Simple high precision:

```bash
/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python tools/build_pseudo_bank.py \
  --predictions output/diagnostics/r15_topk200_maxdets200_20260515/coco_instances_results.json \
  --source-coco magformer_datasets/20260318_1K_32254/annotations/instances_train.json \
  --target-coco magformer_datasets/pseudo_real_512/annotations/instances_target_unlabeled.json \
  --hidden-gt magformer_datasets/pseudo_real_512/annotations/instances_target_unlabeled.json \
  --output-json output/diagnostics/r23_pseudo_bank_20260515/simple_high_precision_pseudo_bank.json \
  --metrics-json output/diagnostics/r23_pseudo_bank_20260515/simple_high_precision_metrics.json \
  --metrics-md output/diagnostics/r23_pseudo_bank_20260515/simple_high_precision_metrics.md \
  --name simple_high_precision \
  --score-min 0.94 --fill-min 0.55 --area-min 300
```

Dynamic density high precision:

```bash
/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python tools/build_pseudo_bank.py \
  --predictions output/diagnostics/r15_topk200_maxdets200_20260515/coco_instances_results.json \
  --source-coco magformer_datasets/20260318_1K_32254/annotations/instances_train.json \
  --target-coco magformer_datasets/pseudo_real_512/annotations/instances_target_unlabeled.json \
  --hidden-gt magformer_datasets/pseudo_real_512/annotations/instances_target_unlabeled.json \
  --output-json output/diagnostics/r23_pseudo_bank_20260515/dynamic_density_high_precision_pseudo_bank.json \
  --metrics-json output/diagnostics/r23_pseudo_bank_20260515/dynamic_density_high_precision_metrics.json \
  --metrics-md output/diagnostics/r23_pseudo_bank_20260515/dynamic_density_high_precision_metrics.md \
  --name dynamic_density_high_precision \
  --score-min 0.94 --fill-min 0.55 --area-min 300 \
  --dynamic-bucket 61-89:score=0.94325,fill=0.55,area=300 \
  --dynamic-bucket 90-inf:score=0.94775,fill=0.55,area=300
```

## Results

| bank | kept | kept/img | P50 | R50 | P75 | R75 |
|---|---:|---:|---:|---:|---:|---:|
| simple_high_precision | 6,052 | 30.260 | 0.937707 | 0.482979 | 0.606742 | 0.312511 |
| dynamic_density_high_precision | 5,461 | 27.305 | 0.951474 | 0.442213 | 0.648416 | 0.301362 |

Dense `90-100` GT bucket:

| bank | images | gt | raw_pred | kept | P50 | R50 | P75 | R75 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| simple_high_precision | 47 | 4,653 | 4,981 | 1,464 | 0.887978 | 0.279390 | 0.443989 | 0.139695 |
| dynamic_density_high_precision | 47 | 4,653 | 4,981 | 1,032 | 0.925388 | 0.205244 | 0.532946 | 0.118203 |

Mask area ratio for matched TP at IoU 0.50:

| bank | count | mean | p50 | p75 | p90 |
|---|---:|---:|---:|---:|---:|
| simple_high_precision | 5,675 | 1.142504 | 1.115942 | 1.221125 | 1.357331 |
| dynamic_density_high_precision | 5,196 | 1.138396 | 1.112328 | 1.213269 | 1.345101 |

## Read

The simple bank reproduces the R22 high-precision finding: precision is high, but recall drops sharply. The dynamic bank improves precision further and lifts P75, but it also removes dense-image positives and lowers dense recall.

The dense bucket remains the blocker. On 47 dense images with 4,653 GT objects, the simple bank keeps only 1,464 pseudo instances and recalls 27.9% at IoU 0.50. The dynamic bank keeps 1,032 and recalls only 20.5% at IoU 0.50.

## Recommendation

Do not enter training with these banks yet. Use the tool first to compare any new pseudo-bank proposal against these two baselines, with dense `90-100` recall as a hard readout. A training-facing bank needs better dense recall without giving up the high P50/P75 precision gained here.

## Validation

Focused tests:

```bash
/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python -m pytest tests/test_build_pseudo_bank.py -q
```

The new tests cover filtering, mask-derived bbox/area/RLE consistency, hidden-GT P/R metrics, and dynamic bucket threshold overrides. The test was verified red first by importing the missing tool, then green after implementation.
