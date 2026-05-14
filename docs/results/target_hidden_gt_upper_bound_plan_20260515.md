# Target Hidden-GT Upper-Bound Plan - 2026-05-15

## Purpose

This run checks whether true target-domain GT supervision can open the upper bound. It trains ordinary supervised MAGFormer on hidden GT from `instances_target_unlabeled.json`.

## Split

Source annotation: `magformer_datasets/pseudo_real_512/annotations/instances_target_unlabeled.json`.

Method: sort images deterministically by `(image_id, file_name)`, then use the first 160 images for train and the final 40 images for dev.

Generated files:

- `annotations/target_unlabeled_train160.json`: 160 images, 9383 annotations.
- `annotations/target_unlabeled_dev40.json`: 40 images, 2367 annotations.

The two splits have no shared `image_id` or `file_name`. Annotation IDs, image IDs, category IDs, and file names are preserved from the source COCO file. Both splits use category `{id: 1, name: component}`. No image is empty.

The dev40 images still live under the target training image/depth folders, so the config uses `train_split: train` and `val_split: train`. The existing 28-image val set is not used for model selection.

## Config

Config: `configs/upper_bound_target160_supervised_1024_teacher8499.yaml`.

Key settings:

- Ordinary supervised path: `vc_suda.enabled=false`.
- Dataset root: `magformer_datasets/pseudo_real_512`.
- Train ann: `annotations/target_unlabeled_train160.json`.
- Dev ann: `annotations/target_unlabeled_dev40.json`.
- Warm start: `output/experiments/vc_suda_stage_b_1024_teacher8499_20260514_005821/checkpoint_iter_0008999.pth`.
- Max iter: 2000.
- Eval/checkpoint period: 500.
- Eval IoU types: `bbox`, `segm`.
- Eval batch size: 1.
- GPUs: 4,5,6,7.
- Output: `output/upper_bound/target160_supervised_1024_teacher8499`.

This config does not mix source data and does not use pseudo labels. The disabled `vc_suda` block is present only to satisfy the config schema.

## Gate

Use dev40 segmentation AP as the gate:

- `segm AP >= 50`: pseudo labels are the main bottleneck.
- `35 <= segm AP < 50`: labels help, but target data or optimization is still not enough.
- `segm AP <= 35`: the current model, data, or eval setup has an upper-bound problem.

After choosing no model from dev40, run the existing 28-image val set only as final sanity.

## Smoke Rule

Only run a 2-3 iteration smoke with a temporary output directory before launch. Do not start the formal long run from this plan step.

## Result

Formal run:

- Config: `configs/upper_bound_target160_supervised_1024_teacher8499.yaml`.
- Split: train160 / dev40.
- `vc_suda.enabled=false`.
- StageB warm start.
- `max_iter=2000`.
- `eval_period=500`.
- Output: `output/upper_bound/target160_supervised_1024_teacher8499`.

The run ended naturally. GPUs were released. No training error was observed.

Checkpoints:

- `checkpoint_iter_0001499.pth`.
- `checkpoint_iter_0001999.pth`.
- `checkpoint_iter_0002000.pth`.

Dev40 metrics:

| Iter | bbox AP | bbox AP50 | bbox AP75 | segm AP | segm AP50 | segm AP75 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 500 | - | - | - | 0.0776 | - | - |
| 1000 | 0.1662 | 0.4668 | 0.1015 | 0.1203 | 0.3657 | 0.0630 |
| 1500 | 0.1573 | 0.4501 | 0.0874 | 0.1128 | 0.3551 | 0.0496 |
| 2000 | 0.1614 | 0.4684 | 0.0869 | 0.1149 | 0.3650 | 0.0515 |

## Decision

Do not extend this run to 4000 iterations.

This target hidden-GT upper-bound run did not open the expected upper bound. It is also clearly below StageB/R3 on `target_unlabeled` full evaluation, so continuing the same training recipe is not useful.

Before any further training, diagnose:

- Config parity with StageB/R3.
- Eval setup and metric scale.
- `train160` / `dev40` split construction and data loading.
