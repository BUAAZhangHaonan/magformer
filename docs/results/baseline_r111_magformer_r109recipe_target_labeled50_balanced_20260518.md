# R111 MagFormer R109 Recipe Target50 Balanced

Date: 2026-05-18
Branch: feature/vc-suda-sim2real
Config: `configs/baseline_supervised_r111_magformer_r109recipe_target_labeled50_balanced_1000.yaml`

## Purpose

R111 migrates the R109 oracle-selected recipe package onto the formal non-leakage R104 target-labeled50 balanced training set.

This run can only be interpreted as the R109 recipe package transfer. It must not attribute effects to any single parameter.

## Recipe Package

- Warm-start from `output/baseline/r100_magformer_r98ckpt_target_labeled25_cleanfit_1024/checkpoint_iter_0000500.pth` through `model.finetune_weights`.
- `runtime.resume: null`.
- `dice_weight: 10.0`.
- `mask_weight: 5.0`.
- `importance_sample_ratio: 0.0`.
- `train_num_points: 12544`.
- `oversample_ratio: 3.0`.
- No RGB augmentation, no depth noise, image size `1024`.
- `runtime.depth_sanity.min_mask_fg_ratio: 0.0009` for small-instance preflight support.

## Formal Splits

Primary non-leakage metrics:

- `val28`: `annotations/instances_val.json`, split `val`.
- `non-overlap175`: `annotations/instances_target_unlabeled_r37_balanced_minus25.json`, target_unlabeled200 minus the 25 promoted training images.

Reference-only metric:

- `full200`: `annotations/instances_target_unlabeled.json`. This includes the 25 promoted training images and is leakage-prone for R111, so it is reference only.

## Baselines

R104 formal baseline:

| eval | segm AP | note |
| --- | ---: | --- |
| val28 | `0.283886` | primary |
| non-overlap175 | `0.329132` | primary |
| full200 reference | `0.340785` | reference only |

Historical R80 reference:

- full200 segm AP: `0.336383`.

## Success Criteria

- Main success: R111 best checkpoint beats R104 non-overlap175 segm AP `0.329132` under the same external 1024 backmap protocol.
- Secondary support: R111 val28 beats R104 val28 segm AP `0.283886`.
- Reference-only: full200 may be compared with R104 `0.340785` and R80 `0.336383`, but it cannot drive the formal conclusion.
- Train-fit check: record whether train50 reaches `0.61+` segm AP.

## Static Validation

Static validation passed before training.

- `validate_config(strict=True)`: valid. The only message was the existing DPE informational note.
- Checked config fields: `runtime.resume=null`, warm-start through `model.finetune_weights`, train50 annotation, val28 annotation/split, `importance_sample_ratio=0.0`, `dice_weight=10.0`, `mask_weight=5.0`, `train_num_points=12544`, `oversample_ratio=3.0`, and `runtime.depth_sanity.min_mask_fg_ratio=0.0009`.
- Annotation counts: train50 `50` images / `3170` anns, val28 `28` images / `1892` anns, non-overlap175 `175` images / `10277` anns, full200 `200` images / `11750` anns.
- Non-overlap175 file exists at `annotations/instances_target_unlabeled_r37_balanced_minus25.json`.
- File-name overlap: train50 vs val28 `0`; train50 vs non-overlap175 `0`; full200 minus non-overlap175 `25` images.
- Loader smoke checked: dataset lengths `50 / 28`; first train batch images `[1, 3, 1024, 1024]`, depths `[1, 1, 1024, 1024]`; first val batch images `[4, 3, 512, 512]`, depths `[4, 1, 512, 512]`.

## Training

Pending.

## External Eval

Pending.

## Judgment

Pending.

## Commits

- Config/docs placeholder: pending.
- Final docs-only update: pending.
