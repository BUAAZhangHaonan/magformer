# R112 MagFormer R111 Recipe Target100 Balanced

Date: 2026-05-18
Branch: feature/vc-suda-sim2real
Config: `configs/baseline_supervised_r112_magformer_r111recipe_target_labeled100_balanced_1000.yaml`

## Purpose

R112 keeps the R111 supervised MagFormer recipe unchanged and only expands labeled target coverage from 50 to 100 images.

No pseudo labels, Stage C, source replay, or new modules are introduced.

## Split Generation

Input files:

- Base labeled target50: `annotations/instances_target_labeled_r37_balanced_plus25.json`.
- Candidate remaining target175: `annotations/instances_target_unlabeled_r37_balanced_minus25.json`.

Generated files:

- Train100: `annotations/instances_target_labeled_r112_balanced_plus75.json`.
- Remaining125: `annotations/instances_target_unlabeled_r112_balanced_minus75.json`.

Selection rule:

- Treat hidden GT in remaining175 as newly promoted target annotations for this pseudo-real simulation.
- Group remaining175 images by annotation count: `low25 <=25`, `mid50 <=50`, `high100 >50`.
- Allocate 50 promoted slots by proportional largest-remainder rounding over those groups: `{'low25': 5, 'mid50': 33, 'high100': 12}`.
- Within each group, sort by per-image median mask area, then annotation count, file name, and image id.
- Pick deterministic evenly spaced indices from each sorted group.

Promoted coverage:

- Promoted bands: `{'low25': 5, 'mid50': 33, 'high100': 12}`.
- Promoted exact instance counts: `{25: 5, 49: 1, 50: 32, 94: 1, 95: 1, 97: 1, 98: 2, 99: 1, 100: 6}`.

Interpretation:

- `train100` uses promoted GT as additional human labels in the pseudo-real simulation.
- `remaining125` is the non-leakage target split for formal target evaluation.
- `full200` remains reference only because it includes promoted training images.

## R111 Recipe Held Fixed

- Warm-start from `output/baseline/r100_magformer_r98ckpt_target_labeled25_cleanfit_1024/checkpoint_iter_0000500.pth` through `model.finetune_weights`.
- `runtime.resume: null`.
- `dice_weight: 10.0`.
- `mask_weight: 5.0`.
- `importance_sample_ratio: 0.0`.
- `train_num_points: 12544`.
- `oversample_ratio: 3.0`.
- No RGB augmentation, no depth noise, image size `1024`.
- `runtime.depth_sanity.min_mask_fg_ratio: 0.0009`.

## Formal Metrics

Primary non-leakage metrics:

- `val28`: `annotations/instances_val.json`, split `val`.
- `remaining125`: `annotations/instances_target_unlabeled_r112_balanced_minus75.json`, split `train`.

Reference-only metric:

- `full200`: `annotations/instances_target_unlabeled.json`, split `train`.

## Success Criteria

- Formal target improvement over R111: remaining125 segm AP should be compared against R111 non-overlap175 segm AP `0.339053` with the split difference stated.
- Target threshold: remaining125 segm AP `>= 0.350`.
- Val28 threshold: val28 segm AP `>= 0.305`.
- Train-fit threshold: train100 segm AP `>= 0.61`.
- Formal 61+ goal: achieved only if train100 reaches `0.61+` and the non-leakage target metrics do not regress against the stated criteria.

## Static Validation

Static validation passed before training.

- `validate_config(strict=True)`: valid. The only message was the existing DPE informational note.
- Checked config fields: `runtime.resume=null`, warm-start through `model.finetune_weights`, train100 annotation, `importance_sample_ratio=0.0`, `dice_weight=10.0`, `mask_weight=5.0`, `train_num_points=12544`, no RGB augmentation, no depth noise, `runtime.depth_sanity.min_mask_fg_ratio=0.0009`, and `vc_suda.enabled=false`.
- Annotation counts: train100 `100` images / `6125` anns, val28 `28` images / `1892` anns, remaining125 `125` images / `7322` anns, full200 reference `200` images / `11750` anns.
- Empty-annotation image ratio: train100 `0.000000`, val28 `0.000000`, remaining125 `0.000000`, full200 `0.000000`.
- File-name overlap: train100 vs remaining125 `0`; train100 vs val28 `0`; remaining125 vs val28 `0`; base train50 vs remaining125 `0`.
- Loader smoke checked: raw dataset lengths train/val `100 / 28`; DataLoader dataset lengths train/val `100 / 25` because `runtime.eval_max_images=25`; first train batch images `[4, 3, 1024, 1024]`, depths `[4, 1, 1024, 1024]`; first val batch images `[4, 3, 512, 512]`, depths `[4, 1, 512, 512]`.

## Training Placeholder

To fill after training:

- tmux session: `r112_magformer_r111recipe_target100`.
- Command shape.
- Output dir and log path.
- Checkpoint trajectory at iter500, iter800, iter1000.

## Results Placeholder

To fill after external eval:

| eval | images | GT anns | predictions | bbox AP | bbox AP50 | bbox AP75 | segm AP | segm AP50 | segm AP75 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train100 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| val28 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| remaining125 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| full200 reference | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| original first50 source sanity | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## Judgment Placeholder

To fill after eval:

- remaining125 `>= 0.350`: TBD.
- val28 `>= 0.305`: TBD.
- train100 `>= 0.61`: TBD.
- Formal comparison with R111: TBD.
- 61+ formal target status: TBD.

## Commits

- Config/docs/split placeholder: TBD.
- Final docs-only update: TBD.
