# R114 MagFormer R113-Warm Target150 Balanced

Date: 2026-05-18
Branch: feature/vc-suda-sim2real
Config: `configs/baseline_supervised_r114_magformer_r113warm_target150_balanced_2000.yaml`

## Purpose

R114 expands labeled target coverage from R113/R112 target100 to target150.

This is not a true resume. It uses the R113 iter2000 checkpoint as a model-only warm-start through `model.finetune_weights`, while optimizer, scheduler, and AMP scaler start fresh with `runtime.resume: null`.

No pseudo labels, Stage C, source replay, or new modules are introduced.

## Split Generation

Input files:

- Base train100: `annotations/instances_target_labeled_r112_balanced_plus75.json`.
- Candidate remaining125: `annotations/instances_target_unlabeled_r112_balanced_minus75.json`.

Generated files:

- Train150: `annotations/instances_target_labeled_r114_balanced_plus125.json`.
- Remaining75: `annotations/instances_target_unlabeled_r114_balanced_minus125.json`.

Selection rule:

- Treat hidden GT in remaining125 as newly promoted target annotations for this pseudo-real simulation.
- Group remaining125 images by annotation count: `low25 <=25`, `mid50 <=50`, `high100 >50`.
- Allocate 50 promoted slots using the R112 proportional largest-remainder recipe: `{low25: 5, mid50: 33, high100: 12}`.
- Within each group, sort by per-image median mask area, then annotation count, file name, and image id.
- Pick deterministic evenly spaced centered indices: `floor((i + 0.5) * n / k)`.

Promoted coverage:

- Promoted bands: `{low25: 5, mid50: 33, high100: 12}`.
- Candidate bands in remaining125: `{'mid50': 82, 'high100': 29, 'low25': 14}`.
- Promoted exact instance counts: `{25: 5, 50: 33, 92: 1, 97: 1, 98: 1, 99: 4, 100: 5}`.

## Non-Leakage Metrics

Primary non-leakage metrics:

- `val28`: `annotations/instances_val.json`, split `val`.
- `remaining75`: `annotations/instances_target_unlabeled_r114_balanced_minus125.json`, split `train`.

Reference-only metric:

- `full200`: `annotations/instances_target_unlabeled.json`, split `train`.

The `full200` row includes promoted training images, so it is only a continuity reference against R113 full200 `0.437321`.

## Remaining75 Risk

`remaining75` is smaller than R113 `remaining125`, so its AP can move more from individual-image composition. The fixed `val28` row is the stable anchor. Remaining75 is still required to stay above the requested threshold.

## Recipe Held Fixed

- Warm-start from `output/baseline/r113_magformer_r112_target100_resume2000/checkpoint_iter_0002000.pth` through `model.finetune_weights`.
- `runtime.resume: null`.
- `solver.max_iter: 2000`.
- `base_lr: 5e-5`, cosine schedule, batch size `4`.
- Image size `1024`.
- No RGB augmentation and no depth noise.
- `importance_sample_ratio: 0.0`.
- `dice_weight: 10.0`, `mask_weight: 5.0`.
- `runtime.depth_sanity.min_mask_fg_ratio: 0.0009`.

## Static Validation

Static validation passed before training.

- `validate_config(strict=True)`: valid. The only message was the existing DPE informational note.
- Warm-start checkpoint exists: `output/baseline/r113_magformer_r112_target100_resume2000/checkpoint_iter_0002000.pth`.
- Config checks: `model.finetune_weights` points to R113 iter2000, `runtime.resume=null`, `solver.max_iter=2000`, train annotation points to R114 train150, no RGB augmentation, no depth noise, `importance_sample_ratio=0.0`, `dice_weight=10.0`, `mask_weight=5.0`, and `runtime.depth_sanity.min_mask_fg_ratio=0.0009`.
- Annotation counts: train150 `150` images / `9083` anns, val28 `28` images / `1892` anns, remaining75 `75` images / `4364` anns, full200 reference `200` images / `11750` anns.
- Empty-annotation image ratio: train150 `0.000000`, val28 `0.000000`, remaining75 `0.000000`, full200 `0.000000`.
- File-name overlap: train150 vs remaining75 `0`; train150 vs val28 `0`; remaining75 vs val28 `0`.
- Loader smoke checked: raw dataset lengths train/val `150 / 28`; DataLoader dataset lengths train/val `150 / 25` because `runtime.eval_max_images=25`; first train batch images `[4, 3, 1024, 1024]`, depths `[4, 1, 1024, 1024]`; first val batch images `[4, 3, 512, 512]`, depths `[4, 1, 512, 512]`.

Current split counts from generation:

| split | images | annotations | empty images |
| --- | ---: | ---: | ---: |
| train150 | 150 | 9083 | 0 |
| remaining75 | 75 | 4364 | 0 |
| val28 | 28 | 1892 | 0 |
| full200 reference | 200 | 11750 | 0 |

## Gate Rules

Iter1000 evals: train150, val28, remaining75, full200 reference.

Stop at iter1000 if any guard triggers:

- val28 segm AP `< 0.300`.
- remaining75 segm AP `< 0.340`.
- train150 segm AP `< 0.55` and full200 reference segm AP `<= 0.437321`.

Iter1500 evals: same metrics.

Stop at iter1500 if any guard triggers:

- val28 segm AP `< 0.300`.
- remaining75 segm AP `< 0.345`.
- train150 improves `< 0.005` while val28 and remaining75 move down.

Otherwise train to iter2000.

## Success Criteria

- train150 segm AP `>= 0.61`.
- val28 segm AP `>= 0.305`.
- remaining75 segm AP `>= 0.350`.
- full200 reference should exceed R113 full200 reference `0.437321`.
- Formal 61+ goal is achieved only if train150 reaches `0.61+` and non-leakage target metrics pass.

## Results

Pending training and evaluation.
