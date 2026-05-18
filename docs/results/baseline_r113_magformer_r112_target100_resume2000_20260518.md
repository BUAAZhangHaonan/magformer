# R113 MagFormer R112 Target100 Resume2000

Date: 2026-05-18
Branch: feature/vc-suda-sim2real
Config: `configs/baseline_supervised_r113_magformer_r112_target100_resume2000.yaml`

## Purpose

R113 only tests whether R112 target100 was under-trained.

The data split and recipe stay fixed from R112. The run true-resumes from the R112 `checkpoint_iter_0001000.pth` training state and extends `solver.max_iter` to `2000`.

No target150 expansion, pseudo labels, Stage C, source replay, or new modules are introduced.

## Unique Variable

- R112: target100, max_iter `1000`.
- R113: same target100 data and recipe, true resume from R112 iter1000, max_iter `2000`.

The warm-start path is disabled with `model.finetune_weights: null` so runtime resume is the only initialization path.

## Under-Training Hypothesis

R112 reached only about 40 equivalent epochs on target100. R111 target50 reached about 80 equivalent epochs and had higher train-fit.

If R113 raises train100 fit while val28 and remaining125 stay stable, R112 was likely under-trained rather than limited only by the target100 data.

## Midpoint Rule

Evaluate `checkpoint_iter_0001500.pth` on:

- train100
- val28
- remaining125

Stop and record if any midpoint guard fails:

- val28 segm AP `< 0.300`
- remaining125 segm AP `< 0.356`
- train100 segm AP `< 0.548`

Otherwise continue to iter2000.

## Success Criteria

Primary criteria:

- train100 segm AP `>= 0.61`
- val28 segm AP `>= 0.305`
- remaining125 segm AP `>= 0.360`

Preferred target-retention criterion:

- remaining125 should not fall below R112 `0.365856`.

## Static Validation

Static validation passed before training.

- `validate_config(strict=True)`: valid. The only message was the existing DPE informational note.
- True resume checkpoint exists: `output/baseline/r112_magformer_r111recipe_target_labeled100_balanced_1000/checkpoint_iter_0001000.pth`.
- Resume checkpoint training state: `iter=1000`, with optimizer, LR scheduler, and scaler state present.
- `model.finetune_weights: null`, so warm-start is disabled while runtime resume is active.
- `solver.max_iter: 2000`, `runtime.eval_period: 500`, `runtime.checkpoint_period: 100`.
- Annotation counts: train100 `100` images / `6125` anns, val28 `28` images / `1892` anns, remaining125 `125` images / `7322` anns, full200 reference `200` images / `11750` anns.
- Empty-annotation image ratio: train100 `0.000000`, val28 `0.000000`, remaining125 `0.000000`, full200 `0.000000`.
- File-name overlap: train100 vs remaining125 `0`; train100 vs val28 `0`; remaining125 vs val28 `0`.
- Loader smoke checked: raw dataset lengths train/val `100 / 28`; DataLoader dataset lengths train/val `100 / 25` because `runtime.eval_max_images=25`; first train batch images `[1, 3, 1024, 1024]`, depths `[1, 1, 1024, 1024]`; first val batch images `[4, 3, 512, 512]`, depths `[4, 1, 512, 512]`.

## Training

Pending.

## Midpoint Eval

Pending.

## Final Eval

Pending.

## Judgment

Pending.
