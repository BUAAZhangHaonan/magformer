# R110 MagFormer Full-Target200 Oracle Dice125

Date: 2026-05-18
Branch: feature/vc-suda-sim2real
Config: `configs/baseline_supervised_r110_magformer_fulltarget200_oracle_dice125_1000.yaml`

## Purpose

R110 tests one variable against R109 full-target200 oracle dice100: `model.magformer.mask_former.dice_weight` changes from `10.0` to `12.5`.

The dose-response hypothesis is that stronger dice pressure should continue the R109 gain on train200 mask AP and AP75 without hurting val28.

## Fixed Controls

- `mask_weight: 5.0`.
- `importance_sample_ratio: 0.0`.
- `train_num_points: 12544`.
- `oversample_ratio: 3.0`.
- `balanced_ce: true`.
- `balanced_ce_min_fg_ratio: 0.05`.
- Warm-start checkpoint, solver, data, depth normalization, and depth-sanity settings match R109.
- `runtime.skip_depth_sanity: false`.
- `runtime.depth_sanity.min_mask_fg_ratio: 0.0009`.
- VC-SUDA, EMA, pseudo bank, source replay, contrastive, and postprocess modules stay disabled.

## Gates

- Iter0499 early stop failure: stop if train200 segm AP `< 0.454` and AP75 `< 0.481`.
- Iter0799 early stop failure: stop if train200 segm AP `< 0.473347` and AP75 `< 0.511923`.
- Any staged val guard failure: stop if val28 segm AP `< 0.326`.
- Final success: train200 segm AP `>= 0.482`, train200 AP75 `>= 0.525`, val28 segm AP `>= 0.3275`, and val28 AP75 `>= 0.300`.
- Prediction inflation guard: train200 pred/GT count ratio `<= 2.30`.
- Mask-area inflation guard: val28 pred/GT total mask area ratio `<= 1.90`.
- Train-fit target check: record whether train200 reaches `0.61+`.

## Static Validation

- Unique intended config variable versus R109: `model.magformer.mask_former.dice_weight: 10.0 -> 12.5`.
- Other config changes are only run naming/output paths.
- `runtime.resume: null`.
- `mask_weight: 5.0`, `importance_sample_ratio: 0.0`, `train_num_points: 12544`, `oversample_ratio: 3.0`.
- `balanced_ce: true`, `balanced_ce_min_fg_ratio: 0.05`.
- `runtime.skip_depth_sanity: false`, `runtime.depth_sanity.min_mask_fg_ratio: 0.0009`.
- Disabled paths checked in config: VC-SUDA, runtime EMA, contrastive.

## Training

Pending.

## External Eval

Pending.

## Prediction Inflation Guards

Pending.

## Judgment

Pending.

## Artifacts

Pending.

## Commits

Pending.
