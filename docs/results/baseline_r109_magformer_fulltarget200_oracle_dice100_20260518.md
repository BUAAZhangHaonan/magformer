# R109 MagFormer Full-Target200 Oracle Dice100

Date: 2026-05-18
Branch: feature/vc-suda-sim2real
Config: `configs/baseline_supervised_r109_magformer_fulltarget200_oracle_dice100_1000.yaml`

## Purpose

R109 tests one variable against R108 full-target200 oracle dice75: `model.magformer.mask_former.dice_weight` changes from `7.5` to `10.0`.

The dose-response hypothesis is simple: if the R108 gain came from stronger dice or boundary pressure, then increasing dice weight to `10.0` should improve train200 mask AP or AP75 again without hurting val28.

## Fixed Controls

- `mask_weight: 5.0`.
- `importance_sample_ratio: 0.0`.
- `train_num_points: 12544`.
- `oversample_ratio: 3.0`.
- `balanced_ce: true`.
- `balanced_ce_min_fg_ratio: 0.05`.
- Warm-start checkpoint, solver, data, depth normalization, and depth-sanity settings match R108.
- `runtime.skip_depth_sanity: false`.
- `runtime.depth_sanity.min_mask_fg_ratio: 0.0009`.
- VC-SUDA, EMA, pseudo bank, source replay, contrastive, and postprocess modules stay disabled.

## Gates

- Iter0499 early stop failure: stop if train200 segm AP `< 0.440` and AP75 `< 0.468`.
- Iter0799 early stop failure: stop if train200 segm AP `< 0.463877` and AP75 `< 0.496536`.
- Final success: iter1000 train200 segm AP `>= 0.475`.
- Final AP75 or gap success: train200 AP75 `>= 0.512` or bbox-segm AP gap remains below R108 gap `0.027410`.
- Val guard: val28 segm AP `>= 0.323`.
- Train-fit target check: record whether train200 reaches `0.61+`.

## Static Validation

- Unique intended config variable versus R108: `model.magformer.mask_former.dice_weight: 7.5 -> 10.0`.
- Other config changes are only run naming/output paths.
- `runtime.resume: null`.
- `mask_weight: 5.0`, `importance_sample_ratio: 0.0`, `train_num_points: 12544`, `oversample_ratio: 3.0`.
- `balanced_ce: true`, `balanced_ce_min_fg_ratio: 0.05`.
- `runtime.skip_depth_sanity: false`, `runtime.depth_sanity.min_mask_fg_ratio: 0.0009`.
- Disabled paths checked in config: VC-SUDA, runtime EMA, contrastive.
- `validate_config(strict=True)`: valid. Only the existing DPE informational note was printed.
- Loader smoke: train/val dataset lengths `200 / 28`; first train batch images `[1, 3, 1024, 1024]`, depths `[1, 1, 1024, 1024]`, target mask counts `[100]`; first val batch images `[4, 3, 512, 512]`, depths `[4, 1, 512, 512]`.
- Depth sanity config passed schema as `DepthSanityConfig` with `min_mask_fg_ratio=0.0009`.

## Training

- Pending.

## External Eval

- Pending.

## Sanity Evals

- Pending.

## Judgment

- Pending.

## Artifacts

- Pending.

## Commits

