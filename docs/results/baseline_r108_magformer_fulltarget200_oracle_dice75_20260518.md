# R108 MagFormer Full-Target200 Oracle Dice75

Date: 2026-05-18
Branch: feature/vc-suda-sim2real
Config: `configs/baseline_supervised_r108_magformer_fulltarget200_oracle_dice75_1000.yaml`

## Purpose

R108 tests one variable against R107 full-target200 oracle random-points: `model.magformer.mask_former.dice_weight` changes from `5.0` to `7.5`.

The hypothesis is that mask shape or boundary supervision is limiting oracle train-fit. If train200 segm AP and AP75 improve without val28 regression, the dice/boundary target is a plausible bottleneck.

## Fixed Controls

- `importance_sample_ratio: 0.0`.
- `train_num_points: 12544`.
- `oversample_ratio: 3.0`.
- `balanced_ce: true`.
- `balanced_ce_min_fg_ratio: 0.05`.
- Warm-start checkpoint, solver, data, and depth-sanity settings match R107.
- `runtime.skip_depth_sanity: false`.
- `runtime.depth_sanity.min_mask_fg_ratio: 0.0009`.
- VC-SUDA, EMA, pseudo bank, source replay, contrastive, and postprocess modules stay disabled.

## Gates

- Early stop failure: if iter0499 train200 segm AP is below R107 iter0499 `0.431359` and AP75 does not improve.
- Success: iter1000 train200 segm AP must be at least `0.4565`, with AP75 or bbox-segm gap improved.
- Val guard: val28 should not clearly regress versus R107 val28 segm AP `0.321195`.
- Train-fit target check: record whether train200 reaches `0.61+`.

## Static Validation

- Unique intended config variable versus R107: `model.magformer.mask_former.dice_weight: 5.0 -> 7.5`.
- Other config changes are only run naming/output paths.
- `runtime.resume: null`.
- `model.finetune_weights: output/baseline/r100_magformer_r98ckpt_target_labeled25_cleanfit_1024/checkpoint_iter_0000500.pth`.
- `importance_sample_ratio: 0.0`, `train_num_points: 12544`, `oversample_ratio: 3.0`.
- `balanced_ce: true`, `balanced_ce_min_fg_ratio: 0.05`.
- `runtime.skip_depth_sanity: false`, `runtime.depth_sanity.min_mask_fg_ratio: 0.0009`.
- Disabled paths checked in config: VC-SUDA, runtime EMA, contrastive.
- `validate_config(strict=True)`: valid. Only the existing DPE informational note was printed.
- Loader smoke: train/val dataset lengths `200 / 28`; first train batch images `[1, 3, 1024, 1024]`, depths `[1, 1, 1024, 1024]`, target mask counts `[50]`; first val batch images `[4, 3, 512, 512]`, depths `[4, 1, 512, 512]`.
- Depth sanity config passed schema as `DepthSanityConfig` with `min_mask_fg_ratio=0.0009`.

## Training

Pending.

## External Eval

Pending.

## Sanity Evals

Pending.

## Judgment

Pending.

## Artifacts

Pending.

## Commits

Pending.
