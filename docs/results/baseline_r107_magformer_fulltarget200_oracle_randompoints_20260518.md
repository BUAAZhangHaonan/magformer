# R107 MagFormer Full-Target200 Oracle Random Points

Date: 2026-05-18
Branch: feature/vc-suda-sim2real
Config: `configs/baseline_supervised_r107_magformer_fulltarget200_oracle_randompoints_1000.yaml`

## Purpose

R107 tests one variable against R106 full-target200 oracle warm-start: `model.magformer.mask_former.importance_sample_ratio` changes from `0.75` to `0.0`.

The hypothesis is that random point sampling gives more foreground mask-loss points on the small full-target oracle set. If train-fit improves, the R106 point sampler was likely a bottleneck.

## Fixed Controls

- `train_num_points: 12544`.
- `oversample_ratio: 3.0`.
- `balanced_ce: true`.
- `balanced_ce_min_fg_ratio: 0.05`.
- Warm-start checkpoint, solver, data, and depth-sanity settings match R106.
- `runtime.skip_depth_sanity: false`.
- `runtime.depth_sanity.min_mask_fg_ratio: 0.0009`.
- VC-SUDA, EMA, pseudo bank, source replay, contrastive, and postprocess modules stay disabled.

## Gates

- Early stop failure: if iter499 train200 segm AP is below R106 iter499 `0.426482`.
- Success: iter1000 train200 segm AP must be at least `0.4565`.
- Train-fit target check: record whether train200 reaches `0.61+`.

## Results

Pending.

## Judgment

Pending.
