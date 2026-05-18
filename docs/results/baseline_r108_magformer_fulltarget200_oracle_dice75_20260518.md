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

- tmux session: `r108_magformer_fulltarget200_oracle_dice75`.
- Command shape: `/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/torchrun --standalone --nproc_per_node=4 tools/train.py --config configs/baseline_supervised_r108_magformer_fulltarget200_oracle_dice75_1000.yaml --gpus 4,5,6,7`.
- Output dir: `output/baseline/r108_magformer_fulltarget200_oracle_dice75_1000`.
- Log: `output/baseline/r108_magformer_fulltarget200_oracle_dice75_1000.tmux.log`.
- Warm-start evidence: all ranks printed `Warm-start missing keys: 0, unexpected keys: 0`.
- Depth sanity evidence: `depth_sanity.json` was written and training continued.
- Start evidence: `start iter=0/1000`.
- Finished: `2026-05-18T08:00:34+08:00`.
- Exit state: tmux session exited; no OOM; GPUs 4-7 released.

## External Eval

Protocol for target rows: bbox+segm, model input `1024`, score threshold `0.05`, mask threshold `0.5`, inference topk `200`, COCO maxDets `200`, `MAGFORMER_MS_DEFORM_ATTN_BACKEND=pytorch`, `--force-pytorch-msda`, strict load `774/774` keys.

| checkpoint | eval | predictions | bbox AP | bbox AP50 | bbox AP75 | segm AP | segm AP50 | segm AP75 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| iter0499 | train200 | 8369 | `0.474579` | `0.818970` | `0.500271` | `0.445688` | `0.782417` | `0.471674` |
| iter0499 | val28 | 3424 | `0.358849` | `0.715475` | `0.323937` | `0.319615` | `0.639968` | `0.287702` |
| iter0799 | train200 | 8190 | `0.491844` | `0.828185` | `0.533415` | `0.463877` | `0.797720` | `0.496536` |
| iter0799 | val28 | 3404 | `0.366566` | `0.718463` | `0.332487` | `0.325376` | `0.649752` | `0.291833` |
| iter1000 | train200 | 8173 | `0.492988` | `0.829200` | `0.534746` | `0.465578` | `0.798654` | `0.500825` |
| iter1000 | val28 | 3401 | `0.368340` | `0.724805` | `0.334711` | `0.325994` | `0.647374` | `0.292385` |

## Mask Area Check

Quick train200 matched-mask statistic: for each GT mask, choose the prediction in the same image with best mask IoU and report predicted mask area divided by GT mask area.

| run | best-IoU median | IoU>=0.5 matched GT count | matched area ratio median | matched area ratio p25/p75 |
| --- | ---: | ---: | ---: | ---: |
| R107 iter1000 | `0.757035` | 9366 | `1.031935` | `0.963981 / 1.119434` |
| R108 iter1000 | `0.766838` | 9601 | `1.022453` | `0.955607 / 1.105364` |

R108 increases matched count and best-IoU median, while the matched area ratio stays close to 1.0 and slightly tightens. This supports a shape/boundary improvement more than a simple area-scale correction.

## Sanity Evals

Original first50 source sanity used the guarded original-first50 protocol:

- Checker/wrapper: `tools/evaluate_teacher_first50_1024_backmap.py`.
- Base config: `configs/finetune_1k_full_1024.yaml`.
- Dataset root: `magformer_datasets/20260318_1K_1566`.
- Annotation: `annotations/instances_all.json`.
- Split/max-images: `all / 50`.
- Topk/maxDets: `100 / 100`.

| eval | checkpoint | predictions | bbox AP | bbox AP50 | bbox AP75 | segm AP | segm AP50 | segm AP75 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original first50 source sanity | iter1000 | 4585 | `0.415011` | `0.719229` | `0.431914` | `0.493190` | `0.770272` | `0.545398` |
| optional target_labeled25 check | iter1000 | 2936 | `0.448519` | `0.794531` | `0.460644` | `0.428303` | `0.775077` | `0.431996` |

## Judgment

- Early stop gate: pass. Iter0499 train200 segm AP is `0.445688`, above R107 iter0499 `0.431359` by `+0.014329`; AP75 is `0.471674`, above R107 `0.450030` by `+0.021644`.
- R108 success gate: pass. Iter1000 train200 segm AP is `0.465578`, above `0.4565` by `+0.009078`.
- Versus R107 iter1000 train200: segm AP improves by `+0.015899` (`0.465578` vs `0.449679`), segm AP75 improves by `+0.024665` (`0.500825` vs `0.476159`).
- Bbox-segm gap improves: AP gap `0.033521 -> 0.027410`; AP75 gap `0.043463 -> 0.033921`.
- Val28 does not regress: R108 iter1000 val28 segm AP is `0.325994`, above R107 `0.321195` by `+0.004799`.
- Train200 `0.61+`: no. Best train200 is `0.465578`.
- Interpretation: R108 supports the dice/boundary bottleneck hypothesis for the 45 AP platform. It breaks the R107 plateau and improves AP75 and bbox-segm gap, but it is still far below the `0.61+` train-fit target.

## Artifacts

- Train log: `output/baseline/r108_magformer_fulltarget200_oracle_dice75_1000.tmux.log`.
- Train200 iter0499: `output/diagnostics/r108_magformer_fulltarget200_oracle_dice75_iter0499_train200_1024_backmap_topk200_20260518`.
- Val28 iter0499: `output/diagnostics/r108_magformer_fulltarget200_oracle_dice75_iter0499_val28_1024_backmap_topk200_20260518`.
- Train200 iter0799: `output/diagnostics/r108_magformer_fulltarget200_oracle_dice75_iter0799_train200_1024_backmap_topk200_20260518`.
- Val28 iter0799: `output/diagnostics/r108_magformer_fulltarget200_oracle_dice75_iter0799_val28_1024_backmap_topk200_20260518`.
- Train200 iter1000: `output/diagnostics/r108_magformer_fulltarget200_oracle_dice75_iter1000_train200_1024_backmap_topk200_20260518`.
- Val28 iter1000: `output/diagnostics/r108_magformer_fulltarget200_oracle_dice75_iter1000_val28_1024_backmap_topk200_20260518`.
- Source first50 iter1000: `output/diagnostics/r108_magformer_fulltarget200_oracle_dice75_iter1000_original_first50_1024_backmap_20260518`.
- Target_labeled25 iter1000: `output/diagnostics/r108_magformer_fulltarget200_oracle_dice75_iter1000_target_labeled25_1024_backmap_topk200_20260518`.

## Commits

- Config/docs placeholder: `c1405e39e1b35410781bccfe65723110514e6acd`.
- Final docs-only update: this docs-only commit.
