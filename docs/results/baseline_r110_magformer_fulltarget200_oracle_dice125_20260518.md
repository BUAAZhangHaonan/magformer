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
- `validate_config(strict=True)`: valid. Only the existing DPE informational note was printed.
- Loader smoke: train/val dataset lengths `200 / 28`; first train batch images `[1, 3, 1024, 1024]`, depths `[1, 1, 1024, 1024]`; first val batch images `[4, 3, 512, 512]`, depths `[4, 1, 512, 512]`.
- Depth sanity config passed schema as `DepthSanityConfig` with `min_mask_fg_ratio=0.0009`.

## Training

- tmux session: `r110_magformer_fulltarget200_oracle_dice125`.
- Command shape: `/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/torchrun --standalone --nproc_per_node=4 tools/train.py --config configs/baseline_supervised_r110_magformer_fulltarget200_oracle_dice125_1000.yaml --gpus 4,5,6,7`.
- Output dir: `output/baseline/r110_magformer_fulltarget200_oracle_dice125_1000`.
- Log: `output/baseline/r110_magformer_fulltarget200_oracle_dice125_1000.tmux.log`.
- Warm-start evidence: all ranks printed `Warm-start missing keys: 0, unexpected keys: 0`.
- Depth sanity evidence: `depth_sanity.json` was written and training continued.
- Start evidence: `start iter=0/1000`.
- Stop state: stopped after the iter0499 val guard failed. Training had already written checkpoints through `checkpoint_iter_0000699.pth` before the external eval result was available.
- Initial launch note: the first launch used `CUDA_VISIBLE_DEVICES=4,5,6,7` and failed before training with `invalid device ordinal`; it was relaunched with the R109 command shape and no `CUDA_VISIBLE_DEVICES`.

## External Eval

Protocol for target rows: bbox+segm, model input `1024`, score threshold `0.05`, mask threshold `0.5`, inference topk `200`, COCO maxDets `200`, `MAGFORMER_MS_DEFORM_ATTN_BACKEND=pytorch`, `--force-pytorch-msda`, strict load.

| checkpoint | eval | predictions | bbox AP | bbox AP50 | bbox AP75 | segm AP | segm AP50 | segm AP75 | bbox-segm AP gap |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| iter0499 | train200 | 27844 | `0.485729` | `0.828109` | `0.519512` | `0.462893` | `0.792648` | `0.496504` | `0.022836` |
| iter0499 | val28 | 4403 | `0.362667` | `0.730626` | `0.319312` | `0.325464` | `0.643783` | `0.293792` | `0.037203` |

Iter0799, iter1000, original first50 source sanity, and optional target_labeled25 were not run because the required iter0499 val guard stopped the run.

## Prediction Inflation Guards

- Stopped-checkpoint train200 pred/GT count ratio: `27844 / 11750 = 2.369702`, fail versus `<= 2.30`.
- Stopped-checkpoint val28 pred/GT total mask area ratio: `1535398 / 718484 = 2.136997`, fail versus `<= 1.90`.
- The guard result points to prediction or mask-area expansion at dice `12.5`, even though train200 AP improved at iter0499.

## Judgment

- Iter0499 train gate: pass. Train200 segm AP is `0.462893`, above `0.454`; AP75 is `0.496504`, above `0.481`.
- Iter0499 val guard: fail. Val28 segm AP is `0.325464`, below the required `0.326`, so the run stopped.
- Versus R109 iter0499 train200: segm AP improves by `+0.009078` (`0.462893` vs `0.453816`); segm AP75 improves by `+0.015674` (`0.496504` vs `0.480830`).
- Versus R109 iter0499 val28: segm AP improves by `+0.001273` (`0.325464` vs `0.324191`) but still misses the R110 staged val guard.
- Final R110 success gates: not reached because of early stop.
- Train200 `0.61+`: no. The run stopped before any final train-fit attempt, and iter0499 train200 segm AP was `0.462893`.
- Interpretation: R110 does not support continuing the dice dose-response to `12.5`. It improved the iter0499 train metrics, but failed the staged val guard and both prediction inflation guards.

## Artifacts

- Train log: `output/baseline/r110_magformer_fulltarget200_oracle_dice125_1000.tmux.log`.
- First failed launch log: `output/baseline/r110_magformer_fulltarget200_oracle_dice125_1000.failed_invalid_ordinal.tmux.log`.
- Eval monitor log: `output/baseline/r110_magformer_fulltarget200_oracle_dice125_eval_monitor.log`.
- Stop reason: `output/baseline/r110_magformer_fulltarget200_oracle_dice125_STOP_REASON.txt`.
- Train200 iter0499: `output/diagnostics/r110_magformer_fulltarget200_oracle_dice125_iter0499_train200_1024_backmap_topk200_20260518`.
- Val28 iter0499: `output/diagnostics/r110_magformer_fulltarget200_oracle_dice125_iter0499_val28_1024_backmap_topk200_20260518`.

## Commits

- Config/docs placeholder: `d0392826b4053744f169f0e84a89a01a2ae58925`.
- Final docs-only update: this docs-only commit.
