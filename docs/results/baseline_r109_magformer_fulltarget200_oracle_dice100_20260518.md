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

- tmux session: `r109_magformer_fulltarget200_oracle_dice100`.
- Command shape: `/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/torchrun --standalone --nproc_per_node=4 tools/train.py --config configs/baseline_supervised_r109_magformer_fulltarget200_oracle_dice100_1000.yaml --gpus 4,5,6,7`.
- Output dir: `output/baseline/r109_magformer_fulltarget200_oracle_dice100_1000`.
- Log: `output/baseline/r109_magformer_fulltarget200_oracle_dice100_1000.tmux.log`.
- Warm-start evidence: all ranks printed `Warm-start missing keys: 0, unexpected keys: 0`.
- Depth sanity evidence: `depth_sanity.json` was written and training continued.
- Start evidence: `start iter=0/1000`.
- Finished: `2026-05-18T08:42:17+08:00`.
- Exit state: `TRAIN_EXIT:0`; no OOM; GPUs 4-7 released.

## External Eval

Protocol for target rows: bbox+segm, model input `1024`, score threshold `0.05`, mask threshold `0.5`, inference topk `200`, COCO maxDets `200`, `MAGFORMER_MS_DEFORM_ATTN_BACKEND=pytorch`, `--force-pytorch-msda`, strict load `774/774` keys.

| checkpoint | eval | predictions | bbox AP | bbox AP50 | bbox AP75 | segm AP | segm AP50 | segm AP75 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| iter0499 | train200 | 24250 | `0.476993` | `0.823535` | `0.504503` | `0.453816` | `0.790905` | `0.480830` |
| iter0499 | val28 | 3907 | `0.361168` | `0.716720` | `0.320617` | `0.324191` | `0.648425` | `0.293962` |
| iter0799 | train200 | 24164 | `0.500684` | `0.834125` | `0.542993` | `0.473347` | `0.806329` | `0.511923` |
| iter0799 | val28 | 3936 | `0.370111` | `0.726670` | `0.336630` | `0.328250` | `0.653587` | `0.298752` |
| iter1000 | train200 | 23941 | `0.500623` | `0.833646` | `0.542334` | `0.476059` | `0.807096` | `0.515388` |
| iter1000 | val28 | 3897 | `0.368435` | `0.724070` | `0.340563` | `0.327925` | `0.652891` | `0.299673` |

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
| original first50 source sanity | iter1000 | 4361 | `0.439993` | `0.734127` | `0.477534` | `0.493461` | `0.781149` | `0.546386` |
| optional target_labeled25 check | iter1000 | 3318 | `0.451875` | `0.793683` | `0.459030` | `0.435688` | `0.778016` | `0.447790` |

## Judgment

- Iter0499 early stop gate: pass. Train200 segm AP is `0.453816`, above `0.440`; AP75 is `0.480830`, above `0.468`.
- Iter0799 early stop gate: pass. Train200 segm AP is `0.473347`, above R108 iter0799 `0.463877`; AP75 is `0.511923`, above R108 `0.496536`.
- Final success gate: pass. Iter1000 train200 segm AP is `0.476059`, above `0.475` by `+0.001059`.
- Final AP75/gap gate: pass on both criteria. Train200 AP75 is `0.515388`, above `0.512`; bbox-segm AP gap is `0.024564`, below R108 gap `0.027410`.
- Versus R108 iter1000 train200: segm AP improves by `+0.010481` (`0.476059` vs `0.465578`); segm AP75 improves by `+0.014563` (`0.515388` vs `0.500825`).
- Versus R108 iter1000 val28: segm AP improves by `+0.001931` (`0.327925` vs `0.325994`) and stays above the `0.323` guard.
- Train200 `0.61+`: no. Best train200 segm AP is `0.476059`.
- Interpretation: R109 supports a positive dice dose-response from `7.5` to `10.0` on the 45-47 AP platform. The gain is real but small, and it still does not solve the `0.61+` train-fit target.

## Artifacts

- Train log: `output/baseline/r109_magformer_fulltarget200_oracle_dice100_1000.tmux.log`.
- Train200 iter0499: `output/diagnostics/r109_magformer_fulltarget200_oracle_dice100_iter0499_train200_1024_backmap_topk200_20260518`.
- Val28 iter0499: `output/diagnostics/r109_magformer_fulltarget200_oracle_dice100_iter0499_val28_1024_backmap_topk200_20260518`.
- Train200 iter0799: `output/diagnostics/r109_magformer_fulltarget200_oracle_dice100_iter0799_train200_1024_backmap_topk200_20260518`.
- Val28 iter0799: `output/diagnostics/r109_magformer_fulltarget200_oracle_dice100_iter0799_val28_1024_backmap_topk200_20260518`.
- Train200 iter1000: `output/diagnostics/r109_magformer_fulltarget200_oracle_dice100_iter1000_train200_1024_backmap_topk200_20260518`.
- Val28 iter1000: `output/diagnostics/r109_magformer_fulltarget200_oracle_dice100_iter1000_val28_1024_backmap_topk200_20260518`.
- Source first50 iter1000: `output/diagnostics/r109_magformer_fulltarget200_oracle_dice100_iter1000_original_first50_1024_backmap_20260518`.
- Target_labeled25 iter1000: `output/diagnostics/r109_magformer_fulltarget200_oracle_dice100_iter1000_target_labeled25_1024_backmap_topk200_20260518`.

## Commits

- Config/docs placeholder: `32643c10254e9a23e4ce841494b8ae9d01939ec3`.
- Final docs-only update: this docs-only commit.
