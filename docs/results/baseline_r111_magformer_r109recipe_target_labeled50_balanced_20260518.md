# R111 MagFormer R109 Recipe Target50 Balanced

Date: 2026-05-18
Branch: feature/vc-suda-sim2real
Config: `configs/baseline_supervised_r111_magformer_r109recipe_target_labeled50_balanced_1000.yaml`

## Purpose

R111 migrates the R109 oracle-selected recipe package onto the formal non-leakage R104 target-labeled50 balanced training set.

This run can only be interpreted as the R109 recipe package transfer. It must not attribute effects to any single parameter.

## Recipe Package

- Warm-start from `output/baseline/r100_magformer_r98ckpt_target_labeled25_cleanfit_1024/checkpoint_iter_0000500.pth` through `model.finetune_weights`.
- `runtime.resume: null`.
- `dice_weight: 10.0`.
- `mask_weight: 5.0`.
- `importance_sample_ratio: 0.0`.
- `train_num_points: 12544`.
- `oversample_ratio: 3.0`.
- No RGB augmentation, no depth noise, image size `1024`.
- `runtime.depth_sanity.min_mask_fg_ratio: 0.0009` for small-instance preflight support.

## Formal Splits

Primary non-leakage metrics:

- `val28`: `annotations/instances_val.json`, split `val`.
- `non-overlap175`: `annotations/instances_target_unlabeled_r37_balanced_minus25.json`, target_unlabeled200 minus the 25 promoted training images.

Reference-only metric:

- `full200`: `annotations/instances_target_unlabeled.json`. This includes the 25 promoted training images and is leakage-prone for R111, so it is reference only.

## Baselines

R104 formal baseline:

| eval | segm AP | note |
| --- | ---: | --- |
| val28 | `0.283886` | primary |
| non-overlap175 | `0.329132` | primary |
| full200 reference | `0.340785` | reference only |

Historical R80 reference:

- full200 segm AP: `0.336383`.

## Success Criteria

- Main success: R111 best checkpoint beats R104 non-overlap175 segm AP `0.329132` under the same external 1024 backmap protocol.
- Secondary support: R111 val28 beats R104 val28 segm AP `0.283886`.
- Reference-only: full200 may be compared with R104 `0.340785` and R80 `0.336383`, but it cannot drive the formal conclusion.
- Train-fit check: record whether train50 reaches `0.61+` segm AP.

## Static Validation

Static validation passed before training.

- `validate_config(strict=True)`: valid. The only message was the existing DPE informational note.
- Checked config fields: `runtime.resume=null`, warm-start through `model.finetune_weights`, train50 annotation, val28 annotation/split, `importance_sample_ratio=0.0`, `dice_weight=10.0`, `mask_weight=5.0`, `train_num_points=12544`, `oversample_ratio=3.0`, and `runtime.depth_sanity.min_mask_fg_ratio=0.0009`.
- Annotation counts: train50 `50` images / `3170` anns, val28 `28` images / `1892` anns, non-overlap175 `175` images / `10277` anns, full200 `200` images / `11750` anns.
- Non-overlap175 file exists at `annotations/instances_target_unlabeled_r37_balanced_minus25.json`.
- File-name overlap: train50 vs val28 `0`; train50 vs non-overlap175 `0`; full200 minus non-overlap175 `25` images.
- Loader smoke checked: dataset lengths `50 / 28`; first train batch images `[1, 3, 1024, 1024]`, depths `[1, 1, 1024, 1024]`; first val batch images `[4, 3, 512, 512]`, depths `[4, 1, 512, 512]`.

## Training

- tmux session: `r111_magformer_r109recipe_target50`.
- Command shape: `/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/torchrun --standalone --nproc_per_node=4 tools/train.py --config configs/baseline_supervised_r111_magformer_r109recipe_target_labeled50_balanced_1000.yaml --gpus 4,5,6,7`.
- Output dir: `output/baseline/r111_magformer_r109recipe_target_labeled50_balanced_1000`.
- Log: `output/baseline/r111_magformer_r109recipe_target_labeled50_balanced_1000.tmux.log`.
- Warm-start evidence: all ranks loaded `output/baseline/r100_magformer_r98ckpt_target_labeled25_cleanfit_1024/checkpoint_iter_0000500.pth` with `missing keys: 0, unexpected keys: 0`.
- Depth sanity evidence: `depth_sanity.json` was written and training continued.
- Training window: started `2026-05-18T09:35:59+08:00`, reached `training completed` at `2026-05-18 09:56:23 +0800`, and exited `TRAIN_EXIT:0` at `2026-05-18T09:56:34+08:00`.
- Checkpoints used for external eval: `checkpoint_iter_0000499.pth`, `checkpoint_iter_0000799.pth`, and `checkpoint_iter_0001000.pth`.
- Final checkpoint: `output/baseline/r111_magformer_r109recipe_target_labeled50_balanced_1000/checkpoint_iter_0001000.pth`.
- No OOM, NaN, traceback, or runtime error was found in the training log. GPU 4-7 memory stayed below the available 24GB/card.

Built-in trainer eval is diagnostic only. It evaluated only 7 images at the final DDP step and is not used for any conclusion below.

## External Eval

Protocol for target rows: bbox+segm, model input `1024`, score threshold `0.05`, mask threshold `0.5`, inference topk `200`, COCO maxDets `200`, `MAGFORMER_MS_DEFORM_ATTN_BACKEND=pytorch`, `--force-pytorch-msda`, strict load `774/774` keys.

Val28 trajectory:

| checkpoint | predictions | bbox AP | bbox AP50 | bbox AP75 | segm AP | segm AP50 | segm AP75 | decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| iter0499 | 3364 | `0.325286` | `0.676381` | `0.273908` | `0.287944` | `0.609109` | `0.242591` | above R104 val28 |
| iter0799 | 3250 | `0.319353` | `0.668743` | `0.269861` | `0.289316` | `0.602030` | `0.255100` | above R104 val28 |
| iter1000 | 3209 | `0.322997` | `0.671994` | `0.274204` | `0.291622` | `0.603718` | `0.252863` | best val checkpoint |

Best-val checkpoint `iter1000` final eval:

| eval | images | GT anns | predictions | bbox AP | bbox AP50 | bbox AP75 | segm AP | segm AP50 | segm AP75 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train50 | 50 | 3170 | 4315 | `0.559680` | `0.860117` | `0.649784` | `0.594213` | `0.896658` | `0.705432` |
| val28 | 28 | 1892 | 3209 | `0.322997` | `0.671994` | `0.274204` | `0.291622` | `0.603718` | `0.252863` |
| non-overlap175 | 175 | 10277 | 17936 | `0.382097` | `0.725028` | `0.368800` | `0.339053` | `0.654743` | `0.316917` |
| full200 reference | 200 | 11750 | 19956 | `0.406006` | `0.747099` | `0.404650` | `0.370225` | `0.683348` | `0.364947` |
| original first50 source sanity | 50 | 3315 | 4178 | `0.393636` | `0.705726` | `0.407877` | `0.479651` | `0.759257` | `0.529190` |

Original first50 source sanity used `tools/evaluate_teacher_first50_1024_backmap.py` with `configs/finetune_1k_full_1024.yaml`, dataset root `magformer_datasets/20260318_1K_1566`, annotation `annotations/instances_all.json`, split `all`, max images `50`, inference topk `100`, and maxDets `100`.

Artifacts:

- Train log: `output/baseline/r111_magformer_r109recipe_target_labeled50_balanced_1000.tmux.log`.
- Val28 iter0499: `output/diagnostics/r111_magformer_r109recipe_target50_iter0499_val28_1024_backmap_topk200_20260518`.
- Val28 iter0799: `output/diagnostics/r111_magformer_r109recipe_target50_iter0799_val28_1024_backmap_topk200_20260518`.
- Val28 iter1000: `output/diagnostics/r111_magformer_r109recipe_target50_iter1000_val28_1024_backmap_topk200_20260518`.
- Train50 iter1000: `output/diagnostics/r111_magformer_r109recipe_target50_iter1000_train50_1024_backmap_topk200_20260518`.
- Non-overlap175 iter1000: `output/diagnostics/r111_magformer_r109recipe_target50_iter1000_nonoverlap175_1024_backmap_topk200_20260518`.
- Full200 reference iter1000: `output/diagnostics/r111_magformer_r109recipe_target50_iter1000_full200_reference_1024_backmap_topk200_20260518`.
- Original first50 source sanity iter1000: `output/diagnostics/r111_magformer_r109recipe_target50_iter1000_original_first50_1024_backmap_20260518`.

## Judgment

R111 formally beats R104 on the non-leakage primary target split.

| comparison | R104 | R111 | delta | judgment |
| --- | ---: | ---: | ---: | --- |
| val28 segm AP | `0.283886` | `0.291622` | `+0.007736` | improved |
| non-overlap175 segm AP | `0.329132` | `0.339053` | `+0.009921` | formal win |
| full200 reference segm AP | `0.340785` | `0.370225` | `+0.029440` | reference-only improved |

R111 full200 reference segm AP `0.370225` is above both R104 full200 reference `0.340785` and historical R80 full200 `0.336383`, but this remains reference-only because full200 includes promoted training images.

Train50 segm AP is `0.594213`, so R111 does not reach `0.61+` train-fit.

Conclusion: the R109 recipe package transfers positively to the formal R104 target-labeled50 balanced setting. This conclusion is about the R109 recipe package as a whole: warm-start from R100, `runtime.resume=null`, `importance_sample_ratio=0.0`, `dice_weight=10.0`, and `mask_weight=5.0`. It must not be attributed to any single parameter in isolation.

## Commits

- Config/docs placeholder: `642cd8d1`.
- Final docs-only update: this docs-only commit.
