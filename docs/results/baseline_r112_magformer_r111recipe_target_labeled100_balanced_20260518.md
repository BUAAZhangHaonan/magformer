# R112 MagFormer R111 Recipe Target100 Balanced

Date: 2026-05-18
Branch: feature/vc-suda-sim2real
Config: `configs/baseline_supervised_r112_magformer_r111recipe_target_labeled100_balanced_1000.yaml`

## Purpose

R112 keeps the R111 supervised MagFormer recipe unchanged and only expands labeled target coverage from 50 to 100 images.

No pseudo labels, Stage C, source replay, or new modules are introduced.

## Split Generation

Input files:

- Base labeled target50: `annotations/instances_target_labeled_r37_balanced_plus25.json`.
- Candidate remaining target175: `annotations/instances_target_unlabeled_r37_balanced_minus25.json`.

Generated files:

- Train100: `annotations/instances_target_labeled_r112_balanced_plus75.json`.
- Remaining125: `annotations/instances_target_unlabeled_r112_balanced_minus75.json`.

Selection rule:

- Treat hidden GT in remaining175 as newly promoted target annotations for this pseudo-real simulation.
- Group remaining175 images by annotation count: `low25 <=25`, `mid50 <=50`, `high100 >50`.
- Allocate 50 promoted slots by proportional largest-remainder rounding over those groups: `{'low25': 5, 'mid50': 33, 'high100': 12}`.
- Within each group, sort by per-image median mask area, then annotation count, file name, and image id.
- Pick deterministic evenly spaced indices from each sorted group.

Promoted coverage:

- Promoted bands: `{'low25': 5, 'mid50': 33, 'high100': 12}`.
- Promoted exact instance counts: `{25: 5, 49: 1, 50: 32, 94: 1, 95: 1, 97: 1, 98: 2, 99: 1, 100: 6}`.

Interpretation:

- `train100` uses promoted GT as additional human labels in the pseudo-real simulation.
- `remaining125` is the non-leakage target split for formal target evaluation.
- `full200` remains reference only because it includes promoted training images.

## R111 Recipe Held Fixed

- Warm-start from `output/baseline/r100_magformer_r98ckpt_target_labeled25_cleanfit_1024/checkpoint_iter_0000500.pth` through `model.finetune_weights`.
- `runtime.resume: null`.
- `dice_weight: 10.0`.
- `mask_weight: 5.0`.
- `importance_sample_ratio: 0.0`.
- `train_num_points: 12544`.
- `oversample_ratio: 3.0`.
- No RGB augmentation, no depth noise, image size `1024`.
- `runtime.depth_sanity.min_mask_fg_ratio: 0.0009`.

## Formal Metrics

Primary non-leakage metrics:

- `val28`: `annotations/instances_val.json`, split `val`.
- `remaining125`: `annotations/instances_target_unlabeled_r112_balanced_minus75.json`, split `train`.

Reference-only metric:

- `full200`: `annotations/instances_target_unlabeled.json`, split `train`.

## Success Criteria

- Formal target improvement over R111: remaining125 segm AP should be compared against R111 non-overlap175 segm AP `0.339053` with the split difference stated.
- Target threshold: remaining125 segm AP `>= 0.350`.
- Val28 threshold: val28 segm AP `>= 0.305`.
- Train-fit threshold: train100 segm AP `>= 0.61`.
- Formal 61+ goal: achieved only if train100 reaches `0.61+` and the non-leakage target metrics do not regress against the stated criteria.

## Static Validation

Static validation passed before training.

- `validate_config(strict=True)`: valid. The only message was the existing DPE informational note.
- Checked config fields: `runtime.resume=null`, warm-start through `model.finetune_weights`, train100 annotation, `importance_sample_ratio=0.0`, `dice_weight=10.0`, `mask_weight=5.0`, `train_num_points=12544`, no RGB augmentation, no depth noise, `runtime.depth_sanity.min_mask_fg_ratio=0.0009`, and `vc_suda.enabled=false`.
- Annotation counts: train100 `100` images / `6125` anns, val28 `28` images / `1892` anns, remaining125 `125` images / `7322` anns, full200 reference `200` images / `11750` anns.
- Empty-annotation image ratio: train100 `0.000000`, val28 `0.000000`, remaining125 `0.000000`, full200 `0.000000`.
- File-name overlap: train100 vs remaining125 `0`; train100 vs val28 `0`; remaining125 vs val28 `0`; base train50 vs remaining125 `0`.
- Loader smoke checked: raw dataset lengths train/val `100 / 28`; DataLoader dataset lengths train/val `100 / 25` because `runtime.eval_max_images=25`; first train batch images `[4, 3, 1024, 1024]`, depths `[4, 1, 1024, 1024]`; first val batch images `[4, 3, 512, 512]`, depths `[4, 1, 512, 512]`.

## Training

- tmux session: `r112_magformer_r111recipe_target100`.
- Command shape: `/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/torchrun --standalone --nproc_per_node=4 tools/train.py --config configs/baseline_supervised_r112_magformer_r111recipe_target_labeled100_balanced_1000.yaml --gpus 4,5,6,7`.
- Output dir: `output/baseline/r112_magformer_r111recipe_target_labeled100_balanced_1000`.
- Log: `output/baseline/r112_magformer_r111recipe_target_labeled100_balanced_1000.tmux.log`.
- Training window: started `2026-05-18T10:25:12+08:00`, reached `training completed` at `2026-05-18 10:45:22 +0800`, and exited `TRAIN_EXIT:0`.
- Warm-start evidence: all ranks loaded `output/baseline/r100_magformer_r98ckpt_target_labeled25_cleanfit_1024/checkpoint_iter_0000500.pth` with `missing keys: 0, unexpected keys: 0`.
- Depth sanity evidence: `depth_sanity.json` was written and training continued.
- Checkpoints used for external eval: `checkpoint_iter_0000499.pth`, `checkpoint_iter_0000799.pth`, and `checkpoint_iter_0001000.pth`.
- Best-val checkpoint: `checkpoint_iter_0001000.pth`.
- No OOM, NaN/Inf loss, traceback, or runtime error was found in the training log.

## External Eval

Protocol for pseudo-real target rows: bbox+segm, model input `1024`, score threshold `0.05`, mask threshold `0.5`, inference topk `200`, COCO maxDets `200`, `MAGFORMER_MS_DEFORM_ATTN_BACKEND=pytorch`, `--force-pytorch-msda`, strict load `774/774` keys.

Val28 and remaining125 trajectory:

| checkpoint | eval | predictions | bbox AP | bbox AP50 | bbox AP75 | segm AP | segm AP50 | segm AP75 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| iter0499 | val28 | 3790 | `0.342703` | `0.697845` | `0.297595` | `0.303805` | `0.621658` | `0.259290` |
| iter0499 | remaining125 | 15193 | `0.411928` | `0.753097` | `0.404615` | `0.363347` | `0.685257` | `0.352682` |
| iter0799 | val28 | 3652 | `0.346260` | `0.702111` | `0.299235` | `0.307989` | `0.623760` | `0.266983` |
| iter0799 | remaining125 | 14338 | `0.410046` | `0.755173` | `0.403579` | `0.365565` | `0.684544` | `0.357031` |
| iter1000 | val28 | 3546 | `0.342340` | `0.701074` | `0.292360` | `0.308258` | `0.628971` | `0.267992` |
| iter1000 | remaining125 | 13978 | `0.410279` | `0.754606` | `0.409132` | `0.365856` | `0.684896` | `0.355083` |

The iter800 early-stop rule did not trigger: iter0799 val28 `0.307989` is not below `0.292`, and remaining125 `0.365565` is not below `0.339`.

Best-val checkpoint `iter1000` final eval:

| eval | images | GT anns | predictions | bbox AP | bbox AP50 | bbox AP75 | segm AP | segm AP50 | segm AP75 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train100 | 100 | 6125 | 9740 | `0.523787` | `0.843677` | `0.593691` | `0.528334` | `0.854196` | `0.600283` |
| val28 | 28 | 1892 | 3546 | `0.342340` | `0.701074` | `0.292360` | `0.308258` | `0.628971` | `0.267992` |
| remaining125 | 125 | 7322 | 13978 | `0.410279` | `0.754606` | `0.409132` | `0.365856` | `0.684896` | `0.355083` |
| full200 reference | 200 | 11750 | 21186 | `0.453909` | `0.791548` | `0.479147` | `0.424739` | `0.747483` | `0.445301` |
| original first50 source sanity | 50 | 3315 | 4396 | `0.348686` | `0.679968` | `0.322296` | `0.433445` | `0.741332` | `0.455530` |

Original first50 source sanity used `tools/evaluate_teacher_first50_1024_backmap.py` with `configs/finetune_1k_full_1024.yaml`, dataset root `magformer_datasets/20260318_1K_1566`, annotation `annotations/instances_all.json`, split `all`, max images `50`, inference topk `100`, and maxDets `100`.

Artifacts:

- Train log: `output/baseline/r112_magformer_r111recipe_target_labeled100_balanced_1000.tmux.log`.
- Val28 iter0499: `output/diagnostics/r112_magformer_r111recipe_target100_iter0499_val28_1024_backmap_topk200_20260518`.
- Remaining125 iter0499: `output/diagnostics/r112_magformer_r111recipe_target100_iter0499_remaining125_1024_backmap_topk200_20260518`.
- Val28 iter0799: `output/diagnostics/r112_magformer_r111recipe_target100_iter0799_val28_1024_backmap_topk200_20260518`.
- Remaining125 iter0799: `output/diagnostics/r112_magformer_r111recipe_target100_iter0799_remaining125_1024_backmap_topk200_20260518`.
- Val28 iter1000: `output/diagnostics/r112_magformer_r111recipe_target100_iter1000_val28_1024_backmap_topk200_20260518`.
- Remaining125 iter1000: `output/diagnostics/r112_magformer_r111recipe_target100_iter1000_remaining125_1024_backmap_topk200_20260518`.
- Train100 iter1000: `output/diagnostics/r112_magformer_r111recipe_target100_iter1000_train100_1024_backmap_topk200_20260518`.
- Full200 reference iter1000: `output/diagnostics/r112_magformer_r111recipe_target100_iter1000_full200_reference_1024_backmap_topk200_20260518`.
- Original first50 source sanity iter1000: `output/diagnostics/r112_magformer_r111recipe_target100_iter1000_original_first50_1024_backmap_20260518`.

## Judgment

R112 formally improves over R111 on the two non-leakage target checks that can be compared, with the caveat that remaining125 is a different non-overlap target split than R111 non-overlap175.

| comparison | R111 | R112 best | delta | judgment |
| --- | ---: | ---: | ---: | --- |
| val28 segm AP | `0.291622` | `0.308258` | `+0.016636` | improved |
| non-leakage target segm AP | `0.339053` on non-overlap175 | `0.365856` on remaining125 | `+0.026803` | improved, split differs |
| full200 reference segm AP | `0.370225` | `0.424739` | `+0.054514` | reference-only improved |

Success checks:

- remaining125 `>= 0.350`: yes, `0.365856`.
- val28 `>= 0.305`: yes, `0.308258`.
- train100 `>= 0.61`: no, `0.528334`.
- Formal comparison with R111: yes on val28 and non-leakage target AP.
- 61+ formal target status: not achieved, because train100 did not reach `0.61+`.

Conclusion: R112 achieves the formal target-cover expansion gain over R111 on held-out target evaluation, but it does not meet the 61+ train-fit goal.

## Commits

- Config/docs/split placeholder: `f69d3ec6`.
- Final docs-only update: this docs-only commit.
