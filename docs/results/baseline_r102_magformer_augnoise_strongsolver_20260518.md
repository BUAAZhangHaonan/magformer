# R102 MagFormer Aug/Noise Strong Solver Ablation

Date: 2026-05-18
Branch: feature/vc-suda-sim2real
Config: `configs/baseline_supervised_r102_magformer_augnoise_strongsolver_1024.yaml`

## Conclusion

R102 is close to R100 on generalization, but lower than R100 on train25 AP75. Keeping R99 flip/photo aug/depth noise under the R100 strong solver still fits and generalizes; it does not collapse like R99/R101.

- Train25 segm AP75: R102 `0.351640` vs R100 `0.523051`.
- Val28 segm AP: R102 `0.267721` vs R100 `0.266639`.
- Target_unlabeled200 segm AP/AP75: R102 `0.310118` / `0.269491` vs R100 `0.312443` / `0.276846`.

## Goal

Test the missing factor corner: keep the R99 augmentation/noise setup, but use the R100 strong solver. This checks whether R99's flip/photo augmentation and depth noise can still fit and generalize once the weak R99 solver is removed.

## Setup

- Init: `output/baseline/r98_magformer_32k_source_short/checkpoint_iter_0000499.pth`
- Train: `target_labeled25`
- Aug/noise: R99 setup, with horizontal random flip, RGB photo augmentation enabled, and Gaussian depth noise enabled.
- Solver: R100 setup, with `base_lr=5e-5`, `warmup_iters=0`, `warmup_factor=1.0`, `weight_decay=0.01`, and `max_iter=500`.
- Loss: unchanged from R99/R100.
- VC-SUDA, EMA, pseudo-labeling, and contrastive paths: disabled.

## Static Validation

Static config validation passed before training.

- `load_config`: loaded `baseline_supervised_r102_magformer_augnoise_strongsolver_1024`.
- Init matches R98 checkpoint: `output/baseline/r98_magformer_32k_source_short/checkpoint_iter_0000499.pth`.
- Train split is target_labeled25: `annotations/instances_target_labeled.json`.
- Aug/noise matches R99: `random_flip=horizontal`, `rgb_photo_aug.enabled=true`, `depth_noise.enabled=true`, `depth_noise.gaussian_std=0.01`.
- Solver matches R100: `base_lr=5e-5`, `warmup_iters=0`, `warmup_factor=1.0`, `weight_decay=0.01`, `max_iter=500`.
- Loss matches R99/R100.
- VC-SUDA, EMA, pseudo-labeling, and contrastive paths are off.

## Training

- tmux session: `r102_magformer_augnoise_strongsolver`
- Command shape: `torchrun --nproc_per_node=4 tools/train.py --config configs/baseline_supervised_r102_magformer_augnoise_strongsolver_1024.yaml`
- Output dir: `output/baseline/r102_magformer_augnoise_strongsolver_1024`
- Warm-start load: `missing=0`, `unexpected=0` on all ranks.
- Training completed at `2026-05-18 04:12:12 +0800`.
- Final checkpoint: `output/baseline/r102_magformer_augnoise_strongsolver_1024/checkpoint_iter_0000500.pth`

Built-in trainer eval remained diagnostic only. It reported train25 segm AP75 around `0.0083`, while the external 1024 backmap gate below is the decision metric.

## External 1024 Backmap Eval

Protocol for all rows: bbox+segm, model input `1024`, score threshold `0.05`, mask threshold `0.5`, inference topk `200`, COCO maxDets `200`, `MAGFORMER_MS_DEFORM_ATTN_BACKEND=pytorch`, `--force-pytorch-msda`, strict load `774/774` keys.

| eval | images | GT anns | predictions | bbox AP | bbox AP50 | bbox AP75 | segm AP | segm AP50 | segm AP75 | topk truncated |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train25 | 25 | 1697 | 2447 | `0.421514` | `0.774641` | `0.406548` | `0.371849` | `0.720236` | `0.351640` | `0/25` |
| val28 | 28 | 1892 | 2753 | `0.328561` | `0.695655` | `0.276939` | `0.267721` | `0.588489` | `0.210378` | `0/28` |
| target_unlabeled200 | 200 | 11750 | 17401 | `0.377745` | `0.730451` | `0.349549` | `0.310118` | `0.642171` | `0.269491` | `0/200` |

Artifacts:

- Train25: `output/diagnostics/r102_magformer_augnoise_strongsolver_iter0500_train_labeled25_1024_backmap_topk200_20260518`
- Val28: `output/diagnostics/r102_magformer_augnoise_strongsolver_iter0500_val28_1024_backmap_topk200_20260518`
- Target_unlabeled200: `output/diagnostics/r102_magformer_augnoise_strongsolver_iter0500_target_unlabeled200_1024_backmap_topk200_20260518`

## Gate

R102 passes the train25 gate.

- Gate: continue if train25 segm AP75 >= `0.10`.
- Observed train25 segm AP75: `0.351640`.
- Val28 and target_unlabeled200 were evaluated because the gate passed.

## Comparison

| run | setup | train25 segm AP75 | val28 segm AP | target_unlabeled200 segm AP | target_unlabeled200 segm AP75 |
| --- | --- | ---: | ---: | ---: | ---: |
| R99 | R99 aug/noise + R99 weak solver | `0.017616` | `0.078015` | `0.082709` | `0.007090` |
| R101 | clean/no-aug + R99 weak solver | `0.025809` | not run | not run | not run |
| R100 | clean/no-aug + R100 strong solver | `0.523051` | `0.266639` | `0.312443` | `0.276846` |
| R102 | R99 aug/noise + R100 strong solver | `0.351640` | `0.267721` | `0.310118` | `0.269491` |

## Interpretation

R102 supports the solver-first interpretation. The R99 augmentation/noise package is not fatal when paired with the R100 solver: target_unlabeled200 AP is within `0.002325` of R100, and AP75 is within `0.007355` of R100.

The main cost is train-fit sharpness. R102 train25 AP75 is `0.171411` below R100, so aug/noise still weakens exact same-set localization. But the generalization result is close to R100, not close to R99/R101.

## Commits

- Config/docs placeholder: `6524feba`
- Final docs-only update: this docs-only commit.
