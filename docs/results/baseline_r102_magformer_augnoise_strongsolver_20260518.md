# R102 MagFormer Aug/Noise Strong Solver Ablation

Date: 2026-05-18
Branch: feature/vc-suda-sim2real
Config: `configs/baseline_supervised_r102_magformer_augnoise_strongsolver_1024.yaml`

## Goal

Test the missing factor corner: keep the R99 augmentation/noise setup, but use the R100 strong solver. This checks whether R99's flip/photo augmentation and depth noise can still fit and generalize once the weak R99 solver is removed.

## Setup

- Init: `output/baseline/r98_magformer_32k_source_short/checkpoint_iter_0000499.pth`
- Train: `target_labeled25`
- Aug/noise: R99 setup, with horizontal random flip, RGB photo augmentation enabled, and Gaussian depth noise enabled.
- Solver: R100 setup, with `base_lr=5e-5`, `warmup_iters=0`, `warmup_factor=1.0`, `weight_decay=0.01`, and `max_iter=500`.
- Loss: unchanged from R99/R100.
- VC-SUDA, EMA, pseudo-labeling, and contrastive paths: disabled.

## Gate

Evaluate external train25 first with the 1024 backmap protocol. If train25 segm AP75 is at least `0.10`, also evaluate val28 and target_unlabeled200. Otherwise stop after train25.

## Static Validation

Static config validation passed before training.

- `load_config`: loaded `baseline_supervised_r102_magformer_augnoise_strongsolver_1024`.
- Init matches R98 checkpoint: `output/baseline/r98_magformer_32k_source_short/checkpoint_iter_0000499.pth`.
- Train split is target_labeled25: `annotations/instances_target_labeled.json`.
- Aug/noise matches R99: `random_flip=horizontal`, `rgb_photo_aug.enabled=true`, `depth_noise.enabled=true`, `depth_noise.gaussian_std=0.01`.
- Solver matches R100: `base_lr=5e-5`, `warmup_iters=0`, `warmup_factor=1.0`, `weight_decay=0.01`, `max_iter=500`.
- Loss matches R99/R100.
- VC-SUDA, EMA, pseudo-labeling, and contrastive paths are off.

## Results

Pending.
