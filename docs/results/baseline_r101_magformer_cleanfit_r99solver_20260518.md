# R101 MagFormer Cleanfit R99 Solver Ablation

Date: 2026-05-18
Branch: feature/vc-suda-sim2real
Config: configs/baseline_supervised_r101_magformer_cleanfit_r99solver_1024.yaml

## Goal

Separate whether the R100 gain comes from optimizer/LR/schedule or from disabling augmentation/noise.

## Setup

R101 keeps the R100 clean/no-aug data setup and R98 initialization, but switches the solver back to the R99 weak solver:

- base_lr: 1.0e-6
- warmup_factor: 0.001
- warmup_iters: 50
- weight_decay: 0.07
- lr_scheduler: cosine

## Results

Pending.

## Gate

Evaluate train25 first. If train25 segm AP75 < 0.10, stop and do not evaluate val28 or target_unlabeled200.

## Conclusion

Pending.
