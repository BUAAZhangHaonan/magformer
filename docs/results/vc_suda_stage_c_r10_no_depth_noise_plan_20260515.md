# VC-SUDA Stage C R10 No-Depth-Noise Continuation Plan - 2026-05-15

## Conclusion

R10 is the next controlled continuation after R9B threshold tuning did not improve from the available R8B `ckpt999`. It continues from R8B `ckpt999` and removes only the depth Gaussian perturbation by setting `data.depth_noise.gaussian_std: 0.0`.

## Setup

- Config: `configs/vc_suda_stage_c_r10_r8b_ckpt999_no_depth_noise_continue_1024_teacher8499.yaml`
- Output dir: `output/vc_suda/stage_c_r10_r8b_ckpt999_no_depth_noise_continue_1024_teacher8499`
- Warm start: `output/vc_suda/stage_c_r8b_lsj10_low_lr_continue_1024_teacher8499/checkpoint_iter_0000999.pth`
- Resume state: `runtime.resume: null`
- Depth noise: `data.depth_noise.enabled=true`, `type=gaussian`, `gaussian_std=0.0`
- LR: `solver.base_lr=1e-5`
- Max iter: `solver.max_iter=1000`
- Checkpoint period: `runtime.checkpoint_period=250`
- Checkpoint retention: `runtime.checkpoint_max_keep: null`
- Eval period: `runtime.eval_period=1000`

R10 keeps the R8B fixed-LSJ and VC-SUDA settings: `data.min_scale=1.0`, `data.max_scale=1.0`, `runtime.contrastive_enabled=true`, pseudo threshold `0.10`, curriculum threshold `0.10 -> 0.10`, `vc_suda.unsupervised_weight=0.02`, and `vc_suda.target_labeled_weight=1.0`.

## Gate

External 1024 backmap eval on `ckpt249` is the first decision point. Stop R10 unless target_unlabeled200 segm AP beats the R8B `ckpt999` baseline `0.319162`; only continue checkpoint sweeps if it is competitive with the current global best R8B `ckpt749` target_unlabeled200 segm AP `0.319922`.

## External Eval Plan

Run target_unlabeled200 external 1024 backmap eval on:

- `ckpt249`: early gate, must be `>0.319162` segm AP and should be close to or above `0.319922`.
- `ckpt499`, `ckpt749`, `ckpt999`: evaluate only if `ckpt249` passes the gate.

Built-in bbox-only eval remains diagnostic only and is not used for checkpoint selection.
