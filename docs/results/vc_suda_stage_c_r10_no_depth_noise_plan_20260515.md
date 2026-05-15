# VC-SUDA Stage C R10 No-Depth-Noise Continuation Plan - 2026-05-15

## Conclusion

R10 was stopped after the `ckpt249` external target_unlabeled200 eval failed the early gate. It continued from R8B `ckpt999` and removed only the depth Gaussian perturbation by setting `data.depth_noise.gaussian_std: 0.0`, but this did not improve early target_unlabeled200 segm AP.

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

## Stop Result

R10 session `vc_suda_stage_c_r10_nodepth_20260515` was stopped around iter 300 after `ckpt249` failed the early gate. No output was deleted.

External 1024 backmap target_unlabeled200 eval on `ckpt249`:

| Checkpoint | bbox AP | bbox AP50 | bbox AP75 | segm AP | segm AP50 | segm AP75 | Pred count |
|---|---:|---:|---:|---:|---:|---:|---:|
| `ckpt249` | 0.3903716504 | 0.7271236372 | 0.3731434519 | 0.3164998465 | 0.6427576211 | 0.2760591309 | 12772 |

Decision:

- R10 `ckpt249` segm AP `0.3164998465` is below R8B `ckpt999` segm AP `0.3191621968` by `-0.0026623503`.
- R10 `ckpt249` segm AP is below global best R8B `ckpt749` segm AP `0.319922` by `-0.0034221535`.
- Disabling Gaussian depth noise did not improve early AP, so R10 should not continue to `ckpt499/749/999`.
- No val28 run was done.

## External Eval Plan

Run target_unlabeled200 external 1024 backmap eval on:

- `ckpt249`: early gate, must be `>0.319162` segm AP and should be close to or above `0.319922`.
- `ckpt499`, `ckpt749`, `ckpt999`: evaluate only if `ckpt249` passes the gate.

Built-in bbox-only eval remains diagnostic only and is not used for checkpoint selection.
