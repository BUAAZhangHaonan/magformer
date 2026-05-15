# VC-SUDA Stage C R8B Low-LR Continuation Plan - 2026-05-15

## Conclusion

R8 score threshold sweep did not beat the R7 target_unlabeled200 reference. R8B is the next single-variable test: continue from R7 `ckpt1999` with a lower LR schedule only.

## R8 Score Threshold Sweep

All four score thresholds stayed at the same rounded AP and did not exceed R7 `ckpt1999` target_unlabeled200 segm AP `0.3171`. Raising the score threshold only reduced prediction count.

| Score threshold | bbox AP | segm AP | Predictions | Passes R7 `0.3171` |
|---:|---:|---:|---:|---|
| 0.075 | 0.3909 | 0.3171 | 12861 | no |
| 0.100 | 0.3909 | 0.3171 | 12751 | no |
| 0.125 | 0.3909 | 0.3171 | 12679 | no |
| 0.150 | 0.3909 | 0.3171 | 12613 | no |

## R8B Setup

- Config: `configs/vc_suda_stage_c_r8b_lsj10_low_lr_continue_1024_teacher8499.yaml`
- Output dir: `output/vc_suda/stage_c_r8b_lsj10_low_lr_continue_1024_teacher8499`
- Warm start: `output/vc_suda/stage_c_r7_a10_lsj10_1024_teacher8499/checkpoint_iter_0001999.pth`
- Resume state: `runtime.resume: null`
- LR: `solver.base_lr=1e-5`
- Max iter: `solver.max_iter=1000`
- Checkpoint period: `runtime.checkpoint_period=250`
- Eval period: `runtime.eval_period=1000`

R8B keeps the R7 fixed-LSJ and VC-SUDA settings: `data.min_scale=1.0`, `data.max_scale=1.0`, `runtime.contrastive_enabled=true`, pseudo threshold `0.10`, curriculum threshold `0.10 -> 0.10`, `vc_suda.unsupervised_weight=0.02`, and `vc_suda.target_labeled_weight=1.0`.

## Early Stop Gate

External 1024 backmap eval on `ckpt249` is the first decision point. Stop R8B unless target_unlabeled200 segm AP is at least the R7 reference `0.3171`.

## External Eval Plan

Run target_unlabeled200 external 1024 backmap eval on:

- `ckpt249`: early-stop gate, must be `>=0.3171` segm AP.
- `ckpt499`, `ckpt749`, `ckpt999`: continue only if `ckpt249` passes.

Built-in bbox-only eval remains diagnostic only and is not used for checkpoint selection.
