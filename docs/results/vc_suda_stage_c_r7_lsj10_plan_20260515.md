# VC-SUDA Stage C R7 Fixed-Scale Result - 2026-05-15

## Conclusion

R7 is the current best VC-SUDA Stage C external 1024 backmap run, but it is still far below the 61+ AP target. The best target_unlabeled200 result is `ckpt1999` with segm AP `0.3171`, which is `+0.0103` over the R3-A10 reference `0.3068`.

Fixing LSJ to `1.0/1.0` helped after the R6 no-contrast failure. It improved AP and reduced target_unlabeled200 prediction count from R3's `15421` to R7's `13017` at the best checkpoint.

## Setup

R7 is a single-variable fixed-LSJ-scale generalization experiment. It inherits the R3-A10 protocol and fixes only the LSJ scale to `1.0`, while keeping contrastive training enabled, pseudo threshold `0.10`, unsupervised weight `0.02`, target labeled weight `1.0`, and the Stage B iter8999 warm start.

## Config

- R7 config commit: `d8e69791a56923f7af3918f25ab26fcb3047f747`
- Config: `configs/vc_suda_stage_c_r7_a10_lsj10_1024_teacher8499.yaml`
- Warm start: `output/experiments/vc_suda_stage_b_1024_teacher8499_20260514_005821/checkpoint_iter_0008999.pth`
- Resume state: `runtime.resume: null`
- LSJ scale: `data.min_scale=1.0`, `data.max_scale=1.0`
- Max iter: `2000`
- Eval period: `1000`
- Checkpoint period: `500`
- Pseudo threshold: `0.10`
- Curriculum thresholds: `start_threshold=0.10`, `end_threshold=0.10`
- Unsupervised weight: `0.02`
- Target labeled weight: `1.0`
- Contrastive runtime switch: `true`
- Built-in eval: `runtime.eval_iou_types: [bbox]`

## Selection Rule

The built-in bbox-only eval is diagnostic only. R7 checkpoint selection must use external same-protocol 1024 backmap eval on numbered checkpoints, with full bbox+segm metrics.

## Early Stop Gate

At `ckpt999`, external 1024 backmap target_unlabeled200 segm AP must beat the R3-A10 reference `0.3068`. A result at or above `0.312` is more meaningful because it clears a small practical margin over R3.

## External Eval Plan

Run external 1024 backmap eval on numbered checkpoints:

- `ckpt499`: smoke trend only.
- `ckpt999`: early-stop gate on target_unlabeled200 segm AP.
- `ckpt1499` and `ckpt1999`: continue only if `ckpt999` passes the gate.

## External 1024 Backmap Results

### target_unlabeled200

| Checkpoint | bbox AP | bbox AP50 | bbox AP75 | segm AP | segm AP50 | segm AP75 | Predictions |
|---|---:|---:|---:|---:|---:|---:|---:|
| ckpt499 | 0.3836 | 0.7253 | 0.3635 | 0.3031 | 0.6409 | 0.2531 | 13942 |
| ckpt999 | 0.3874 | 0.7325 | 0.3680 | 0.3105 | 0.6466 | 0.2705 | 14326 |
| ckpt1499 | 0.3904 | 0.7326 | 0.3706 | 0.3145 | 0.6414 | 0.2736 | 13308 |
| ckpt1999 | **0.3916** | **0.7342** | **0.3724** | **0.3171** | 0.6417 | **0.2761** | 13017 |

`ckpt999` passed the early-stop gate by beating R3-A10 target_unlabeled200 segm AP `0.3068`. `ckpt1999` is the best target_unlabeled200 checkpoint by both bbox AP and segm AP.

### val28

| Checkpoint | bbox AP | bbox AP50 | bbox AP75 | segm AP | segm AP50 | segm AP75 | Predictions |
|---|---:|---:|---:|---:|---:|---:|---:|
| ckpt999 | 0.3450 | **0.6949** | 0.3014 | 0.2694 | 0.6009 | 0.2017 | 2225 |
| ckpt1499 | 0.3486 | 0.6886 | 0.3136 | 0.2737 | 0.6036 | 0.2121 | 2062 |
| ckpt1999 | **0.3516** | 0.6899 | **0.3241** | **0.2771** | **0.6038** | **0.2198** | 2012 |

`ckpt1999` is also the best val28 checkpoint by bbox AP and segm AP.

## Readout

- Best checkpoint by external target_unlabeled200 segm AP: `ckpt1999`.
- Best target_unlabeled200 segm AP: `0.3171`, `+0.0103` over R3-A10 `0.3068`.
- R7 reduced target_unlabeled200 prediction count versus R3 at the best checkpoint: `15421 -> 13017`.
- R6 no-contrast failed, while R7 kept contrastive training and only fixed LSJ scale to `1.0/1.0`.
- R7 is the current best Stage C external 1024 backmap result, but it remains far below the 61+ target.
