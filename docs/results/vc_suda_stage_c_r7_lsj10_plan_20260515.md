# VC-SUDA Stage C R7 Fixed-Scale Plan - 2026-05-15

## Conclusion

R7 is a single-variable fixed-LSJ-scale generalization experiment. It inherits the R3-A10 protocol and fixes only the LSJ scale to `1.0`, while keeping contrastive training enabled, pseudo threshold `0.10`, unsupervised weight `0.02`, target labeled weight `1.0`, and the Stage B iter8999 warm start.

## Config

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
