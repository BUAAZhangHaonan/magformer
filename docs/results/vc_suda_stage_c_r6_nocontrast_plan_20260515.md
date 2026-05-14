# VC-SUDA Stage C R6 No-Contrast Plan - 2026-05-15

## Conclusion

R6 is a single-variable no-contrast generalization experiment. It inherits the R3-A10 protocol and disables only the runtime contrastive loss while keeping the Stage B 8999 warm start, threshold `0.10`, unsupervised weight `0.02`, and target labeled weight `1.0`.

## Config

- Config: `configs/vc_suda_stage_c_r6_a10_nocontrast_1024_teacher8499.yaml`
- Warm start: `output/experiments/vc_suda_stage_b_1024_teacher8499_20260514_005821/checkpoint_iter_0008999.pth`
- Resume state: `runtime.resume: null`
- Max iter: `2000`
- Eval period: `1000`
- Checkpoint period: `500`
- Pseudo threshold: `0.10`
- Curriculum thresholds: `start_threshold=0.10`, `end_threshold=0.10`
- Unsupervised weight: `0.02`
- Unsupervised warmup epochs: `10`
- Target labeled weight: `1.0`
- Contrastive runtime switch: `false`
- Built-in eval: `runtime.eval_iou_types: [bbox]`

## Selection Rule

The built-in bbox-only eval is diagnostic only. It must not select the best checkpoint. R6 checkpoint selection must use external same-protocol 1024 backmap eval on numbered checkpoints, with full bbox+segm metrics.

## Gate

Evaluate numbered checkpoints externally at least at iter1000 and iter2000. Continue to treat R3-A10 iter1000 as the current Stage C reference unless R6 improves the external 1024 backmap result.
