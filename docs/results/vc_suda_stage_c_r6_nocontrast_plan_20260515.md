# VC-SUDA Stage C R6 No-Contrast Plan - 2026-05-15

## Conclusion

R6 is a single-variable no-contrast generalization experiment. It inherits the R3-A10 protocol and disables only the runtime contrastive loss while keeping the Stage B 8999 warm start, threshold `0.10`, unsupervised weight `0.02`, and target labeled weight `1.0`.

External 1024 backmap early eval at `ckpt499` shows that removing contrast hurts Stage C generalization. R6 target_unlabeled200 segm AP is `0.2410`, which is `-0.0658` below the R3 target_unlabeled200 segm AP `0.3068`. This is below the stop threshold `0.02`, so the R6 no-contrast run was stopped and should not be continued.

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

## Early Eval Result

External 1024 backmap eval at `ckpt499`:

| Split | Predictions | bbox AP | bbox AP50 | bbox AP75 | segm AP | segm AP50 | segm AP75 |
|---|---:|---:|---:|---:|---:|---:|---:|
| target_unlabeled200 | 17056 | 0.3124 | 0.6578 | 0.2550 | 0.2410 | 0.5751 | 0.1587 |
| val28 | 2580 | 0.2747 | 0.6236 | 0.1986 | 0.1990 | 0.5272 | 0.1020 |

Decision:

- Stop R6 no-contrast after `ckpt499` early eval.
- Do not continue the run to iter1000 or iter2000.
- Keep output artifacts for traceability, but do not select R6 for Stage C.
