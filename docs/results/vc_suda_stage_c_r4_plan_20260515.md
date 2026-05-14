# VC-SUDA Stage C R4 Partial-Label Plan - 2026-05-15

## Conclusion

R4 partial-label is a single-code-factor rerun of R3-A10. The only intended variable is the partial-label pseudo loss semantics fixed in commit `dfbc09c4cce1d0c9a2564f165c0990d04bfeb463`.

## Baseline

Current best Stage C checkpoint is R3-A10 at iter1000:

- Config: `configs/vc_suda_stage_c_r3_a10_1024_teacher8499.yaml`
- Checkpoint: `output/vc_suda/stage_c_r3_a10_1024_teacher8499/checkpoint_iter_0000999.pth`
- Same-protocol full eval: `bbox_AP=0.341141`, `segm_AP=0.260302`
- Stage B same-protocol floor: `bbox_AP=0.332271`, `segm_AP=0.252354`

## R4 Config

R4 uses this config:

- `configs/vc_suda_stage_c_r4_partial_a10_1024_teacher8499.yaml`

It inherits all R3-A10 training parameters:

- Warm start: `model.finetune_weights: output/experiments/vc_suda_stage_b_1024_teacher8499_20260514_005821/checkpoint_iter_0008999.pth`
- Resume state: `runtime.resume: null`
- Max iter: `4000`
- Eval period: `1000`
- Checkpoint period: `1000`
- Pseudo threshold: `0.10`
- Curriculum thresholds: `start_threshold=0.10`, `end_threshold=0.10`
- Unsupervised weight: `0.02`
- Built-in eval: `runtime.eval_iou_types: [bbox]`
- Built-in eval subset: `runtime.eval_max_images: 28`
- Built-in eval batch size: `runtime.eval_batch_size: 4`
- GPUs: `[4, 5, 6, 7]`

Only these config fields differ from R3-A10:

- `name`
- `runtime.output_dir`
- `runtime.logger.log_dir`
- `runtime.logger.run_name`

## Hard Gate

At iter1000, run the same-protocol 1024 backmap full bbox+segm eval. Continue R4 only if both conditions hold:

- `segm_AP > 0.260302`
- `bbox_AP >= 0.332271`

The built-in 28-image bbox-only eval remains a fast training diagnostic. It is not the R4 gate.

## Stop Rules

- Do not start R4 from any Stage C checkpoint.
- Do not use `runtime.resume` for the R4 launch.
- Do not change threshold, unsupervised weight, max iter, eval settings, checkpoint period, or GPU list from R3-A10.
- Treat a missing, empty, or non-finite full eval result as a failed checkpoint.
