# VC-SUDA Stage C R5 TL05 Plan - 2026-05-15

## Conclusion

R5-TL05 is a single-variable follow-up to R3-A10. It keeps the R3-A10 Stage C training protocol fixed and changes only `vc_suda.target_labeled_weight` from the old default `1.0` to `0.5`.

## Baseline

Current best Stage C checkpoint is R3-A10 at iter1000:

- Config: `configs/vc_suda_stage_c_r3_a10_1024_teacher8499.yaml`
- Checkpoint: `output/vc_suda/stage_c_r3_a10_1024_teacher8499/checkpoint_iter_0000999.pth`
- Same-protocol full eval: `bbox_AP=0.341141`, `segm_AP=0.260302`
- Stage B same-protocol floor: `bbox_AP=0.332271`, `segm_AP=0.252354`

## R5 Config

R5 uses this config:

- `configs/vc_suda_stage_c_r5_tl05_a10_1024_teacher8499.yaml`

It inherits the R3-A10 protocol:

- Warm start: `model.finetune_weights: output/experiments/vc_suda_stage_b_1024_teacher8499_20260514_005821/checkpoint_iter_0008999.pth`
- Resume state: `runtime.resume: null`
- Max iter: `4000`
- Eval period: `1000`
- Checkpoint period: `1000`
- Pseudo threshold: `0.10`
- Curriculum thresholds: `start_threshold=0.10`, `end_threshold=0.10`
- Unsupervised weight: `0.02`
- Target labeled weight: `0.5`
- Built-in eval: `runtime.eval_iou_types: [bbox]`
- Built-in eval subset: `runtime.eval_max_images: 28`
- Built-in eval batch size: `runtime.eval_batch_size: 4`
- GPUs: `[4, 5, 6, 7]`

Only these config fields should differ from R3-A10:

- `name`
- `runtime.output_dir`
- `runtime.logger.log_dir`
- `runtime.logger.run_name`
- `vc_suda.target_labeled_weight`

## Hard Gate

At iter1000, run the same-protocol 1024 backmap full bbox+segm eval. Continue R5 only if both conditions hold:

- `segm_AP > 0.260302`
- `bbox_AP >= 0.332271`

The built-in 28-image bbox-only eval remains a fast training diagnostic. It is not the R5 gate.

## Stop Rules

- Do not start R5 from any Stage C checkpoint.
- Do not use `runtime.resume` for the R5 launch.
- Do not change threshold, curriculum thresholds, unsupervised weight, max iter, eval settings, checkpoint period, or GPU list from R3-A10.
- Treat a missing, empty, or non-finite full eval result as a failed checkpoint.
