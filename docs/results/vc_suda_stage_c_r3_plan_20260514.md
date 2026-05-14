# VC-SUDA Stage C R3-A10 Plan - 2026-05-14

## Conclusion

R3-A10 keeps the R2 setup and changes only the pseudo-label threshold from 0.20 to 0.10. The sweep gate shows 0.20 was too sparse and degraded, while 0.10 is the first passing gate.

## Threshold Sweep Gate

- `threshold=0.15`: `empty_ratio=0.055`, FAIL.
- `threshold=0.10`: `empty_ratio=0.05`, PASS.
- `threshold=0.10`: `keep_rate=0.337493`.

## Config

- Config: `configs/vc_suda_stage_c_r3_a10_1024_teacher8499.yaml`
- Warm start: `model.finetune_weights: output/experiments/vc_suda_stage_b_1024_teacher8499_20260514_005821/checkpoint_iter_0008999.pth`
- Resume state: `runtime.resume: null`
- Max iter: `4000`
- Eval period: `1000`
- Checkpoint period: `1000`
- Unsupervised weight: `0.02`
- Pseudo threshold: `0.10`
- Curriculum threshold: `start_threshold=0.10`, `end_threshold=0.10`
- Built-in eval: `runtime.eval_iou_types: [bbox]`
- Built-in eval subset: `runtime.eval_max_images: 28`
- Built-in eval batch size: `runtime.eval_batch_size: 4`
- GPUs: `[4, 5, 6, 7]`

## R3-A10 Gate

At iter1000, run the same-protocol 1024 backmap full bbox+segm eval. Stop R3-A10 if either metric is lower than the Stage B same-protocol floor:

- Stage B same-protocol `bbox_AP=0.3323`
- Stage B same-protocol `segm_AP=0.2524`

The built-in 28-image bbox-only eval remains a fast training diagnostic. It is not the R3-A10 gate.

## Hard Rules

- Do not start R3-A10 from any Stage C checkpoint.
- Do not use `runtime.resume` for the R3-A10 launch.
- Do not change variables other than the threshold, name, output paths, and logger run name from R2.
- Stop at iter1000 if same-protocol 1024 backmap full bbox+segm eval falls below the Stage B floor.
- Treat a missing, empty, or non-finite eval result as a failed checkpoint.

## References

- R2 plan: `docs/results/vc_suda_stage_c_r2_plan_20260514.md`
- R2 status: `docs/results/vc_suda_stage_c_r2_status_20260514.md`
- Stage B post-eval: `docs/results/vc_suda_stage_b_post_eval_20260514.md`
