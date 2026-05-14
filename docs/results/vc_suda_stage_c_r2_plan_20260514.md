# VC-SUDA Stage C-R2 Plan - 2026-05-14

## Conclusion

Stage C-R2 must restart from the Stage B checkpoint, not from the degraded Stage C run. The original Stage C final eval regressed below the Stage B reference, so R2 does not resume any Stage C state. The training-time eval should be bbox-only on the pseudo-real validation subset, and full bbox+segm eval is reserved for final checkpoints.

## Config

- Config: `configs/vc_suda_stage_c_r2_1024_teacher8499.yaml`
- Warm start: `model.finetune_weights: output/experiments/vc_suda_stage_b_1024_teacher8499_20260514_005821/checkpoint_iter_0008999.pth`
- Resume state: `runtime.resume: null`
- Max iter: `4000`
- Eval period: `1000`
- Checkpoint period: `1000`
- Unsupervised weight: `0.02`
- Pseudo threshold: `0.20`
- Built-in eval: `runtime.eval_iou_types: [bbox]`
- Built-in eval subset: `runtime.eval_max_images: 28`
- Built-in eval batch size: `runtime.eval_batch_size: 4`

## Hard Rules

- Do not start R2 from any Stage C checkpoint.
- Do not use `runtime.resume` for the R2 launch.
- Do not use bbox+segm COCO eval every 1000 iterations during R2 training.
- Do not treat the bbox-only subset metric as the final metric.
- For daily diagnosis outside the 28-image pseudo-real val split, use bbox-only eval with `max_images` 200 or 300 and `eval_batch_size` 4 or 8. Run segmentation metrics only for final candidate checkpoints.
- Subset eval is restricted with `COCOeval.params.imgIds`, so non-evaluated images are not counted as misses and cannot create false low AP.

## Eval Policy

Every 1000 iterations, use the built-in bbox-only pseudo-real val eval to check direction. This eval covers the 28-image pseudo-real validation split and should be fast enough for routine diagnostics.

Final selection needs a separate full bbox+segm eval on the final candidate checkpoint. Use the same formal eval protocol as Stage B post-eval so Stage B and R2 are comparable.

## Stop Rules

- Stop and roll back the Stage C plan if the bbox-only subset metric is consecutively below the Stage B reference trend at the 1000-iteration checkpoints.
- Stop and roll back the Stage C plan if the final full `segm_AP` is below the Stage B post-eval `segm_AP=0.1128`.
- Treat a missing, empty, or non-finite eval result as a failed checkpoint, not as a result to smooth or repair.

## References

- Stage B post-eval: `docs/results/vc_suda_stage_b_post_eval_20260514.md`
- Stage C final degradation: `docs/results/vc_suda_stage_c_final_eval_20260514.md`
- Stage C mid eval: `docs/results/vc_suda_stage_c_mid_eval_20260514.md`
