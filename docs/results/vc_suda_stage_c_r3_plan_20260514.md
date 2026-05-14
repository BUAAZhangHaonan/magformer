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

## R3-A10 Iter1000 Full Eval Result

R3-A10 training was gracefully stopped with Ctrl-C after the iter1000 gate point, at about iter1049. Keep this checkpoint as the gate checkpoint:

- `output/vc_suda/stage_c_r3_a10_1024_teacher8499/checkpoint_iter_0000999.pth`

Full eval output directory:

- `output/experiments/vc_suda_stage_c_r3_a10_iter1000_full_eval_20260514`

Same-protocol 1024 backmap full eval metrics:

- bbox AP/AP50/AP75: `0.341141` / `0.700864` / `0.289313`
- segm AP/AP50/AP75: `0.260302` / `0.594588` / `0.180459`

Gate comparison:

- Stage B same-protocol floor: bbox AP `0.332271`, segm AP `0.252354`
- R3-A10 gain over Stage B: bbox `+0.008870`, segm `+0.007948`
- R2 iter1000 full eval: bbox AP `0.319161`, segm AP `0.243594`
- R3-A10 gain over R2 iter1000: bbox `+0.021981`, segm `+0.016709`

Conclusion: R3-A10 iter1000 full eval gate PASSED. R3-A10 is the current best pseudo_real val same-protocol candidate, but it is still far from the 61+ target. Do not claim the target is complete.

## R3-A10 Resume2000 Full Eval Result

R3-A10 was resumed from the iter1000 run to iter2000. This was a true resume run:

- `runtime.resume` was set.
- The launch log included `skip model.finetune_weights`.
- The run resumed from iter 999, epoch 3.

Generated checkpoint:

- `output/vc_suda/stage_c_r3_a10_resume2000_1024_teacher8499/checkpoint_iter_0001999.pth`

Full eval output directory:

- `output/experiments/vc_suda_stage_c_r3_a10_iter2000_full_eval_20260514`

Same-protocol 1024 backmap full eval metrics:

- bbox AP/AP50/AP75: `0.338920` / `0.702177` / `0.295085`
- segm AP/AP50/AP75: `0.257460` / `0.589447` / `0.183245`

Gate comparison:

- bbox vs Stage B floor `0.332271`: PASS.
- segm vs R3-A10 iter1000 current best `0.260302`: FAIL.

Conclusion: R3-A10 resume2000 full eval gate FAILED overall. The current best remains R3-A10 iter1000:

- `output/vc_suda/stage_c_r3_a10_1024_teacher8499/checkpoint_iter_0000999.pth`
- segm AP: `0.260302`

## Next Step Recommendation

Continue single-variable experiments only. The next recommended gate/smoke is `threshold=0.05` with `unsup_weight=0.02`, keeping the rest of R3-A10 fixed. Do not change both threshold and unsupervised weight in the same experiment.

## R3-C05 Plan

R3-C05 is a single-variable threshold relaxation experiment. It keeps all R3-A10 settings fixed, including `vc_suda.unsupervised_weight=0.02`, Stage B checkpoint warm start, `runtime.resume: null`, bbox-only built-in eval with `eval_max_images=28` and `eval_batch_size=4`, and GPUs `[4, 5, 6, 7]`. The only training variable changed from R3-A10 is the pseudo-label threshold and curriculum thresholds, from `0.10` to `0.05`.

Config:

- `configs/vc_suda_stage_c_r3_c05_1024_teacher8499.yaml`

Stage B teacher pseudo-label diagnostics on `target_unlabeled=200`:

- `threshold=0.05`: PASS, `empty_ratio=0.045`, `keep_rate=0.4752309438`, avg pseudo/image `43.985`.
- `threshold=0.10`: PASS, `empty_ratio=0.050`, `keep_rate=0.3370968613`, avg pseudo/image `31.200`.

Diagnostic artifacts:

- `/tmp/vc_suda_r3_c05_diag_t005_200.json`
- `/tmp/vc_suda_r3_c05_diag_t010_200.json`

Gate:

- At iter1000, run the same-protocol 1024 backmap full bbox+segm eval.
- Continue only if `segm_AP > 0.260302` and `bbox_AP >= 0.332271`.
- Stop R3-C05 if either gate condition fails.

## R3-B10-U001 Plan

R3-B10-U001 is a single-variable follow-up to R3-A10. It keeps the R3-A10 threshold and curriculum thresholds at `0.10`, keeps the Stage B checkpoint warm start, keeps `runtime.resume: null`, keeps bbox-only built-in eval with `eval_max_images=28` and `eval_batch_size=4`, and keeps GPUs `[4, 5, 6, 7]`.

The only training variable changed from R3-A10 is `vc_suda.unsupervised_weight`, reduced from `0.02` to `0.01`.

Config:

- `configs/vc_suda_stage_c_r3_b10_u001_1024_teacher8499.yaml`

Gate:

- At iter1000, run the same-protocol 1024 backmap full bbox+segm eval.
- Continue only if `segm_AP > 0.260302` and `bbox_AP >= 0.332271`.
- Stop R3-B10-U001 if either gate condition fails.

## R3-B10-U001 Iter1000 Full Eval Result

R3-B10-U001 changes only `vc_suda.unsupervised_weight` from `0.02` to `0.01`. The pseudo-label threshold remains `0.10`, and the rest follows R3-A10.

Training was gracefully stopped with Ctrl-C after the iter1000 gate point, with the final observed iteration at about iter1096. The full eval used this checkpoint:

- `checkpoint_iter_0000999.pth`

Full eval output directory:

- `output/experiments/vc_suda_stage_c_r3_b10_u001_iter1000_full_eval_20260514/`

Same-protocol 1024 backmap full eval metrics:

- bbox AP/AP50/AP75: `0.3243506347` / `0.6776710211` / `0.2778714832`
- segm AP/AP50/AP75: `0.2525289573` / `0.5815385929` / `0.1746253896`

Gate comparison:

- Stage B same-protocol floor: bbox AP `0.332271`, segm AP `0.252354`
- R3-A10 current best: bbox AP `0.341141`, segm AP `0.260302`
- R3-B10-U001 vs Stage B: bbox is lower, segm is only effectively tied.
- R3-B10-U001 vs R3-A10: both bbox and segm are lower.

Conclusion: R3-B10-U001 iter1000 full eval gate FAILED. Do not continue R3-B. Lowering `unsupervised_weight` to `0.01` is worse than R3-A10. The current best remains R3-A10 with `output/vc_suda/stage_c_r3_a10_1024_teacher8499/checkpoint_iter_0000999.pth`.

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
