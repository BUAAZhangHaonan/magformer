# VC-SUDA R71 Balanced CE Off Diagnostic - 2026-05-17

## Conclusion

R71 passes the single-variable target-quality check.

The only semantic change from R46 was `model.magformer.mask_former.balanced_ce: true -> false`. Target_unlabeled200 improved from R46 segm AP/AP75 `0.323252/0.287485` to `0.331164/0.309062`. The target area signal moved smaller, not larger: bbox/mask area ratios are `0.793524/0.807673` by the image-level p50 convention.

Source first50 is segm AP `0.447108`, above the R46 source record `0.400942` but still far below the Teacher source sanity line. This remains a target-domain diagnostic because source was recorded but not the main gate.

## Config Diff Verification

Base config:

- `configs/vc_suda_stage_c_r46_pseudoreal_source_true_resume_1024_teacher8499.yaml`

R71 config:

- `configs/vc_suda_stage_c_r71_r46_balanced_ce_off_1024.yaml`

Resolved config comparison found exactly five changed leaves:

- `name`: R46 identity -> `vc_suda_stage_c_r71_r46_balanced_ce_off_1024`
- `runtime.output_dir`: R46 output -> `output/vc_suda/stage_c_r71_r46_balanced_ce_off_1024`
- `runtime.logger.log_dir`: R46 logs -> R71 logs
- `runtime.logger.run_name`: R46 identity -> R71 identity
- `model.magformer.mask_former.balanced_ce`: `true -> false`

Held fixed against R46:

- true resume: `output/vc_suda/stage_c_r12_32k_source_r8b_ckpt999_continue_1024_teacher8499/checkpoint_iter_0000499.pth`
- `solver.max_iter: 750`
- `solver.ims_per_batch: 4`
- `solver.base_lr: 1.0e-05`
- `runtime.eval_period/checkpoint_period: 250/250`
- `runtime.eval_max_images/eval_batch_size/eval_saves_best: 28/4/false`
- pseudo-real source and target split annotations
- `vc_suda.unsupervised_weight: 0.02`
- all sampling, L2-SP, multi-source, and loss weights unchanged

Dry-run resolved config evidence:

- `balanced_ce=false`
- `resume=...checkpoint_iter_0000499.pth`
- `max_iter=750`
- `source_root=magformer_datasets/pseudo_real_512`
- `source_ann=annotations/instances_source.json`
- `target_unlabeled_ann=annotations/instances_target_unlabeled.json`
- `gpus=[4,5,6,7]`

## Validation Before Training

Commands/checks run:

- `python tools/verify_vc_suda_stage.py --config configs/vc_suda_stage_c_r71_r46_balanced_ce_off_1024.yaml --stage C --emit-data-evidence --evidence-max-samples 1`
- `python -m pytest tests/test_vc_suda_train_entrypoint.py -q`: `15 passed`
- `python -m pytest tests/test_teacher_first50_protocol.py tests/test_check_eval_protocol.py -q`: `9 passed`
- `git diff --check`: passed
- Ruff changed Python files: skipped because the config milestone changed only YAML

One broader config test command was also tried:

- `python -m pytest tests/test_config_wiring.py tests/test_vc_suda_train_entrypoint.py tests/test_teacher_first50_protocol.py tests/test_check_eval_protocol.py -q`

That command failed only in `tests/test_config_wiring.py` because `configs/magformer_aligned_comparison.yaml` is missing in the current repo state. The R71-specific config load, preflight, train entrypoint, and protocol tests passed.

## Config Milestone Commit

- Commit: `da270374ce9775f457b525ea133154200f11a7d7`
- Message: `config: add r71 balanced ce off diagnostic`
- Direct server push timed out after 60 seconds with no output.
- Bundle bridge push succeeded from local bridge.
- `git ls-remote origin refs/heads/feature/vc-suda-sim2real` verified `da270374ce9775f457b525ea133154200f11a7d7`.

## Training

Command contract:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
MAGFORMER_MS_DEFORM_ATTN_BACKEND=cuda \
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
python -m torch.distributed.run --standalone --nproc_per_node=4 tools/train.py \
  --config configs/vc_suda_stage_c_r71_r46_balanced_ce_off_1024.yaml \
  --output-dir output/vc_suda/stage_c_r71_r46_balanced_ce_off_1024 \
  --gpus 0,1,2,3 \
  --num-workers 2
```

Run details:

- tmux session: `r71_balanced_ce_off`
- Final log: `output/logs/r71_balanced_ce_off_train_20260517_attempt2.log`
- Start: `2026-05-17T05:34:18+08:00`
- End: `2026-05-17T05:45:00+08:00`
- Exit: `EXIT_CODE=0`
- Wrapper time: `641` seconds
- Resume: iter `499` from R12 `checkpoint_iter_0000499.pth`
- Completion: `750/750`
- Aggregate training line near end: iter `740/750`, loss `12.5418`, `loss_ce=0.0032`, `loss_dice=0.0405`, `loss_mask=0.0014`
- Final eval in trainer: bbox AP/AP50/AP75 `0.1508/0.4382/0.0679` on the 28-image runtime eval
- Log scan found no Traceback, OOM, CUDA error, RuntimeError, non-finite loss, or shape mismatch. The only `inf` strings are expected `best_metric -inf` state text.
- Final memory sample after exit: `14904 MiB / 257582 MiB` used, below the 90% cap.

Checkpoints:

- `output/vc_suda/stage_c_r71_r46_balanced_ce_off_1024/checkpoint_iter_0000749.pth`
- `output/vc_suda/stage_c_r71_r46_balanced_ce_off_1024/checkpoint_iter_0000750.pth`

The formal external eval uses `checkpoint_iter_0000750.pth`.

## External Eval

Target protocol checker passed for `pseudo_real_target_unlabeled200`.

Source protocol checker passed for `original_first50_teacher` with `--allow-nondefault-weights` because this intentionally evaluates the R71 checkpoint under the fixed original first50 protocol.

| Split | Eval dir | bbox AP/AP50/AP75 | segm AP/AP50/AP75 |
|---|---|---:|---:|
| target_unlabeled200 | `output/diagnostics/r71_balanced_ce_off_target_unlabeled200_20260517` | `0.363805 / 0.708577 / 0.337151` | `0.331164 / 0.647341 / 0.309062` |
| original first50 | `output/diagnostics/r71_balanced_ce_off_original_first50_20260517` | `0.459816 / 0.745868 / 0.498579` | `0.447108 / 0.746673 / 0.474642` |

Eval artifacts:

- Target metrics: `output/diagnostics/r71_balanced_ce_off_target_unlabeled200_20260517/metrics.cocoeval.json`
- Target predictions: `output/diagnostics/r71_balanced_ce_off_target_unlabeled200_20260517/coco_instances_results.json`
- Target protocol summary: `output/diagnostics/r71_balanced_ce_off_target_unlabeled200_20260517/protocol_summary.json`
- Source metrics: `output/diagnostics/r71_balanced_ce_off_original_first50_20260517/metrics.cocoeval.json`
- Source predictions: `output/diagnostics/r71_balanced_ce_off_original_first50_20260517/coco_instances_results.json`
- Source protocol summary: `output/diagnostics/r71_balanced_ce_off_original_first50_20260517/protocol_summary.json`

## Target Area, Oracle, Tiny

Area ratio uses the R61/R65/R69 image-level p50 convention.

| Signal | R46 | R71 |
|---|---:|---:|
| Predictions | `13,758` | `14,440` |
| Pred count p50 | `63.0` | `66.0` |
| GT bbox area p50 over images | `779.75` | `779.75` |
| Pred bbox area p50 over images | `777.50` | `618.75` |
| Target bbox area ratio | `0.997114` | `0.793524` |
| GT mask area p50 over images | `501.75` | `501.75` |
| Pred mask area p50 over images | `513.75` | `405.25` |
| Target mask area ratio | `1.023916` | `0.807673` |

Oracle summary:

- Oracle output: `output/diagnostics/r71_balanced_ce_off_target_oracle_20260517/r53_prediction_union_oracle_summary.json`
- Mask oracle recall R@50/R@75/R@90: `0.685787 / 0.366298 / 0.042979`
- One-to-one oracle recall R@50/R@75: `0.685702 / 0.366298`
- Matched oracle-score segm AP/AP50/AP75: `0.372277 / 0.683168 / 0.366337`
- Dense `>90` oracle R@50/R@75: `0.511713 / 0.187621`
- Tiny `area <= 256` oracle R@50/R@75: `0.187457 / 0.015678`
- Tiny one-to-one matched@50/matched@75: `550 / 46`

Dense geometry audit:

- Audit JSON: `output/diagnostics/r71_balanced_ce_off_target_unlabeled200_20260517/r71_balanced_ce_off_target_unlabeled200_audit.json`
- Target mask TP/FP/FN: `8058 / 6382 / 3692`
- Target bbox TP/FP/FN: `8767 / 5673 / 2983`
- Matched TP bbox area ratio median/p90: `0.883117 / 1.096774`
- Matched TP mask area ratio median/p90: `0.975572 / 1.213654`

## R46 Comparison And Decision

| Metric | R46 | R71 | Result |
|---|---:|---:|---|
| target segm AP | `0.323252` | `0.331164` | improve |
| target segm AP75 | `0.287485` | `0.309062` | improve |
| target tiny oracle R@75 | `0.010566` | `0.015678` | improve |
| target dense `>90` oracle R@75 | `0.170213` | `0.187621` | improve |
| target matched oracle-score segm AP | `0.362376` | `0.372277` | improve |
| target mask area ratio | `1.023916` | `0.807673` | smaller |
| source first50 segm AP | `0.400942` | `0.447108` | improve vs R46 source record, still far below Teacher |

Decision: pass for the R71 target-domain single-variable diagnostic.

Balanced CE off does not cause mask enlargement here. It improves target AP, AP75, tiny R@75, dense R@75, and matched oracle-score AP while making the predicted mask/bbox p50 smaller than GT. The residual issue is not oversized masks; the remaining tiny recall is still low in absolute terms.
