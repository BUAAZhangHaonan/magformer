# VC-SUDA R73 balanced_ce=false + target sampling, 2026-05-17

R73 passes the single-variable diagnostic.

Target segm AP is essentially flat versus R71, `0.330836` vs `0.331164`, so it does not beat the main AP line. The rescue signal is tiny/dense oracle quality: tiny `<=256` R@75 improves `0.015678 -> 0.018064`, dense `>90` R@75 improves `0.187621 -> 0.191274`, and global oracle R@75 improves `0.366298 -> 0.368681`. The AP drop is only `-0.000328`, so this is not a segm AP collapse.

## Scope

R73 copies R71 and adds only the R52 prediction-only `target_unlabeled_sampling` repeat stats.

Config:

- `configs/vc_suda_stage_c_r73_balanced_ce_off_target_sampling_1024.yaml`
- Based on `configs/vc_suda_stage_c_r71_r46_balanced_ce_off_1024.yaml`
- Keeps `balanced_ce: false`
- Keeps source, resume, solver, loss weights, thresholds, target_labeled weight, and unsupervised weight unchanged
- Uses R52 stats: `output/diagnostics/r52_target_unlabeled_sampling_20260516/target_sampling_stats.json`

## Config Diff Verification

Recorded output:

- `output/diagnostics/r73_balanced_ce_off_target_sampling_setup_20260517/config_diff_verification.txt`

Semantic YAML diff allowed only these paths:

- `name`
- `runtime.output_dir`
- `runtime.logger.log_dir`
- `runtime.logger.run_name`
- `vc_suda.target_unlabeled_sampling`

The verifier reported:

```text
unexpected_count=0
semantic_change_ok=true
```

The grep evidence confirms `balanced_ce: false`, `base_lr: 1.0e-05`, the R12 ckpt499 resume path, and `unsupervised_weight: 0.02` stayed fixed.

## Tests And Preflight

Clean verification files:

- `output/diagnostics/r73_balanced_ce_off_target_sampling_setup_20260517/pytest_target_sampling_pass.txt`
- `output/diagnostics/r73_balanced_ce_off_target_sampling_setup_20260517/verify_stage_c_pass.txt`
- `output/diagnostics/r73_balanced_ce_off_target_sampling_setup_20260517/stats_gt_leakage_scan_pass.txt`

Commands covered:

```bash
python -m pytest   tests/test_build_target_unlabeled_sampling_stats.py   tests/test_vc_suda_data_protocol.py::test_target_unlabeled_samples_never_expose_ground_truth_labels   tests/test_vc_suda_data_protocol.py::test_target_unlabeled_repeat_sequence_controls_only_target_branch   tests/test_vc_suda_data_protocol.py::test_target_unlabeled_sampling_stats_reject_gt_or_annotation_fields   tests/test_vc_suda_data_protocol.py::test_target_collate_never_emits_unlabeled_label_fields   tests/test_vc_suda_train_entrypoint.py::test_vc_suda_build_datasets_passes_target_unlabeled_sampling_stats   tests/test_vc_suda_stage_c_preflight.py -q

python tools/verify_vc_suda_stage.py   --config configs/vc_suda_stage_c_r73_balanced_ce_off_target_sampling_1024.yaml   --stage C   --emit-data-evidence   --evidence-max-samples 1
```

Results:

- Pytest: `34 passed`
- Stage verifier: `PASS`, including true-resume semantics and `target_unlabeled_sequence=254`
- Stats leakage scan: `gt_leakage_scan=pass`
- Sampling stats buckets: normal `158`, dense `30`, dense_tiny `12`, repeat sum `254`

## Config Milestone Commit

- Commit: `85dd358d26b575b6599c5c164fc67bbb0f1302af`
- Message: `config: add r73 target sampling diagnostic`
- Direct GitHub push from `4029` timed out with exit `124`
- Bundle bridge push succeeded from local bridge
- `git ls-remote` verified `85dd358d26b575b6599c5c164fc67bbb0f1302af refs/heads/feature/vc-suda-sim2real`

## Training

Training ran in tmux on GPUs 4-7.

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 MAGFORMER_MS_DEFORM_ATTN_BACKEND=cuda OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 python -m torch.distributed.run --standalone --nproc_per_node=4 tools/train.py   --config configs/vc_suda_stage_c_r73_balanced_ce_off_target_sampling_1024.yaml   --output-dir output/vc_suda/stage_c_r73_balanced_ce_off_target_sampling_1024   --gpus 0,1,2,3   --num-workers 2
```

Run details:

- tmux session: `r73_balanced_ce_off_target_sampling`
- First wrapper attempt: `output/logs/r73_balanced_ce_off_target_sampling_train_20260517.log`, exit `127`, caused only by a bad `/usr/bin/time -f` quote; training did not start
- Final training log: `output/logs/r73_balanced_ce_off_target_sampling_train_20260517_attempt2.log`
- Start: `2026-05-17T06:20:27+08:00`
- End: `2026-05-17T06:32:23+08:00`
- Exit: `EXIT_CODE=0`
- Resume: R12 iter `499`, `checkpoint_iter_0000499.pth`
- Dataset line: `source=1008`, `target_labeled=25`, `target_unlabeled=200`, `target_unlabeled_sequence=254`
- Completion: `750/750`
- Final aggregate line near end: iter `740/750`, loss `12.4200`, `loss_ce=0.0052`, `loss_dice=0.0383`, `loss_mask=0.0014`
- Log scan found `0` Traceback/OOM/CUDA OOM/RuntimeError/non-finite/shape mismatch
- Memory stayed below the 90% cap; sampled training RAM was about `30.6GB / 257.6GB`
- Sampled GPU memory during training was about `22GB` on GPUs 4-7

Checkpoints:

- `output/vc_suda/stage_c_r73_balanced_ce_off_target_sampling_1024/checkpoint_iter_0000749.pth`
- `output/vc_suda/stage_c_r73_balanced_ce_off_target_sampling_1024/checkpoint_iter_0000750.pth`

The formal external eval uses `checkpoint_iter_0000750.pth`.

## External Eval

Target protocol checker passed for `pseudo_real_target_unlabeled200`.

Source protocol checker passed for `original_first50_teacher` with `--allow-nondefault-weights`, because this intentionally evaluates the R73 checkpoint under the fixed original first50 protocol.

| Split | Eval dir | bbox AP/AP50/AP75 | segm AP/AP50/AP75 |
|---|---|---:|---:|
| target_unlabeled200 | `output/diagnostics/r73_balanced_ce_off_target_sampling_target_unlabeled200_20260517` | `0.364197 / 0.715224 / 0.337237` | `0.330836 / 0.647090 / 0.308940` |
| original first50 | `output/diagnostics/r73_balanced_ce_off_target_sampling_original_first50_20260517` | `0.460812 / 0.753760 / 0.494333` | `0.444746 / 0.745866 / 0.473542` |

Eval artifacts:

- Target metrics: `output/diagnostics/r73_balanced_ce_off_target_sampling_target_unlabeled200_20260517/metrics.cocoeval.json`
- Target predictions: `output/diagnostics/r73_balanced_ce_off_target_sampling_target_unlabeled200_20260517/coco_instances_results.json`
- Target protocol summary: `output/diagnostics/r73_balanced_ce_off_target_sampling_target_unlabeled200_20260517/protocol_summary.json`
- Source metrics: `output/diagnostics/r73_balanced_ce_off_target_sampling_original_first50_20260517/metrics.cocoeval.json`
- Source predictions: `output/diagnostics/r73_balanced_ce_off_target_sampling_original_first50_20260517/coco_instances_results.json`
- Source protocol summary: `output/diagnostics/r73_balanced_ce_off_target_sampling_original_first50_20260517/protocol_summary.json`

## Target Area, Oracle, Tiny

Area ratio uses the R61/R65/R69 image-level p50 convention.

| Signal | R71 | R73 |
|---|---:|---:|
| Predictions | `14,440` | `14,478` |
| Pred count p50 | `66.0` | `67.0` |
| GT bbox area p50 over images | `779.75` | `779.75` |
| Pred bbox area p50 over images | `618.75` | `613.25` |
| Target bbox area ratio | `0.793524` | `0.786470` |
| GT mask area p50 over images | `501.75` | `501.75` |
| Pred mask area p50 over images | `405.25` | `400.75` |
| Target mask area ratio | `0.807673` | `0.798705` |

Oracle summary:

- Oracle output: `output/diagnostics/r73_balanced_ce_off_target_sampling_target_oracle_20260517/r53_prediction_union_oracle_summary.json`
- Mask oracle recall R@50/R@75/R@90: `0.685702 / 0.368681 / 0.043660`
- One-to-one oracle recall R@50/R@75: `0.685532 / 0.368681`
- Matched oracle-score segm AP/AP50/AP75: `0.372277 / 0.683168 / 0.366337`
- Dense `>90` oracle R@50/R@75: `0.511283 / 0.191274`
- Tiny `area <= 256` oracle R@50/R@75: `0.192229 / 0.018064`

Dense geometry audit:

- Audit JSON: `output/diagnostics/r73_balanced_ce_off_target_sampling_target_unlabeled200_20260517/r73_balanced_ce_off_target_sampling_target_unlabeled200_audit.json`
- Target mask TP/FP/FN: `8057 / 6421 / 3693`
- Target bbox TP/FP/FN: `8797 / 5681 / 2953`
- Matched TP bbox area ratio median/p90: `0.885057 / 1.098651`
- Matched TP mask area ratio median/p90: `0.975845 / 1.216373`

## Comparison And Decision

| Signal | R71 | R73 | Direction |
|---|---:|---:|---|
| target segm AP | `0.331164` | `0.330836` | `-0.000328` |
| target segm AP75 | `0.309062` | `0.308940` | `-0.000122` |
| target bbox AP | `0.363805` | `0.364197` | small improve |
| target tiny oracle R@75 | `0.015678` | `0.018064` | improve |
| target dense `>90` oracle R@75 | `0.187621` | `0.191274` | improve |
| target global oracle R@75 | `0.366298` | `0.368681` | improve |
| target mask area ratio | `0.807673` | `0.798705` | smaller |
| source first50 segm AP | `0.447108` | `0.444746` | secondary drop |

Decision: pass.

R73 does not beat R71 target segm AP, but the AP loss is tiny and AP75 is effectively flat. The intended tiny/dense readout improves, especially tiny R@75 and dense R@75, without AP collapse. This supports keeping target_unlabeled dense/tiny repeat sampling as a useful single-variable signal on top of `balanced_ce=false`, while still treating the formal AP gain as not established.
