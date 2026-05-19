# VC-SUDA R142 Pseudo Warmup Budget Fix - 2026-05-19

## Scope

Task 4b fixed the R142 pseudo loss effective weight issue by adding an explicit
iteration-based unsupervised warmup budget. Historical configs keep the legacy
`unsupervised_warmup_epochs` behavior unless `vc_suda.unsupervised_warmup_iters`
is positive.

## Red Test Evidence

Command:

```bash
/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python -m pytest -q \
  tests/test_vc_suda_trainer_runtime.py::test_unsupervised_weight_uses_positive_iteration_warmup_budget \
  tests/test_vc_suda_trainer_runtime.py::test_unsupervised_weight_keeps_epoch_warmup_when_iteration_budget_disabled \
  tests/test_vc_suda_trainer_runtime.py::test_stage_c_train_step_logs_effective_unsupervised_weight \
  tests/test_vc_suda_train_entrypoint.py::test_stage_c_r142_config_uses_32254_total_25654_source_train_split_online_ema \
  tests/test_diagnose_vc_suda_signals.py::test_signal_diagnostic_unsupervised_weight_uses_iteration_budget
```

Expected failing state before the fix:

```text
4 failed, 1 passed
old iter 249 weight: 4.748318485951744e-05, expected 0.125
train losses missing key: unsupervised_weight
R142 config missing cfg.vc_suda.unsupervised_warmup_iters
diagnostic iter 499 weight: 0.00018993273943806977, expected 0.5
```

## Green Checks

Focused red/green selection:

```text
.....                                                                    [100%]
```

Broader targeted check:

```bash
/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python -m pytest -q \
  tests/test_vc_suda_trainer_runtime.py \
  tests/test_vc_suda_train_entrypoint.py::test_stage_c_r142_config_uses_32254_total_25654_source_train_split_online_ema \
  tests/test_diagnose_vc_suda_signals.py
```

```text
38 passed, 1 skipped
```

Full VC-SUDA entrypoint file:

```text
24 passed
```

## R142 Smoke

Tmux session:

```text
r142_pseudo_budget_fix_smoke_20260519
```

Output:

```text
output/vc_suda/r142_pseudo_budget_fix_smoke_10iter_20260519
```

Command used single visible GPU7:

```bash
CUDA_VISIBLE_DEVICES=7 \
/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python tools/train.py \
  --config output/vc_suda/r142_pseudo_budget_fix_smoke_10iter_20260519/config_smoke.yaml \
  --gpus 0 \
  --num-workers 1
```

Smoke config kept the R142 data/loss setup, changed `solver.ims_per_batch=1`,
`solver.max_iter=509`, `runtime.log_period=1`, and `runtime.eval_max_images=1`.
The run resumed from iter 499 and wrote 10 train rows, iters 499-508.

Loader/preflight evidence:

```text
[CocoRgbdDataset] Loaded 25654 images from train split
[CocoRgbdDataset] Loaded 150 images from train split
[CocoRgbdDataset] Loaded 75 images from train split
[SemiSupervisedDataset] Stage=C, source=25654, target_labeled=150, target_unlabeled=75, target_unlabeled_sequence=75, offline_pseudo=disabled
depth_sanity.should_abort=false
```

Key train metrics from `metrics_log.jsonl`:

| Metric | Min | Max | Mean | Last |
| --- | ---: | ---: | ---: | ---: |
| `train/unsupervised_weight` | 0.5 | 0.5 | 0.5 | 0.5 |
| `train/pseudo_total` | 1.34375 | 2.72070 | 2.05303 | 2.03516 |
| `train/pseudo_kept_count` | 13 | 63 | 30.8 | 37 |
| weighted pseudo / total | 1.3257% | 5.6866% | 3.3355% | 4.5171% |

This confirms the pseudo branch is no longer suppressed by the loader-length
epoch warmup near the resumed R142 iterations.
