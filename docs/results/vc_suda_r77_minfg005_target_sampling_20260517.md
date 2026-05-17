# VC-SUDA R77 min_fg 0.05 + target sampling, 2026-05-17

R77 passes by the AP75 rule, but it is not useful for the dense/tiny sampling goal.

Target segm AP is effectively flat versus R74, `0.335561` vs `0.336010` (`-0.000449`). Target segm AP75 improves slightly, `0.316798` vs `0.315963`, and global oracle R@75 improves slightly, `0.373447` vs `0.373191`. The dense `>90` and tiny `<=256` oracle R@75 signals do not improve, so R52/R73 repeat sampling does not stack cleanly on top of R74.

## Scope

Config:

- `configs/vc_suda_stage_c_r77_minfg005_target_sampling_1024.yaml`
- Copied from `configs/vc_suda_stage_c_r74_balanced_ce_minfg005_1024.yaml`
- Keeps `balanced_ce: true`
- Keeps `balanced_ce_min_fg_ratio: 0.05`
- Keeps source, resume, solver, loss weights, thresholds, and unsupervised settings unchanged
- Adds only R52/R73 `vc_suda.target_unlabeled_sampling` using `output/diagnostics/r52_target_unlabeled_sampling_20260516/target_sampling_stats.json`

## Config Diff Verification

Recorded output:

- `output/diagnostics/r77_minfg005_target_sampling_setup_20260517/config_diff_verification.txt`

Allowed semantic YAML diff relative to R74:

- `name`
- `runtime.output_dir`
- `runtime.logger.log_dir`
- `runtime.logger.run_name`
- `vc_suda.target_unlabeled_sampling`

Verifier result:

```text
changed: name
changed: runtime.logger.log_dir
changed: runtime.logger.run_name
changed: runtime.output_dir
added: vc_suda.target_unlabeled_sampling
balanced_ce=True
balanced_ce_min_fg_ratio=0.05
unexpected_count=0
semantic_change_ok=true
```

## Tests And Preflight

Artifacts:

- `output/diagnostics/r77_minfg005_target_sampling_setup_20260517/pytest_target_sampling_pass.txt`
- `output/diagnostics/r77_minfg005_target_sampling_setup_20260517/pytest_eval_protocol_pass.txt`
- `output/diagnostics/r77_minfg005_target_sampling_setup_20260517/verify_stage_c_pass.txt`
- `output/diagnostics/r77_minfg005_target_sampling_setup_20260517/stats_gt_leakage_scan_pass.txt`
- `output/diagnostics/r77_minfg005_target_sampling_setup_20260517/git_diff_check_untracked.txt`

Commands covered:

```bash
python -m pytest tests/test_build_target_unlabeled_sampling_stats.py   tests/test_vc_suda_data_protocol.py::test_target_unlabeled_samples_never_expose_ground_truth_labels   tests/test_vc_suda_data_protocol.py::test_target_unlabeled_repeat_sequence_controls_only_target_branch   tests/test_vc_suda_data_protocol.py::test_target_unlabeled_sampling_stats_reject_gt_or_annotation_fields   tests/test_vc_suda_data_protocol.py::test_target_collate_never_emits_unlabeled_label_fields   tests/test_vc_suda_train_entrypoint.py::test_vc_suda_build_datasets_passes_target_unlabeled_sampling_stats   tests/test_vc_suda_stage_c_preflight.py -q
python -m pytest tests/test_check_eval_protocol.py tests/test_teacher_first50_protocol.py -q
python tools/verify_vc_suda_stage.py --config configs/vc_suda_stage_c_r77_minfg005_target_sampling_1024.yaml --stage C --emit-data-evidence --evidence-max-samples 1
git diff --no-index --check /dev/null configs/vc_suda_stage_c_r77_minfg005_target_sampling_1024.yaml
```

Results:

- Target-sampling/preflight pytest: `34 passed`
- Eval-protocol pytest: `9 passed`
- Stage C verifier: `PASS`, true-resume from R12 iter `499`, `target_unlabeled_sequence=254`
- Stats leakage scan: `gt_leakage_scan=pass`, buckets normal/dense/dense_tiny `158/30/12`, repeat sum `254`
- `git diff --check`: clean
- `ruff`: skipped because no Python files changed

## Config Milestone Commit

- 4029 local commit: `ff5e87649fd3874eca891efd3d11055d98632dd0`
- Message: `config: add r77 target sampling diagnostic`
- Direct GitHub push from `4029` timed out with exit `124`
- GitHub current head differed from 4029 `origin/feature/vc-suda-sim2real`, so the config commit was cherry-picked by bundle bridge onto GitHub current head
- GitHub verified commit: `a12d170c294e56323d275c16a8a9e4300c289843`
- `git ls-remote origin refs/heads/feature/vc-suda-sim2real` verified `a12d170c294e56323d275c16a8a9e4300c289843`

## Training

Training ran in tmux on GPUs 4-7.

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 MAGFORMER_MS_DEFORM_ATTN_BACKEND=cuda OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 python -m torch.distributed.run --standalone --nproc_per_node=4 tools/train.py   --config configs/vc_suda_stage_c_r77_minfg005_target_sampling_1024.yaml   --output-dir output/vc_suda/stage_c_r77_minfg005_target_sampling_1024   --gpus 0,1,2,3   --num-workers 2
```

Run details:

- tmux session: `r77_minfg005_target_sampling`
- Train log: `output/logs/r77_minfg005_target_sampling_train_20260517.log`
- Resource log: `output/logs/r77_minfg005_target_sampling_resource_20260517.log`
- Start: `2026-05-17T08:23:04+08:00`
- End: `2026-05-17T08:36:09+08:00`
- Exit: `EXIT_CODE=0`
- Resume: R12 `checkpoint_iter_0000499.pth`, restored EMA teacher state, resumed from iter `499`
- Dataset line: `source=1008`, `target_labeled=25`, `target_unlabeled=200`, `target_unlabeled_sequence=254`
- Completion: iter `750/750`, final eval ran
- Final checkpoint: `output/vc_suda/stage_c_r77_minfg005_target_sampling_1024/checkpoint_iter_0000750.pth`
- Final aggregate line near end: iter `740/750`, loss `14.0991`, `loss_ce=0.0033`, `loss_dice=0.0399`, `loss_mask=0.0104`
- Failure scan: no Traceback, OOM, RuntimeError, non-finite, or shape mismatch
- Peak sampled RAM: `12.28%` of `257582 MB`
- Peak sampled GPU memory: GPU4 `22937`, GPU5 `22649`, GPU6 `22627`, GPU7 `21915` MiB

## External Eval

Protocol checkers passed:

- Target: `output/diagnostics/r77_minfg005_target_sampling_target_unlabeled200_20260517/protocol_summary.json`
- Source first50: `output/diagnostics/r77_minfg005_target_sampling_original_first50_20260517/protocol_summary.json`

| Split | Eval dir | bbox AP/AP50/AP75 | segm AP/AP50/AP75 |
|---|---|---:|---:|
| target_unlabeled200 | `output/diagnostics/r77_minfg005_target_sampling_target_unlabeled200_20260517` | `0.387835 / 0.729294 / 0.374191` | `0.335561 / 0.650480 / 0.316798` |
| original first50 | `output/diagnostics/r77_minfg005_target_sampling_original_first50_20260517` | `0.473208 / 0.766182 / 0.512562` | `0.436693 / 0.746921 / 0.451885` |

## Target Area, Oracle, Tiny/Dense

Area ratio uses the image-level p50 convention.

| Signal | R74 | R77 |
|---|---:|---:|
| Predictions | `14,181` | `14,155` |
| Pred count p50 | `65.0` | `65.0` |
| Target bbox area ratio | `0.863738` | `0.861815` |
| Target mask area ratio | `0.892377` | `0.893871` |

Oracle:

- Output: `output/diagnostics/r77_minfg005_target_sampling_target_oracle_20260517/r77_oracle_bound_summary.json`
- Mask oracle R@50/R@75/R@90: `0.689787 / 0.373447 / 0.046894`
- One-to-one oracle R@50/R@75: `0.689702 / 0.373447`
- Matched oracle-score segm AP/AP50/AP75: `0.375248 / 0.683168 / 0.376238`
- Dense `>90` oracle R@50/R@75: `0.510638 / 0.191274`
- Tiny `area <= 256` oracle R@50/R@75: `0.180300 / 0.016701`

Dense geometry audit:

- Audit JSON: `output/diagnostics/r77_minfg005_target_sampling_target_unlabeled200_20260517/r77_minfg005_target_sampling_target_unlabeled200_audit.json`
- Target mask TP/FP/FN: `8105 / 6050 / 3645`
- Target bbox TP/FP/FN: `8941 / 5214 / 2809`

## Comparison And Decision

| Signal | R74 | R77 | Direction |
|---|---:|---:|---|
| target bbox AP | `0.387567` | `0.387835` | `+0.000268` |
| target bbox AP75 | `0.373927` | `0.374191` | `+0.000264` |
| target segm AP | `0.336010` | `0.335561` | `-0.000449` |
| target segm AP75 | `0.315963` | `0.316798` | `+0.000835` |
| target oracle R@75 | `0.373191` | `0.373447` | `+0.000256` |
| dense `>90` oracle R@75 | `0.193853` | `0.191274` | worse |
| tiny `<=256` oracle R@75 | `0.016701` | `0.016701` | flat/slightly lower |
| target bbox area ratio | `0.863738` | `0.861815` | smaller |
| target mask area ratio | `0.892377` | `0.893871` | slightly larger |

Decision: pass by the AP75 rule, but not useful for the original dense/tiny repeat-sampling hypothesis.

R77 does not beat R74 target segm AP, but the AP drop is under `0.001` and target AP75 improves. The intended dense/tiny readout does not improve: dense R@75 falls and tiny R@75 is effectively unchanged. Keep R74 as the best primary config; R77 is a narrow AP75/overall-oracle confirmation, not evidence that target repeat sampling compounds with `balanced_ce_min_fg_ratio=0.05`.
