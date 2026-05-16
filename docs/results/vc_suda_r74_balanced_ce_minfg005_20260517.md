# VC-SUDA R74 balanced_ce min_fg 0.05, 2026-05-17

R74 passes.

It is the requested single-variable test on top of R46: keep `balanced_ce: true`, raise `balanced_ce_min_fg_ratio` from `0.01` to `0.05`, and leave resume/source/solver/loss/target/sampling settings unchanged. Target segm AP/AP75 improves over both R46 and R71: `0.336010 / 0.315963`. Bbox AP also recovers relative to R71: `0.387567` vs `0.363805`.

## Scope

Config:

- `configs/vc_suda_stage_c_r74_balanced_ce_minfg005_1024.yaml`
- Based on `configs/vc_suda_stage_c_r46_pseudoreal_source_true_resume_1024_teacher8499.yaml`
- Keeps `model.magformer.mask_former.balanced_ce: true`
- Changes `model.magformer.mask_former.balanced_ce_min_fg_ratio: 0.01 -> 0.05`
- No target sampling
- No source retention
- Same R46 true resume: `output/vc_suda/stage_c_r12_32k_source_r8b_ckpt999_continue_1024_teacher8499/checkpoint_iter_0000499.pth`

## Config Diff Verification

Recorded output:

- `output/diagnostics/r74_balanced_ce_minfg005_setup_20260517/config_diff_verification.txt`

Allowed identity differences:

- `name`
- `runtime.output_dir`
- `runtime.logger.log_dir`
- `runtime.logger.run_name`

Only semantic difference:

- `model.magformer.mask_former.balanced_ce_min_fg_ratio: 0.01 -> 0.05`

Verifier result:

```text
unexpected_count=0
balanced_ce=True
balanced_ce_min_fg_ratio=0.05
semantic_change_ok=true
```

## Tests And Preflight

Commands:

```bash
python -m pytest tests/test_vc_suda_train_entrypoint.py tests/test_check_eval_protocol.py tests/test_teacher_first50_protocol.py tests/test_vc_suda_stage_c_preflight.py -q
python tools/verify_vc_suda_stage.py --config configs/vc_suda_stage_c_r74_balanced_ce_minfg005_1024.yaml --stage C --emit-data-evidence --evidence-max-samples 1
git diff --check -- configs/vc_suda_stage_c_r74_balanced_ce_minfg005_1024.yaml
```

Results:

- Pytest: `51 passed`
- Stage verifier: `PASS`, including true-resume semantics from R12 iter `499`
- `git diff --check`: clean
- `ruff`: skipped because no Python files changed

Config milestone:

- Commit: `68851863ac93645261d854d079b3a14d929c2c17`
- Message: `config: add r74 balanced ce min fg diagnostic`
- Direct remote push from `4029` timed out with exit `124`
- Bundle bridge push succeeded
- `git ls-remote origin refs/heads/feature/vc-suda-sim2real` verified `68851863ac93645261d854d079b3a14d929c2c17`

## Training

Training ran in tmux on GPUs 4-7.

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 MAGFORMER_MS_DEFORM_ATTN_BACKEND=cuda OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 python -m torch.distributed.run --standalone --nproc_per_node=4 tools/train.py \
  --config configs/vc_suda_stage_c_r74_balanced_ce_minfg005_1024.yaml \
  --output-dir output/vc_suda/stage_c_r74_balanced_ce_minfg005_1024 \
  --gpus 0,1,2,3 \
  --num-workers 2
```

Run details:

- tmux session: `r74_balanced_ce_minfg005`
- Captured training log: `output/logs/r74_balanced_ce_minfg005_train_20260517.log`
- Resource log: `output/logs/r74_balanced_ce_minfg005_resource_20260517.log`
- Start line: `2026-05-17 06:52:53`, resumed from iter `499`
- Completion line: `2026-05-17 07:07:27`, training completed and final eval ran at iter `750`
- Wall time from wrapper: `15:15.08`
- Final checkpoint: `output/vc_suda/stage_c_r74_balanced_ce_minfg005_1024/checkpoint_iter_0000750.pth`
- Log scan: `Traceback=0`, `CUDA out of memory=0`, `RuntimeError=0`, `non-finite=0`, `shape mismatch=0`
- Peak sampled system RAM: `31,299 MB / 257,582 MB` = `12.15%`
- Peak sampled GPU memory: GPU4 `22927`, GPU5 `22615`, GPU6 `22627`, GPU7 `22167` MiB

Note: the wrapper had a logging-variable expansion bug after the training command finished, so its final `EXIT_CODE` field is blank. The training process itself completed, final eval ran, and `checkpoint_iter_0000750.pth` exists.

## External Eval

Protocol checkers passed:

- Target: `output/diagnostics/r74_balanced_ce_minfg005_target_unlabeled200_20260517/protocol_summary.json`
- Source first50: `output/diagnostics/r74_balanced_ce_minfg005_original_first50_20260517/protocol_summary.json`

| Split | Eval dir | bbox AP/AP50/AP75 | segm AP/AP50/AP75 |
|---|---|---:|---:|
| target_unlabeled200 | `output/diagnostics/r74_balanced_ce_minfg005_target_unlabeled200_20260517` | `0.387567 / 0.728979 / 0.373927` | `0.336010 / 0.656434 / 0.315963` |
| original first50 | `output/diagnostics/r74_balanced_ce_minfg005_original_first50_20260517` | `0.471960 / 0.765091 / 0.506017` | `0.437474 / 0.746447 / 0.452494` |

## Target Area, Oracle, Tiny/Dense

Area ratio uses the image-level p50 convention.

| Signal | R71 | R73 | R74 |
|---|---:|---:|---:|
| Predictions | `14,440` | `14,478` | `14,181` |
| Pred count p50 | `66.0` | `67.0` | `65.0` |
| Target bbox area ratio | `0.793524` | `0.786470` | `0.863738` |
| Target mask area ratio | `0.807673` | `0.798705` | `0.892377` |

Oracle:

- Output: `output/diagnostics/r74_balanced_ce_minfg005_target_oracle_20260517/r53_prediction_union_oracle_summary.json`
- Mask oracle R@50/R@75/R@90: `0.691064 / 0.373191 / 0.046128`
- One-to-one oracle R@50/R@75: `0.690723 / 0.373191`
- Matched oracle-score segm AP/AP50/AP75: `0.376238 / 0.693069 / 0.376238`
- Dense `>90` oracle R@50/R@75: `0.512573 / 0.193853`
- Tiny `area <= 256` oracle R@50/R@75: `0.183367 / 0.016701`

Dense geometry audit:

- Audit JSON: `output/diagnostics/r74_balanced_ce_minfg005_target_unlabeled200_20260517/r74_balanced_ce_minfg005_target_unlabeled200_audit.json`
- Target mask TP/FP/FN: `8120 / 6061 / 3630`
- Target bbox TP/FP/FN: `8940 / 5241 / 2810`
- Matched TP bbox area ratio median/p90: `0.934372 / 1.168390`
- Matched TP mask area ratio median/p90: `1.032026 / 1.295268`

## Comparison And Decision

| Signal | R46 | R71 | R73 | R74 |
|---|---:|---:|---:|---:|
| target bbox AP | `0.395575` | `0.363805` | `0.364197` | `0.387567` |
| target segm AP | `0.323252` | `0.331164` | `0.330836` | `0.336010` |
| target segm AP75 | `0.287485` | `0.309062` | `0.308940` | `0.315963` |
| target dense `>90` oracle R@75 | not rerun here | `0.187621` | `0.191274` | `0.193853` |
| target tiny oracle R@75 | not primary | `0.015678` | `0.018064` | `0.016701` |
| target mask area ratio | `1.023916` | `0.807673` | `0.798705` | `0.892377` |

Decision: pass and useful.

R74 beats R71 target segm AP and AP75 while recovering much of R71's bbox AP drop. It also keeps area ratios below R46, so raising `balanced_ce_min_fg_ratio` reduced the mask-size pressure without compressing predictions as much as `balanced_ce=false`. Tiny R@75 does not beat R73, but dense R@75 and AP both improve, so the main diagnosis supports the capped balanced CE path.
