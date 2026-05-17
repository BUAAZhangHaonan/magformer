# VC-SUDA R76 balanced_ce min_fg 0.03, 2026-05-17

R76 is useful, but it does not beat R74.

It is the requested single-variable test between R46 and R74: keep `balanced_ce: true`, set `balanced_ce_min_fg_ratio: 0.03`, and leave resume/source/solver/loss/target/sampling settings unchanged. Target segm AP/AP75 is `0.334614 / 0.314388`, which is above R46 and R71 but below R74 `0.336010 / 0.315963`. This confirms `0.05` is the better cap in this family.

## Scope

Config:

- `configs/vc_suda_stage_c_r76_balanced_ce_minfg003_1024.yaml`
- Based on `configs/vc_suda_stage_c_r46_pseudoreal_source_true_resume_1024_teacher8499.yaml`
- Keeps `model.magformer.mask_former.balanced_ce: true`
- Changes `model.magformer.mask_former.balanced_ce_min_fg_ratio: 0.01 -> 0.03`
- No target sampling
- No source retention
- Same R46 true resume: `output/vc_suda/stage_c_r12_32k_source_r8b_ckpt999_continue_1024_teacher8499/checkpoint_iter_0000499.pth`

## Config And Preflight

Reusable setup evidence:

- `output/diagnostics/r76_balanced_ce_minfg003_setup_20260517/config_diff_verification.txt`

Verifier result:

```text
unexpected_count=0
balanced_ce=True
balanced_ce_min_fg_ratio=0.03
semantic_change_ok=true
```

Fresh lightweight Stage C verifier passed:

```bash
python tools/verify_vc_suda_stage.py --config configs/vc_suda_stage_c_r76_balanced_ce_minfg003_1024.yaml --stage C --emit-data-evidence --evidence-max-samples 1
```

Key checks included R12 `ckpt499` true-resume semantics, target_unlabeled `200`, val `28`, bbox-only quick eval, unlabeled label stripping, and non-constant depth.

## Training

Training ran in tmux on GPUs 4-7.

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 MAGFORMER_MS_DEFORM_ATTN_BACKEND=cuda OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 python -m torch.distributed.run --standalone --nproc_per_node=4 tools/train.py \
  --config configs/vc_suda_stage_c_r76_balanced_ce_minfg003_1024.yaml \
  --output-dir output/vc_suda/stage_c_r76_balanced_ce_minfg003_1024 \
  --gpus 0,1,2,3 \
  --num-workers 2
```

Run details:

- tmux session: `r76_balanced_ce_minfg003`
- Captured training log: `output/logs/r76_balanced_ce_minfg003_train_20260517_attempt2.log`
- Resource log: `output/logs/r76_balanced_ce_minfg003_resource_20260517_attempt2.log`
- Start: `2026-05-17T07:43:38+08:00`
- End: `2026-05-17T07:56:46+08:00`
- Exit: `EXIT_CODE=0`
- Resume: iter `499` from R12 `checkpoint_iter_0000499.pth`
- Final checkpoint: `output/vc_suda/stage_c_r76_balanced_ce_minfg003_1024/checkpoint_iter_0000750.pth`
- Log scan: `Traceback=0`, `CUDA out of memory=0`, `RuntimeError=0`, `non-finite=0`, `shape mismatch=0`
- Peak sampled system RAM: `31,585 MB / 257,582 MB` = `12.26%`
- Peak sampled GPU memory: GPU4 `22967`, GPU5 `22615`, GPU6 `22627`, GPU7 `22361` MiB

Note: the first tmux launch inherited a conflicting CUDA library path and failed before training with a PyTorch import error. The completed run explicitly unset `LD_LIBRARY_PATH` and `CUDA_HOME` after activating `magformer`, matching the existing project launch protocol.

## External Eval

Protocol checkers passed:

- Target: `output/diagnostics/r76_balanced_ce_minfg003_target_unlabeled200_20260517/protocol_summary.json`
- Source first50: `output/diagnostics/r76_balanced_ce_minfg003_original_first50_20260517/protocol_summary.json`

| Split | Eval dir | bbox AP/AP50/AP75 | segm AP/AP50/AP75 |
|---|---|---:|---:|
| target_unlabeled200 | `output/diagnostics/r76_balanced_ce_minfg003_target_unlabeled200_20260517` | `0.392850 / 0.730694 / 0.379484` | `0.334614 / 0.656427 / 0.314388` |
| original first50 | `output/diagnostics/r76_balanced_ce_minfg003_original_first50_20260517` | `0.475012 / 0.766590 / 0.511675` | `0.431587 / 0.745889 / 0.442038` |

## Target Area, Oracle, Tiny/Dense

Area ratio uses the image-level p50 convention.

| Signal | R76 |
|---|---:|
| Predictions | `14,132` |
| Pred count p50 | `65.0` |
| GT bbox area p50 over images | `779.75` |
| Pred bbox area p50 over images | `700.0` |
| Target bbox area ratio | `0.897724` |
| GT mask area p50 over images | `501.75` |
| Pred mask area p50 over images | `461.0` |
| Target mask area ratio | `0.918784` |

Oracle:

- Output: `output/diagnostics/r76_balanced_ce_minfg003_target_oracle_20260517/r53_prediction_union_oracle_summary.json`
- Mask oracle R@50/R@75/R@90: `0.690043 / 0.370213 / 0.045617`
- One-to-one oracle R@50/R@75: `0.689872 / 0.370213`
- Matched oracle-score segm AP/AP50/AP75: `0.374015 / 0.690644 / 0.376238`
- Matched oracle-score bbox AP/AP50/AP75: `0.421432 / 0.751907 / 0.422119`
- Dense `>90` oracle R@50/R@75/R@90: `0.508059 / 0.189985 / 0.005158`
- Tiny `area <= 256` oracle R@50/R@75/R@90: `0.176551 / 0.015678 / 0.0`
- Tiny one-to-one matched@50/matched@75: `518 / 46`

Dense geometry audit:

- Audit JSON: `output/diagnostics/r76_balanced_ce_minfg003_target_unlabeled200_20260517/r76_balanced_ce_minfg003_target_unlabeled200_audit.json`
- Target mask TP/FP/FN: `8108 / 6024 / 3642`
- Target bbox TP/FP/FN: `8970 / 5162 / 2780`
- Matched TP bbox area ratio median/p90: `0.960926 / 1.201363`
- Matched TP mask area ratio median/p90: `1.055891 / 1.328135`

## Comparison And Decision

| Signal | R46 | R71 | R74 | R76 |
|---|---:|---:|---:|---:|
| target bbox AP | `0.395575` | `0.363805` | `0.387567` | `0.392850` |
| target segm AP | `0.323252` | `0.331164` | `0.336010` | `0.334614` |
| target segm AP75 | `0.287485` | `0.309062` | `0.315963` | `0.314388` |
| target dense `>90` oracle R@75 | `0.170213` | `0.187621` | `0.193853` | `0.189985` |
| target tiny oracle R@75 | `0.010566` | `0.015678` | `0.016701` | `0.015678` |
| target mask area ratio | `1.023916` | `0.807673` | `0.892377` | `0.918784` |

Decision: useful, but not a new best.

R76 improves over R46 and R71 on target segm AP/AP75, and its target bbox AP is higher than R74. It does not beat R74 on the main target mask metrics: segm AP, segm AP75, dense R@75, and tiny R@75 are all slightly lower. Keep R74 as current best and prefer `balanced_ce_min_fg_ratio=0.05` over `0.03` for this branch.
