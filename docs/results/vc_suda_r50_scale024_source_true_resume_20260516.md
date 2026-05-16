# VC-SUDA R50 Scale0.24 Source True-Resume Gate - 2026-05-16

## Conclusion

R50 is record-only, not a pass.

The formal external `target_unlabeled200` 1024 backmap eval at `ckpt0750` produced segm AP/AP50/AP75 `0.3204125968/0.6497095155/0.2831707360`. This is slightly above the R12/R33 AP line `0.3200483457`, but below the current best R46 AP/AP75 line `0.3232518088/0.2874850698`. Oracle and miss diagnostics also do not improve over R46.

No longer training was started for this gate.

## Scope

R50 tested the materialized scale0.24 source dataset from R49 under the R35/R12 true-resume protocol.

- Config: `configs/vc_suda_stage_c_r50_scale024_source_true_resume_1024_teacher8499.yaml`
- Source root: `magformer_datasets/20260318_1K_32254_scale024`
- Source annotation: `annotations/instances_train_scale024.json`
- Resume checkpoint: `output/vc_suda/stage_c_r12_32k_source_r8b_ckpt999_continue_1024_teacher8499/checkpoint_iter_0000499.pth`
- Max iter: `750`
- Training tmux: `r50_scale024_source_gate_20260516`

Held fixed from the R35/R46 true-resume protocol:

- Target split: `target_labeled=25`, `target_unlabeled=200`, `val=28`.
- `data.image_size: 1024`.
- `solver.max_iter: 750`, `solver.ims_per_batch: 4`, `solver.base_lr: 1.0e-05`.
- Depth minmax per-sample normalization.
- External eval: 1024 backmap, `bbox,segm`, score `0.05`, mask `0.5`, topk/maxDets `200`.

## Training

The existing tmux run was monitored to completion. It was not restarted.

Run details:

- Start state: true-resumed from R12 iter `499`.
- Completed: `750/750`.
- Completion timestamp: `2026-05-16 18:46:58`.
- Training progress wall time shown by tqdm: `10:29`.
- Final built-in eval timestamp: `2026-05-16 18:47:28`.
- Peak trainer memory: `20587.08 MB` from `peak_memory_mb.txt`.
- Peak `nvidia-smi` GPU memory sample: `22963 MiB`.
- Peak sampled system RAM used: `176636 MB` of `257582 MB`.
- Final visible rank losses at iter `750`: `35.7808`, `45.3110`, `13.7490`, `32.1352`.

Actual checkpoints found in the output directory:

- `output/vc_suda/stage_c_r50_scale024_source_true_resume_1024_teacher8499/checkpoint_iter_0000749.pth`
- `output/vc_suda/stage_c_r50_scale024_source_true_resume_1024_teacher8499/checkpoint_iter_0000750.pth`

The formal gate uses `checkpoint_iter_0000750.pth` because it is the actual final checkpoint.

Log scan:

- No `Traceback`, OOM, CUDA OOM, non-finite loss, or shape mismatch was found in the R50 training log scan.
- The training log contains the expected R12 resume evidence: `Resumed from iter 499`.
- The only post-exit warning was the PyTorch NCCL process-group destroy warning after completion.
- The tmux pane also contained repeated `awk: line 1: runaway string constant` messages from the live monitor, not from the training script.

Built-in val28 final eval at iter `750` was diagnostic only:

| Metric | AP | AP50 | AP75 |
|---|---:|---:|---:|
| bbox | `0.1132161488` | `0.3683439206` | `0.0418453298` |

## External Eval

Output directory:

- `output/diagnostics/r50_scale024_source_true_resume_ckpt0750_target_unlabeled200_1024_backmap_20260516`

Command:

```bash
CUDA_VISIBLE_DEVICES=4 \
MAGFORMER_MS_DEFORM_ATTN_BACKEND=pytorch \
/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python tools/evaluate_1024_backmap.py \
  --base-config configs/eval_vc_suda_stage_b_1024_teacher8499_segm.yaml \
  --dataset-root magformer_datasets/pseudo_real_512 \
  --ann annotations/instances_target_unlabeled.json \
  --split train \
  --weights output/vc_suda/stage_c_r50_scale024_source_true_resume_1024_teacher8499/checkpoint_iter_0000750.pth \
  --output-dir output/diagnostics/r50_scale024_source_true_resume_ckpt0750_target_unlabeled200_1024_backmap_20260516 \
  --image-size 1024 \
  --batch-size 1 \
  --num-workers 0 \
  --score-threshold 0.05 \
  --mask-threshold 0.5 \
  --iou-types bbox,segm \
  --inference-topk 200 \
  --max-dets 200 \
  --dump-inference-stats output/diagnostics/r50_scale024_source_true_resume_ckpt0750_target_unlabeled200_1024_backmap_20260516/inference_stats.json \
  --force-pytorch-msda
```

Validation:

- Forced PyTorch MSDA path: `[MSDeformAttn] forced PyTorch core path`.
- Strict load: `774/774`, `missing=0`, `unexpected=0`, `shape_mismatch=0`.
- Evaluated all `200` target_unlabeled images.
- Predictions: `13,633`.
- Exit: `EXIT_CODE=0`.
- Runtime by `/usr/bin/time`: `real 137.12` seconds.

Metrics:

| Metric | AP | AP50 | AP75 |
|---|---:|---:|---:|
| bbox | `0.3943117447` | `0.7344062147` | `0.3803233178` |
| segm | `0.3204125968` | `0.6497095155` | `0.2831707360` |

## Oracle And Miss Diagnostics

R38-style oracle score output:

- Output dir: `output/diagnostics/r50_oracle_score_upper_bound_20260516`.
- Predictions: `13,633`.
- Max-IoU oracle R@50/R@75/R@90: `0.6855319149/0.3440851064/0.0371914894`.
- One-to-one oracle R@50/R@75: `0.6854468085/0.3440851064`.
- Dense `>90` oracle R@75: `0.1661293789`.
- Small `<=256` oracle R@75: `0.0095432856`.
- Matched-IoU score oracle segm AP/AP50/AP75: `0.3613861386/0.6831683168/0.3465346535`.

R40-style miss atlas output:

- Output dir: `output/diagnostics/r50_oracle_miss_atlas_20260516`.
- No-cover/low-quality/good/high-quality counts: `3695/4012/3606/437`.
- Rates: `31.45%/34.14%/30.69%/3.72%`.
- Bad GT instances: `7707`, all in images with prediction candidates.
- Bad subtypes: mask-alignment-shape `4232` (`54.91%`), mask-too-large `3049` (`39.56%`), mask-too-small `257` (`3.33%`), mask-offset `169` (`2.19%`).

Compared with R46:

| Signal | R46 | R50 | Direction |
|---|---:|---:|---|
| segm AP | `0.3232518088` | `0.3204125968` | worse |
| segm AP75 | `0.2874850698` | `0.2831707360` | worse |
| predictions | `13,758` | `13,633` | lower |
| oracle R@75 | `0.348511` | `0.344085` | worse |
| dense `>90` R@75 | `0.170213` | `0.166129` | worse |
| small `<=256` R@75 | `0.010566` | `0.009543` | worse |
| no-cover / low-quality / good / high-quality | `3692/3963/3631/464` | `3695/4012/3606/437` | worse mix |

Compared with R12/R33, R50 improves segm AP only:

| Signal | R12/R33 | R50 | Direction |
|---|---:|---:|---|
| segm AP | `0.3200483457` | `0.3204125968` | slightly better |
| segm AP75 | `0.2841442923` | `0.2831707360` | worse |
| oracle R@75 | `0.347830` | `0.344085` | worse |

## Gate Decision

Gate lines from this handoff:

- Current best R46: segm AP `0.323252`, AP75 `0.287485`.
- Pass: AP `>0.323252`, AP75 `>=0.287485`, and oracle/miss not worse.
- Fail: AP `<0.320048` or AP/AP75/oracle all fail to improve.
- Record-only: between R12 and R46.

R50 result:

- Segm AP/AP75: `0.3204125968/0.2831707360`.
- Oracle R@75: `0.3440851064`.
- Miss mix: `3695/4012/3606/437`.

Decision: record-only.

R50 is not a pass because AP and AP75 are below R46, and oracle/miss are worse than R46. It is not a hard fail under the handoff rule because segm AP is slightly above the R12/R33 AP line. Do not promote this checkpoint over R46.

## Files

- Config: `configs/vc_suda_stage_c_r50_scale024_source_true_resume_1024_teacher8499.yaml`.
- Training output: `output/vc_suda/stage_c_r50_scale024_source_true_resume_1024_teacher8499`.
- External eval output: `output/diagnostics/r50_scale024_source_true_resume_ckpt0750_target_unlabeled200_1024_backmap_20260516`.
- Oracle output: `output/diagnostics/r50_oracle_score_upper_bound_20260516`.
- Miss atlas output: `output/diagnostics/r50_oracle_miss_atlas_20260516`.
- Result doc: `docs/results/vc_suda_r50_scale024_source_true_resume_20260516.md`.
- Summary update: `docs/results/results_summary.md`.

## Validation

Commands completed:

```bash
/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python tools/evaluate_1024_backmap.py ... --force-pytorch-msda
/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python /tmp/r38_oracle_bound.py ...
/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python /tmp/r40_oracle_miss_atlas.py ...
```

Pre-commit validation for the doc commit must include:

```bash
git diff --check
```
