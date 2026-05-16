# VC-SUDA R51 Scale0.24 Source1K True-Resume Gate - 2026-05-16

## Conclusion

R51 fails the formal gate.

The formal external `target_unlabeled200` 1024 backmap eval at `ckpt0750` produced segm AP/AP50/AP75 `0.3195861918/0.6483751360/0.2818685231`. This is below R46 `0.3232518088/0.649728/0.2874850698` and below R50 full scale0.24 segm AP `0.3204125968`.

Gate reading: ordinary resized source/provenance is worse than pseudo_real source for this 1K subset gate. The R51 result also does not show that the 1K subset beats the full scale0.24 source run.

No longer training was started for this gate.

## Scope

R51 tested the validated `scale024` source 1K subset under the R35/R12 true-resume protocol.

- Config: `configs/vc_suda_stage_c_r51_scale024_source1k_true_resume_1024_teacher8499.yaml`
- Source annotation: `magformer_datasets/20260318_1K_32254_scale024/annotations/instances_train_scale024_r51_1k.json`
- Subset validation: `seed=51`, images `1008`, annotations `55303`, mask area p50 `434.477`, target overlap `0`
- Resume checkpoint: `output/vc_suda/stage_c_r12_32k_source_r8b_ckpt999_continue_1024_teacher8499/checkpoint_iter_0000499.pth`
- Max iter: `750`
- Training tmux: `r51_scale024_source1k_gate_20260516`

Held fixed from the R35/R46/R50 protocol:

- Target split: `target_labeled=25`, `target_unlabeled=200`, `val=28`.
- `data.image_size: 1024`.
- `solver.max_iter: 750`, `solver.ims_per_batch: 4`, `solver.base_lr: 1.0e-05`.
- Depth minmax per-sample normalization.
- External eval: 1024 backmap, `bbox,segm`, score `0.05`, mask `0.5`, topk/maxDets `200`.

## Training

The existing tmux run was monitored to completion. It was not restarted.

Run details:

- Start timestamp: `2026-05-16T19:25:41+08:00`.
- Start state: true-resumed from R12 iter `499`.
- Completed: `750/750`.
- Training completed timestamp: `2026-05-16 19:34:15`.
- End timestamp: `2026-05-16T19:34:23+08:00`.
- `/usr/bin/time` wall time: `8:42.64`.
- `metrics_log.jsonl` trainer peak memory: `20565.0669 MB`.
- Takeover sampled GPU memory peaks: GPU4 `22903 MiB`, GPU5 `22951 MiB`, GPU6 `22505 MiB`, GPU7 `21909 MiB`.
- Takeover sampled system RAM peak used: `37035 MiB`; minimum available RAM: `210420 MiB`.
- `/usr/bin/time` max resident set size: `6548944 KB`.
- Last logged aggregate train loss: iter `740`, loss `17.1969`.
- Final visible per-rank tqdm losses at iter `750`: `17.4578`, `72.5508`, `16.2276`, `35.7280`.

Actual checkpoints found in the output directory:

- `output/vc_suda/stage_c_r51_scale024_source1k_true_resume_1024_teacher8499/checkpoint_iter_0000749.pth`
- `output/vc_suda/stage_c_r51_scale024_source1k_true_resume_1024_teacher8499/checkpoint_iter_0000750.pth`

The formal gate uses `checkpoint_iter_0000750.pth` because it is the actual final checkpoint.

Log scan:

- No `Traceback`, OOM, CUDA OOM, non-finite loss, missing/unexpected key error, or shape mismatch was found in the R51 training log scan.
- The training log contains the expected R12 resume evidence: `Resumed from iter 499`.
- The only post-exit warning was the PyTorch NCCL process-group destroy warning after completion.

Built-in val28 final eval at iter `750` was diagnostic only:

| Metric | AP | AP50 | AP75 |
|---|---:|---:|---:|
| bbox | `0.1133872826` | `0.3736005063` | `0.0439164753` |

## External Eval

Output directory:

- `output/diagnostics/r51_scale024_source1k_ckpt0750_target_unlabeled200_1024_backmap_20260516`

Command:

```bash
CUDA_VISIBLE_DEVICES=4 \
MAGFORMER_MS_DEFORM_ATTN_BACKEND=pytorch \
/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python tools/evaluate_1024_backmap.py \
  --base-config configs/eval_vc_suda_stage_b_1024_teacher8499_segm.yaml \
  --dataset-root magformer_datasets/pseudo_real_512 \
  --ann annotations/instances_target_unlabeled.json \
  --split train \
  --weights output/vc_suda/stage_c_r51_scale024_source1k_true_resume_1024_teacher8499/checkpoint_iter_0000750.pth \
  --output-dir output/diagnostics/r51_scale024_source1k_ckpt0750_target_unlabeled200_1024_backmap_20260516 \
  --image-size 1024 \
  --batch-size 1 \
  --num-workers 0 \
  --score-threshold 0.05 \
  --mask-threshold 0.5 \
  --iou-types bbox,segm \
  --inference-topk 200 \
  --max-dets 200 \
  --dump-inference-stats output/diagnostics/r51_scale024_source1k_ckpt0750_target_unlabeled200_1024_backmap_20260516/inference_stats.json \
  --force-pytorch-msda
```

Validation:

- Forced PyTorch MSDA path: `[MSDeformAttn] forced PyTorch core path`.
- Strict load: `774/774`, `missing=0`, `unexpected=0`, `shape_mismatch=0`.
- Evaluated all `200` target_unlabeled images.
- Predictions: `13,691`.
- Topk truncated images: `0/200`.
- Exit: `EXIT_CODE=0`.
- Runtime by `/usr/bin/time`: `real 131.67` seconds.

Metrics:

| Metric | AP | AP50 | AP75 |
|---|---:|---:|---:|
| bbox | `0.3943881435` | `0.7333336771` | `0.3798259450` |
| segm | `0.3195861918` | `0.6483751360` | `0.2818685231` |

## Oracle And Miss Diagnostics

R38-style oracle score output:

- Output dir: `output/diagnostics/r51_oracle_score_upper_bound_20260516`.
- Predictions: `13,691`.
- Max-IoU oracle R@50/R@75/R@90: `0.6851914894/0.3445106383/0.0366808511`.
- One-to-one oracle R@50/R@75: `0.6851914894/0.3445106383`.
- Dense `>90` oracle R@75: `0.1654846336`.
- Small `<=256` oracle R@75: `0.0095432856`.
- Matched-IoU score oracle segm AP/AP50/AP75: `0.3594059406/0.6831683168/0.3465346535`.

R40-style miss atlas output:

- Output dir: `output/diagnostics/r51_oracle_miss_atlas_20260516`.
- No-cover/low-quality/good/high-quality counts: `3699/4003/3617/431`.
- Rates: `31.48%/34.07%/30.78%/3.67%`.
- Bad GT instances: `7702`, all in images with prediction candidates.
- Bad subtypes: mask-alignment-shape `4211` (`54.67%`), mask-too-large `3051` (`39.61%`), mask-too-small `284` (`3.69%`), mask-offset `156` (`2.03%`).

Compared with R46:

| Signal | R46 | R51 | Direction |
|---|---:|---:|---|
| segm AP | `0.3232518088` | `0.3195861918` | worse |
| segm AP75 | `0.2874850698` | `0.2818685231` | worse |
| predictions | `13,758` | `13,691` | lower |
| oracle R@75 | `0.348511` | `0.344511` | worse |
| dense `>90` R@75 | `0.170213` | `0.165485` | worse |
| small `<=256` R@75 | `0.010566` | `0.009543` | worse |
| no-cover / low-quality / good / high-quality | `3692/3963/3631/464` | `3699/4003/3617/431` | worse mix |

Compared with R50 full scale0.24:

| Signal | R50 full | R51 source1K | Direction |
|---|---:|---:|---|
| segm AP | `0.3204125968` | `0.3195861918` | worse |
| segm AP75 | `0.2831707360` | `0.2818685231` | worse |
| predictions | `13,633` | `13,691` | higher |
| oracle R@75 | `0.344085` | `0.344511` | slightly better |
| dense `>90` R@75 | `0.166129` | `0.165485` | worse |
| no-cover / low-quality / good / high-quality | `3695/4012/3606/437` | `3699/4003/3617/431` | mixed, not better |

Compared with R12/R33:

| Signal | R12/R33 | R51 | Direction |
|---|---:|---:|---|
| segm AP | `0.3200483457` | `0.3195861918` | worse |
| segm AP75 | `0.2841442923` | `0.2818685231` | worse |
| oracle R@75 | `0.347830` | `0.344511` | worse |

## Gate Decision

Gate lines from this handoff:

- Current best R46: segm AP `0.323252`, AP75 `0.287485`.
- R50 full scale0.24: segm AP `0.320413`, AP75 `0.283171`.
- Pass: AP `>0.323252`, AP75 `>=0.287485`, and oracle not worse.
- R51 `< R46`: ordinary resized source/provenance is worse than pseudo_real source.
- R51 `>= R50 full`: full scale024 did not help under the 250-step schedule.

R51 result:

- Segm AP/AP75: `0.3195861918/0.2818685231`.
- Oracle R@75: `0.3445106383`.
- Miss mix: `3699/4003/3617/431`.

Decision: fail, record-only.

R51 is below R46, below R50 full, and below R12/R33 on AP, AP75, and oracle. It is not a formal pass and should not be promoted.

## Files

- Config: `configs/vc_suda_stage_c_r51_scale024_source1k_true_resume_1024_teacher8499.yaml`.
- Training output: `output/vc_suda/stage_c_r51_scale024_source1k_true_resume_1024_teacher8499`.
- Training log: `output/diagnostics/r51_scale024_source1k_train_20260516/train.log`.
- External eval output: `output/diagnostics/r51_scale024_source1k_ckpt0750_target_unlabeled200_1024_backmap_20260516`.
- Oracle output: `output/diagnostics/r51_oracle_score_upper_bound_20260516`.
- Miss atlas output: `output/diagnostics/r51_oracle_miss_atlas_20260516`.
- Result doc: `docs/results/vc_suda_r51_scale024_source1k_true_resume_20260516.md`.
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
