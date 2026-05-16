# VC-SUDA R52 target_unlabeled sampling gate, 2026-05-16

## Conclusion

R52 is record-only/fail for the target sampling gate.

The formal external `target_unlabeled200` 1024 backmap eval at `ckpt0750` produced segm AP/AP50/AP75 `0.3225963120/0.6494580497/0.2869152875`. This is above the R12/R33 fail floor `0.3200483457/0.2841442923`, but below the current best R46 line `0.3232518088/0.2874850698`.

Oracle/miss diagnostics do not rescue it. Global oracle R@75 and dense `>90` R@75 are slightly below R46; small `<=256` R@75 improves, but the main dense target bottleneck does not. Do not promote R52 over R46, and do not continue expanding target sampling from this result.

## Scope

R52 tested prediction-only target_unlabeled repeat sampling under the R46 pseudo-real source true-resume protocol.

- Config: `configs/vc_suda_stage_c_r52_pseudoreal_source_target_sampling_true_resume_1024_teacher8499.yaml`
- Resume checkpoint: `output/vc_suda/stage_c_r12_32k_source_r8b_ckpt999_continue_1024_teacher8499/checkpoint_iter_0000499.pth`
- Max iter: `750`
- Training tmux: `r52_target_sampling_gate_20260516`
- Sampling stats: `output/diagnostics/r52_target_unlabeled_sampling_20260516/target_sampling_stats.json`

Sampling setup from the reviewed R52 mechanism:

| Bucket | Images | Repeat |
|---|---:|---:|
| normal | `158` | `1` |
| dense | `30` | `2` |
| dense_tiny | `12` | `3` |

Expanded target_unlabeled sequence length: `254`.

Held fixed from R46:

- Source remains `magformer_datasets/pseudo_real_512/annotations/instances_source.json`.
- Target splits remain `target_labeled=25`, `target_unlabeled=200`, `val=28`.
- `data.image_size: 1024`.
- `solver.max_iter: 750`, `solver.ims_per_batch: 4`, `solver.base_lr: 1.0e-05`.
- External eval uses 1024 backmap, `bbox,segm`, score `0.05`, mask `0.5`, topk/maxDets `200`.

## Training

Command contract:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
MAGFORMER_MS_DEFORM_ATTN_BACKEND=cuda \
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 tools/train.py \
  --config configs/vc_suda_stage_c_r52_pseudoreal_source_target_sampling_true_resume_1024_teacher8499.yaml \
  --output-dir output/vc_suda/stage_c_r52_pseudoreal_source_target_sampling_true_resume_1024_teacher8499 \
  --gpus 0,1,2,3 \
  --num-workers 2
```

Run details:

- Start timestamp: `2026-05-16T20:13:14+08:00`.
- Start state: true-resumed from R12 iter `499` on all ranks.
- Dataset line: `source=1008`, `target_labeled=25`, `target_unlabeled=200`, `target_unlabeled_sequence=254`.
- Completed: `750/750`.
- End timestamp: `2026-05-16T20:23:17+08:00`.
- Exit: `EXIT_CODE=0`.
- `/usr/bin/time` wall time: `10:02.50`.
- `/usr/bin/time` max resident set size: `5619168 KB`.
- `metrics_log.jsonl` trainer peak memory: `20602.0059 MB`.
- Sampled physical GPU memory peaks on GPUs 4-7: `22927/22615/22633/22417 MiB`.
- Sampled system RAM peak used: `31495 MiB`; minimum available RAM: `215789 MiB`.
- Last logged aggregate train loss: iter `740`, loss `19.6662`.

Actual checkpoints found:

- `output/vc_suda/stage_c_r52_pseudoreal_source_target_sampling_true_resume_1024_teacher8499/checkpoint_iter_0000749.pth`
- `output/vc_suda/stage_c_r52_pseudoreal_source_target_sampling_true_resume_1024_teacher8499/checkpoint_iter_0000750.pth`

The formal gate uses `checkpoint_iter_0000750.pth` because it is the actual final checkpoint.

Log scan:

- `Traceback`: `0`
- `OutOfMemoryError` / `out of memory` / `CUDA out of memory`: `0`
- `non-finite` / `nonfinite`: `0`
- `missing keys` / `unexpected keys`: `0`
- `shape mismatch`: `0`

Built-in val28 final eval at iter `750` was diagnostic only:

| Metric | AP | AP50 | AP75 |
|---|---:|---:|---:|
| bbox | `0.1239872185` | `0.4008999505` | `0.0436046855` |

## External Eval

Output directory:

- `output/diagnostics/r52_target_sampling_ckpt0750_target_unlabeled200_1024_backmap_20260516`

Command:

```bash
CUDA_VISIBLE_DEVICES=4 \
MAGFORMER_MS_DEFORM_ATTN_BACKEND=pytorch \
/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python tools/evaluate_1024_backmap.py \
  --base-config configs/eval_vc_suda_stage_b_1024_teacher8499_segm.yaml \
  --dataset-root magformer_datasets/pseudo_real_512 \
  --ann annotations/instances_target_unlabeled.json \
  --split train \
  --weights output/vc_suda/stage_c_r52_pseudoreal_source_target_sampling_true_resume_1024_teacher8499/checkpoint_iter_0000750.pth \
  --output-dir output/diagnostics/r52_target_sampling_ckpt0750_target_unlabeled200_1024_backmap_20260516 \
  --image-size 1024 \
  --batch-size 1 \
  --num-workers 0 \
  --score-threshold 0.05 \
  --mask-threshold 0.5 \
  --iou-types bbox,segm \
  --inference-topk 200 \
  --max-dets 200 \
  --dump-inference-stats output/diagnostics/r52_target_sampling_ckpt0750_target_unlabeled200_1024_backmap_20260516/inference_stats.json \
  --force-pytorch-msda
```

Validation:

- Forced PyTorch MSDA path: `[MSDeformAttn] forced PyTorch core path`.
- Strict load: `774/774`, `missing=0`, `unexpected=0`, `shape_mismatch=0`.
- Evaluated all `200` target_unlabeled images.
- Predictions: `13,735`.
- Topk truncated images: `0/200`.
- Exit: `EXIT_CODE=0`.
- Runtime by `/usr/bin/time`: `2:10.82`.

Metrics:

| Metric | AP | AP50 | AP75 |
|---|---:|---:|---:|
| bbox | `0.3959065924` | `0.7333279217` | `0.3841737349` |
| segm | `0.3225963120` | `0.6494580497` | `0.2869152875` |

## Oracle And Miss Diagnostics

R38-style oracle score output:

- Output dir: `output/diagnostics/r52_oracle_score_upper_bound_20260516`.
- Predictions: `13,735`.
- Max-IoU oracle R@50/R@75/R@90: `0.6867234043/0.3482553191/0.0388936170`.
- One-to-one oracle R@50/R@75: `0.6865531915/0.3482553191`.
- Dense `>90` oracle R@75: `0.1695680206`.
- Small `<=256` oracle R@75: `0.0119291070`.
- Matched-IoU score oracle segm AP/AP50/AP75: `0.3623762376/0.6831683168/0.3465346535`.

R40-style miss atlas output:

- Output dir: `output/diagnostics/r52_oracle_miss_atlas_20260516`.
- No-cover/low-quality/good/high-quality counts: `3681/3977/3635/457`.
- Rates: `31.33%/33.85%/30.94%/3.89%`.
- Bad GT instances: `7658`, all in images with prediction candidates.
- Bad subtypes: mask-alignment-shape `4207` (`54.94%`), mask-too-large `3026` (`39.51%`), mask-too-small `259` (`3.38%`), mask-offset `166` (`2.17%`).

Compared with R46:

| Signal | R46 | R52 | Direction |
|---|---:|---:|---|
| segm AP | `0.3232518088` | `0.3225963120` | worse |
| segm AP75 | `0.2874850698` | `0.2869152875` | worse |
| predictions | `13,758` | `13,735` | lower |
| oracle R@75 | `0.348511` | `0.348255` | worse |
| dense `>90` R@75 | `0.170213` | `0.169568` | worse |
| small `<=256` R@75 | `0.010566` | `0.011929` | better |
| no-cover / low-quality / good / high-quality | `3692/3963/3631/464` | `3681/3977/3635/457` | mixed, not better |

Compared with R12/R38/R40:

| Signal | R12/R38/R40 | R52 | Direction |
|---|---:|---:|---|
| segm AP | `0.3200483457` | `0.3225963120` | better |
| segm AP75 | `0.2841442923` | `0.2869152875` | better |
| oracle R@75 | `0.347830` | `0.348255` | better |
| dense `>90` R@75 | `0.172577` | `0.169568` | worse |
| small `<=256` R@75 | `0.009884` | `0.011929` | better |
| no-cover / low-quality / good / high-quality | `3709/3954/3653/434` | `3681/3977/3635/457` | mixed |

## Gate Decision

Gate lines from the handoff:

- Pass: segm AP `>0.3232518088`, AP75 `>=0.2874850698`, and oracle/miss not worse.
- Fail: segm AP `<0.3200483457` or AP75 `<0.2841442923`.
- Record-only: between R12 and R46 without dense/tiny improvement.

R52 result:

- Segm AP/AP75: `0.3225963120/0.2869152875`.
- Oracle R@75: `0.3482553191`.
- Dense `>90` R@75: `0.1695680206`.
- Small `<=256` R@75: `0.0119291070`.
- Miss mix: `3681/3977/3635/457`.

Decision: record-only/fail.

R52 clears the hard fail floor and improves ordinary AP/AP75 over R12/R33. It is still below R46, so it does not pass the target sampling gate. AP, AP75, global oracle R@75, dense R@75, and high-quality count remain below R46. The small-object oracle improvement is real but isolated, so target_unlabeled repeat sampling is not a better checkpoint than R46 under the 250-iter gate. Do not continue expanding target sampling from R52.

## Files

- Config: `configs/vc_suda_stage_c_r52_pseudoreal_source_target_sampling_true_resume_1024_teacher8499.yaml`.
- Training output: `output/vc_suda/stage_c_r52_pseudoreal_source_target_sampling_true_resume_1024_teacher8499`.
- Training log: `output/diagnostics/r52_target_sampling_train_20260516/train.log`.
- External eval output: `output/diagnostics/r52_target_sampling_ckpt0750_target_unlabeled200_1024_backmap_20260516`.
- Oracle output: `output/diagnostics/r52_oracle_score_upper_bound_20260516`.
- Miss atlas output: `output/diagnostics/r52_oracle_miss_atlas_20260516`.
- Result doc: `docs/results/vc_suda_r52_target_unlabeled_sampling_gate_20260516.md`.
- Summary update: `docs/results/results_summary.md`.

## Validation

Commands completed:

```bash
/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python -m torch.distributed.run ... tools/train.py ...
/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python tools/evaluate_1024_backmap.py ... --force-pytorch-msda
/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python /tmp/r38_oracle_bound.py ...
/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python /tmp/r40_oracle_miss_atlas.py ...
```

No longer training was started.
