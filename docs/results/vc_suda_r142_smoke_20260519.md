# VC-SUDA R142 Smoke Results - 2026-05-19

## Scope

Confirmed R142 after `1bca337bf223af490e0268fd0ae2caab9eb62559` with targeted unit checks, one real-batch no-write preflight forward, and a 50-iteration tmux smoke. No formal training was started.

## Environment

- Host: `ssh 4029`
- Repo: `/home/hdd3/zhanghaonan/magformer`
- HEAD: `1bca337bf223af490e0268fd0ae2caab9eb62559`
- Working tree before smoke: clean
- GPU: `CUDA_VISIBLE_DEVICES=7` only; GPU7 was idle before launch and idle after completion
- CPU thread limits: `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`, `OPENBLAS_NUM_THREADS=4`, `NUMEXPR_NUM_THREADS=4`

## Unit Checks

Command:

```bash
/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python -m pytest -q   tests/test_depth_validity_tool_boundaries.py   tests/test_raw_depth_validity_pipeline.py::test_collate_batches_depth_valid_masks   tests/test_raw_depth_validity_pipeline.py::test_depth_prior_valid_hole_uses_raw_validity_not_normalized_depth_range   tests/test_vc_suda_train_entrypoint.py::test_stage_c_r142_config_uses_32254_total_25654_source_train_split_online_ema
```

Output:

```text
.........                                                                [100%]
```

## Real-Batch No-Write Forward Smoke

Command summary: loaded `configs/vc_suda_stage_c_r142_32254_train25654_source_target150_fixed.yaml`, built `build_datasets` and `build_data_loaders` with `batch_size=1`, took one source batch, verified `source_depth_valid_masks`, and passed it into `model.collect_preflight_diagnostics(...)` on GPU7.

Key output:

```text
[CocoRgbdDataset] Loaded 25654 images from train split
[CocoRgbdDataset] Loaded 150 images from train split
[CocoRgbdDataset] Loaded 75 images from train split
[SemiSupervisedDataset] Stage=C, source=25654, target_labeled=150, target_unlabeled=75, target_unlabeled_sequence=75, offline_pseudo=disabled
source_images_shape=(1, 3, 1024, 1024)
source_depths_shape=(1, 1, 1024, 1024)
source_depth_valid_masks_shape=(1, 1, 1024, 1024) dtype=torch.bool valid_frac=1.000000
diagnostics_keys=[confidence_maps, pred_masks]
pred_masks_shape=(1, 200, 256, 256) finite=True
forward_smoke=PASS
```

## Training Smoke

Tmux session:

```text
r142_smoke_20260519_codex
```

Output directory:

```text
output/vc_suda/r142_32254_train25654_target150_smoke_50iter_20260519
```

Saved files include:

```text
launch_command.sh
run.log
config_smoke.yaml
config_resolved.yaml
metrics_log.jsonl
metrics_log.csv
depth_sanity.json
checkpoint_iter_0000549.pth
```

Launch command:

```bash
cd /home/hdd3/zhanghaonan/magformer
export CUDA_VISIBLE_DEVICES=7
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python tools/train.py   --config output/vc_suda/r142_32254_train25654_target150_smoke_50iter_20260519/config_smoke.yaml   --gpus 0   --num-workers 1
```

Smoke config kept the R142 data and loss settings, used single visible GPU7 as `cuda:0`, changed `solver.ims_per_batch` to `1`, and set `solver.max_iter` to `549`. The R142 resume checkpoint starts at iter 499, so this ran 50 train rows, iters 499-548, then saved final state at iter 549.

Loader and preflight log evidence:

```text
[CocoRgbdDataset] Loaded 25654 images from train split
[CocoRgbdDataset] Loaded 150 images from train split
[CocoRgbdDataset] Loaded 75 images from train split
[SemiSupervisedDataset] Stage=C, source=25654, target_labeled=150, target_unlabeled=75, target_unlabeled_sequence=75, offline_pseudo=disabled
[Train] Wrote depth sanity report to output/vc_suda/r142_32254_train25654_target150_smoke_50iter_20260519/depth_sanity.json
[VCSUDATrainer] Resumed from iter 499, epoch 0, best_metric -inf
[2026-05-19 19:46:29] training completed
[2026-05-19 19:46:30] eval_summary mAP=0.2149
```

Depth sanity:

```text
depth_sanity_should_abort=False
depth_sanity_reasons=[]
```

Loss metrics from 50 train rows:

| Metric | Min | Max | Last |
| --- | ---: | ---: | ---: |
| `train/source_total_loss` | 3.06673 | 23.0191 | 3.73327 |
| `train/tl_total_loss_raw` | 12.1728 | 54.2983 | 12.1728 |
| `train/tl_total_loss_weighted` | 12.1728 | 54.2983 | 12.1728 |
| `train/pseudo_total` | 0.841309 | 3.11719 | 1.88477 |
| `train/pseudo_kept_count` | 11 | 85 | 45 |
| `train/pseudo_keep_rate` | 0.11 | 0.85 | 0.45 |

Pseudo weighted contribution was estimated as `(train/loss - train/source_total_loss - train/tl_total_loss_weighted) / train/loss` because the weighted pseudo term is not logged directly.

```text
pseudo_weighted_residual: min=0.000188351 max=0.000675201 mean=0.000419374 last=0.000431776
pseudo_weighted_contribution_ratio: min=0.000510% max=0.002714% mean=0.001370% last=0.002714%
```

This is below 5%. I did not tune it. The low contribution matches the configured unsupervised warmup behavior near the resumed iteration.

## GPU Status

Before smoke, GPU7 was idle:

```text
7, NVIDIA GeForce RTX 3090, 1 MiB used, 24576 MiB total, 0% util
```

During smoke, GPU7 peaked around 20.1 GB per `metrics_log.jsonl` and showed active utilization in `nvidia-smi`.

After smoke, GPU7 was idle again:

```text
7, 1 MiB used, 24576 MiB total, 0% util
```

Other nonzero GPUs were busy during this task; GPU0 was not used.

## Go / No-Go

Go for the requested checks:

- R142 loader uses source train split count 25,654, target_labeled 150, target_unlabeled 75.
- `source_depth_valid_masks` exists in real batches and is accepted by `collect_preflight_diagnostics`.
- Depth sanity/preflight did not abort.
- Source, target labeled, and pseudo/unsupervised loss paths all produced nonzero metrics.

Caveat:

- Pseudo weighted contribution was below 5%; actual mean was 0.001370%, last was 0.002714%. No tuning was performed.
