# VC-SUDA Stage C Launch - 2026-05-14

## Summary

- Status: launched and running under tmux.
- tmux session: `vc_suda_stage_c_20260514_0734`.
- torchrun PID: `3565245`.
- Output directory: `output/experiments/vc_suda_stage_c_1024_teacher8499_20260514_0734`.
- Launch log: `output/experiments/vc_suda_stage_c_1024_teacher8499_20260514_0734/train_launch.log`.
- Git HEAD at launch: `2d1e6e4e1226edbf8be83d339bd1ccdcd953fd0e`.

## Pre-Launch Checks

- `git status --short`: clean before launch.
- `git rev-parse --short HEAD`: `2d1e6e4`.
- GPU 4-7 were idle before launch: each showed 15 MiB used and 0% utilization.

## Strict Preflight

Command:

```bash
python tools/verify_vc_suda_stage.py --config configs/vc_suda_stage_c_1024_teacher8499.yaml
```

Result:

- `PASS config=configs/vc_suda_stage_c_1024_teacher8499.yaml`.
- Checks included: stage, unlabeled split, eval IoU types, unsupervised weight, EMA teacher, runtime no generic EMA, depth norm, target unlabeled/val overlap, checkpoint semantics, unlabeled batch strip, depth nonconstant.
- Reported `unsupervised_weight: 0.1`.
- Reported `target_unlabeled_images: 200` and `val_images: 28`.

## Pseudo-Label Diagnostic

Command:

```bash
CUDA_VISIBLE_DEVICES=4 python tools/diagnose_vc_suda_pseudo_labels.py \
  --config configs/vc_suda_stage_c_1024_teacher8499.yaml \
  --max-images 20 \
  --device cuda \
  --output-json /tmp/vc_suda_stage_c_pseudo_diag_20260514.json
```

Result:

- `PASS pseudo_label_diagnostics`.
- Effective threshold: `0.20`, source `vc_suda.curriculum`.
- Images sampled: `20`.
- Predictions: `1984`.
- Kept pseudo labels: `253`.
- `keep_rate: 0.1275201612903226`, above the `0.10` gate.
- `empty_ratio: 0.0`, below the `0.05` gate.

## Launch Command

The tmux server had inherited `LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:`, which conflicts with the conda env PyTorch build `2.5.1+cu124`. The first tmux attempt failed before training with a PyTorch import error. The formal running launch explicitly unsets `LD_LIBRARY_PATH` inside tmux.

```bash
source /home/hdd3/zhanghaonan/anaconda3/etc/profile.d/conda.sh
conda activate magformer
unset LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=4,5,6,7
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --standalone --nproc_per_node=4 tools/train.py \
  --config configs/vc_suda_stage_c_1024_teacher8499.yaml \
  --output-dir output/experiments/vc_suda_stage_c_1024_teacher8499_20260514_0734 \
  --gpus 0,1,2,3 \
  --num-workers 2 \
  2>&1 | tee -a output/experiments/vc_suda_stage_c_1024_teacher8499_20260514_0734/train_launch.log
```

## Initial Runtime Evidence

- DDP started 4 train ranks under `torchrun`.
- Runtime command in `run_metadata.json` uses config `configs/vc_suda_stage_c_1024_teacher8499.yaml`, output dir above, `--gpus 0,1,2,3`, and `--num-workers 2`.
- `run_metadata.json` records git dirty state as `false` at launch.
- Stage C dataset log on all ranks: `SemiSupervisedDataset] Stage=C, source=1008, target_labeled=25, target_unlabeled=200`.
- Warm-start loaded from `output/experiments/vc_suda_stage_b_1024_teacher8499_20260514_005821/checkpoint_iter_0008999.pth`.
- Warm-start key check on all ranks: `missing keys: 0, unexpected keys: 0`.
- EMA teacher log on all ranks: `EMA teacher enabled, momentum=0.999, warmup_steps=500`.
- Resolved config confirms `vc_suda.stage: C`, `pseudo_label.quality_threshold: 0.2`, `unsupervised_weight: 0.1`, and `unsupervised_warmup_epochs: 10`.
- Pseudo-loss path is active for Stage C in `VCSUDADDPTrainer`: teacher pseudo targets are built, `criterion.pseudo_label_loss()` returns `pseudo_loss_*` and `pseudo_total`, and `total_loss` adds `unsup_weight * pseudo_total`. The console logger only prints keys starting with `loss_`, so it does not print `pseudo_loss_*` by default.
- Non-fatal warning: SHA256 sidecar was not found for the Stage B checkpoint.
- Depth sanity report: `should_abort: false`, `aborted: false`.

## Initial Metrics

First structured metric:

```text
[2026-05-14 07:34:41]  iter=0/9000  eta=20:08:13  time=8.06s  lr=1e-07  loss=25.3507  loss_ce=0.0485  loss_dice=0.1240  loss_mask=0.0229
```

20-iteration structured metric:

```text
[2026-05-14 07:35:26]  iter=20/9000  eta=05:24:41  time=2.17s  lr=2.764e-06  loss=53.0734  loss_ce=0.1049  loss_dice=0.3396  loss_mask=0.0754
```

At the monitoring check after iter 20, GPUs 4-7 were actively used with about 20.5 GiB each allocated.

## First Checkpoint Monitoring

Checkpoint and metrics evidence at the first checkpoint boundary:

- `checkpoint_iter_0000499.pth` exists in `output/experiments/vc_suda_stage_c_1024_teacher8499_20260514_0734`, size `783499956` bytes (`748M` from `ls -lh`), mtime `2026-05-14T07:58:44+0800`.
- `metrics_log.jsonl` contained `28` structured rows at this check. The latest structured row was `iter=540` at `2026-05-14T08:00:58+08:00`, with `train/loss=58.839805603027344`, `train/lr=9.983652156021148e-05`, `iter_time_sec=1.8512292730505577`, and `peak_memory_mb=20070.0732421875`.
- First post-checkpoint metric row `iter=500` at `2026-05-14T07:58:51+08:00`: `train/pseudo_loss_ce=4.503488540649414`, `train/pseudo_loss_mask=0.013189372606575489`, `train/pseudo_loss_dice=0.6931500434875488`, `train/pseudo_total=47.795745849609375`.
- Pre-checkpoint comparison row `iter=480` at `2026-05-14T07:57:42+08:00`: `train/pseudo_loss_ce=1.0644007921218872`, `train/pseudo_loss_mask=0.0006682949606329203`, `train/pseudo_loss_dice=0.42097532749176025`, `train/pseudo_total=16.64353370666504`.
- `train_launch.log` latest sampled progress reached the 540-step console metric: `[2026-05-14 08:00:58]  iter=540/9000  eta=04:20:59  time=1.85s  lr=9.98365e-05  loss=58.8398  loss_ce=0.0072  loss_dice=0.1341  loss_mask=0.0448`. The log did not print `pseudo_loss_*` by default, so pseudo-loss evidence is from `metrics_log.jsonl`.
- GPU resource check showed the active 4-rank training workers on physical GPUs 4-7, each using about 20.9 GiB of a 24 GiB RTX 3090. Sampled utilization was `64%`, `33%`, `74%`, and `100%` on GPUs 4-7.
- Host resource check: system memory `251Gi` total, `212Gi` available; `/home/hdd3` had `7.7T` free (`44%` used).
- Training processes remained running under torchrun parent PID `3565245`; worker PIDs `3565260`-`3565263` had elapsed time about `27:17` at the resource check.

## Hard-Error Scan

No training hard error was found in the running log during initial monitoring. The only error-like lines were the non-fatal SHA256 sidecar warnings. The earlier failed tmux attempt is not the active run and was caused by inherited CUDA 12.1 library path in tmux.
