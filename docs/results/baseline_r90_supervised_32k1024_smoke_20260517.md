# R90 supervised-only 32K_1024 cache Teacher8499 smoke, 2026-05-17

R90 is a clean supervised baseline smoke. It is not VC-SUDA and it is not a long training run.

## Design

- Config: `configs/baseline_supervised_r90_32k1024_cache_teacher8499_smoke.yaml`
- Base config: `configs/baseline_supervised_r89_32k1024_cache_pilot.yaml`
- Source data: `magformer_datasets/20260318_1K_32254`
- Train annotation: `cache/coco_loader/instances_train.sqlite`
- Warm start: `output/experiments/20260510_1k_finetune_full_1024_v13/checkpoint_iter_0008499.pth`
- Output dir: `output/baseline/r90_supervised_32k1024_cache_teacher8499_smoke`
- GPUs: `[4, 5, 6, 7]`
- Workers: `2`
- Length: `solver.max_iter: 150`

The run keeps VC-SUDA disabled. Runtime EMA, contrastive loss, VC-SUDA EMA teacher, and offline pseudo labels are disabled. `model.weights` and `runtime.resume` are null, so the run is a model-only warm start from Teacher8499.

## Validation

Static assertions must pass before launch:

- `vc_suda.enabled == false`
- `runtime.ema_enabled == false`
- `runtime.contrastive_enabled == false`
- `vc_suda.ema_teacher.enabled == false`
- `vc_suda.offline_pseudo.enabled == false`
- Teacher8499 checkpoint exists
- 32K_1024 cache sqlite exists
- output dir does not already exist before launch
- `solver.max_iter == 150`, which is below the 200-iter hard cap

After the smoke, check the training log for Traceback, OOM, non-finite loss, and a normal final checkpoint. Prefer running the external original source first50 sanity with the project wrapper and PyTorch deform backend. If that eval is too slow, record that it was not run and use the smoke loss/log status only.

## Hard Stop

Stop immediately on any OOM, Traceback, non-finite loss, missing Teacher8499 checkpoint, missing sqlite cache, or accidental attempt to exceed 200 iterations.

## Not Long Training

This config is capped at 150 iterations. Its output, logs, and checkpoints are local artifacts only and must not be committed. A real R90 supervised baseline needs a separate config, training length, eval cadence, and result note.

## Run Log

Static assertions passed before launch. The checks covered disabled VC-SUDA, runtime EMA, contrastive loss, VC-SUDA EMA teacher, offline pseudo labels, Teacher8499 checkpoint existence, sqlite cache existence, output-dir isolation, GPU list, worker count, and the `150 <= 200` iteration cap.

First launch attempt failed immediately with `RuntimeError: CUDA error: invalid device ordinal`. Cause: `CUDA_VISIBLE_DEVICES=4,5,6,7` remapped visible devices to local `0..3`, while the project DDP resolver still used config GPUs `[4, 5, 6, 7]`. No training iteration ran in that attempt. The failed attempt log is local only: `output/logs/r90_supervised_32k1024_cache_teacher8499_smoke_attempt1_invalid_device.log`.

The completed smoke was relaunched in tmux without `CUDA_VISIBLE_DEVICES`, so the config-selected physical GPUs `[4, 5, 6, 7]` were used directly.

- Tmux session: `r90_supervised_smoke`
- Train log: `output/logs/r90_supervised_32k1024_cache_teacher8499_smoke.log`
- Output dir: `output/baseline/r90_supervised_32k1024_cache_teacher8499_smoke`
- Checkpoint: `output/baseline/r90_supervised_32k1024_cache_teacher8499_smoke/checkpoint_iter_0000150.pth`
- Final rank0 train loss line: `iter=149/150`, `loss=5.0590`, `loss_ce=0.0051`, `loss_dice=0.0339`, `loss_mask=0.0299`
- Error scan: no `Traceback`, OOM, or non-finite loss in the completed run log

The training loop also ran its built-in 28-image bbox eval at the end. That is secondary only. It reported bbox AP `0.8330`.

## Source First50 Sanity

External source first50 sanity was run after the smoke with the project wrapper, protocol checker, and PyTorch deform backend.

- Eval dir: `output/diagnostics/r90_supervised_32k1024_cache_teacher8499_smoke_original_first50_20260517`
- Protocol checker: `original_first50_teacher`, `--allow-nondefault-weights`, status `pass`
- Weights: `output/baseline/r90_supervised_32k1024_cache_teacher8499_smoke/checkpoint_iter_0000150.pth`
- Data: `magformer_datasets/20260318_1K_1566`, `annotations/instances_all.json`, split `all`, first 50 images
- Backend: `--force-pytorch-msda`

| Metric | AP | AP50 | AP75 | AP_small | AP_medium |
|---|---:|---:|---:|---:|---:|
| bbox | `0.574709` | `0.802785` | `0.604269` | `0.233615` | `0.725088` |
| segm | `0.527230` | `0.805543` | `0.604934` | `0.179843` | `0.665290` |

Artifacts are local and must not be committed:

- `output/diagnostics/r90_supervised_32k1024_cache_teacher8499_smoke_original_first50_20260517/eval.log`
- `output/diagnostics/r90_supervised_32k1024_cache_teacher8499_smoke_original_first50_20260517/metrics.cocoeval.json`
- `output/diagnostics/r90_supervised_32k1024_cache_teacher8499_smoke_original_first50_20260517/protocol_check.log`
- `output/diagnostics/r90_supervised_32k1024_cache_teacher8499_smoke_original_first50_20260517/protocol_check.summary.json`
