# R91 supervised-only 32K_1024 cache Teacher8499 low-LR smoke, 2026-05-17

R91 tests whether lowering the R90 supervised-only smoke LR from `1.0e-5` to `1.0e-6` preserves the original-source Teacher8499 anchor better after 150 iterations.

## Zero-Step Anchor

- Eval dir: `output/diagnostics/r91_anchor_teacher8499_original_first50_20260517`
- Weights: `output/experiments/20260510_1k_finetune_full_1024_v13/checkpoint_iter_0008499.pth`
- Protocol: `original_first50_teacher`
- Protocol checker: pass
- Data: `magformer_datasets/20260318_1K_1566`, `annotations/instances_all.json`, split `all`, first 50 images
- Backend: `--force-pytorch-msda`

| Metric | AP | AP50 | AP75 | AP_small | AP_medium |
|---|---:|---:|---:|---:|---:|
| bbox | `0.651953` | `0.853144` | `0.717547` | `0.303519` | `0.802621` |
| segm | `0.625293` | `0.867953` | `0.724339` | `0.263661` | `0.767420` |

The anchor reproduced the known Teacher8499 first50 line and passed the `>=0.61` gate, so R91 smoke may run.

## R90 Comparison

R90 smoke used the same supervised-only setup with `solver.base_lr: 1.0e-5`, `solver.max_iter: 150`, Teacher8499 warm start, cache train annotation, VC-SUDA disabled, runtime EMA disabled, contrastive loss disabled, and offline pseudo labels disabled.

R90 external source first50 after 150 iterations:

| Metric | AP | AP50 | AP75 | AP_small | AP_medium |
|---|---:|---:|---:|---:|---:|
| bbox | `0.574709` | `0.802785` | `0.604269` | `0.233615` | `0.725088` |
| segm | `0.527230` | `0.805543` | `0.604934` | `0.179843` | `0.665290` |

## R91 Design

- Config: `configs/baseline_supervised_r91_32k1024_cache_teacher8499_lr1e6_smoke.yaml`
- Base config: `configs/baseline_supervised_r90_32k1024_cache_teacher8499_smoke.yaml`
- Only semantic training variable: `solver.base_lr: 1.0e-6`
- Output dir: `output/baseline/r91_supervised_32k1024_cache_teacher8499_lr1e6_smoke`
- GPUs: `[4, 5, 6, 7]`
- Length: `solver.max_iter: 150`

The run keeps VC-SUDA disabled. Runtime EMA, contrastive loss, VC-SUDA EMA teacher, and offline pseudo labels stay disabled. `model.weights` and `runtime.resume` stay null, so the run is a model-only warm start from Teacher8499.

## Static Assertions

Static assertions passed before launch. The checks covered:

- Only semantic training variable changed from R90: `solver.base_lr` from `1.0e-5` to `1.0e-6`
- Experiment identity and output paths changed only for isolation
- `solver.max_iter == 150`
- `runtime.gpus == [4, 5, 6, 7]`
- `model.finetune_weights` exists: `output/experiments/20260510_1k_finetune_full_1024_v13/checkpoint_iter_0008499.pth`
- Cache exists: `magformer_datasets/20260318_1K_32254/cache/coco_loader/instances_train.sqlite`
- Output dir did not exist before launch
- `model.weights` and `runtime.resume` are null
- VC-SUDA, runtime EMA, contrastive loss, VC-SUDA EMA teacher, and offline pseudo labels are disabled

## Hard Stop

Stop immediately on missing Teacher8499 checkpoint, missing sqlite cache, existing output dir before launch, OOM, Traceback, non-finite loss, failed protocol checker, or an accidental run longer than 150 iterations.

R91 pass criterion after external original first50 eval: segm AP `>=0.58`. If the score is below `0.58`, the conclusion is that low LR is still insufficient and long training should not start.

## Run Log

R91 smoke was launched in tmux without `CUDA_VISIBLE_DEVICES`, so the config-selected physical GPUs `[4, 5, 6, 7]` were used directly.

- Tmux session: `r91_supervised_smoke`
- Train log: `output/logs/r91_supervised_32k1024_cache_teacher8499_lr1e6_smoke.log`
- Output dir: `output/baseline/r91_supervised_32k1024_cache_teacher8499_lr1e6_smoke`
- Checkpoint: `output/baseline/r91_supervised_32k1024_cache_teacher8499_lr1e6_smoke/checkpoint_iter_0000150.pth`
- Final rank0 train loss line: `iter=149/150`, `loss=7.9839`, `loss_ce=0.0304`, `loss_dice=0.0480`, `loss_mask=0.0518`
- Built-in 28-image bbox eval: AP `0.7075`
- Error scan: no `Traceback`, OOM, or non-finite loss in the completed run log

## R91 Source First50 Sanity

External source first50 sanity was run after the smoke with the project wrapper, protocol checker, and PyTorch deform backend.

- Eval dir: `output/diagnostics/r91_supervised_32k1024_cache_teacher8499_lr1e6_smoke_original_first50_20260517`
- Protocol checker: `original_first50_teacher`, `--allow-nondefault-weights`, status `pass`
- Weights: `output/baseline/r91_supervised_32k1024_cache_teacher8499_lr1e6_smoke/checkpoint_iter_0000150.pth`
- Data: `magformer_datasets/20260318_1K_1566`, `annotations/instances_all.json`, split `all`, first 50 images
- Backend: `--force-pytorch-msda`

| Metric | AP | AP50 | AP75 | AP_small | AP_medium |
|---|---:|---:|---:|---:|---:|
| bbox | `0.624154` | `0.834811` | `0.678567` | `0.278837` | `0.776992` |
| segm | `0.559725` | `0.838747` | `0.641859` | `0.199102` | `0.703864` |

## Conclusion

R91 does not pass the smoke gate. Source first50 segm AP is `0.559725`, below the `0.58` pass standard. Low LR improved over R90 segm AP `0.527230`, but it still did not preserve enough of the zero-step anchor segm AP `0.625293`. Do not start long training from this R91 design.

Artifacts are local and must not be committed:

- `output/logs/r91_supervised_32k1024_cache_teacher8499_lr1e6_smoke.log`
- `output/baseline/r91_supervised_32k1024_cache_teacher8499_lr1e6_smoke/`
- `output/diagnostics/r91_anchor_teacher8499_original_first50_20260517/`
- `output/diagnostics/r91_supervised_32k1024_cache_teacher8499_lr1e6_smoke_original_first50_20260517/`
