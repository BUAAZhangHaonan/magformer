# VC-SUDA R59 Teacher8499 retention-min result - 2026-05-16

## Conclusion

R59 fails the retention gate.

It started from healthy Teacher8499 and ran the intended minimal Stage B absorption step, but it did not retain source sanity and it did not keep target performance. Original first50 source segm AP is `0.4731567886`, below the required `0.55`. Target_unlabeled200 segm AP is `0.0020426278`, below the required `0.30` and far below the R12 target reference `0.320048`.

## Setup

- Config: `configs/vc_suda_stage_b_r59_teacher8499_retention_min_1024.yaml`
- Config commit: `ec10124193f7ba9219fd9508a64ad4ad88e027c5`
- Base config: `configs/vc_suda_stage_b_1024_teacher8499.yaml`
- Warm-start: `output/experiments/20260510_1k_finetune_full_1024_v13/checkpoint_iter_0008499.pth`
- Output dir: `output/vc_suda/vc_suda_stage_b_r59_teacher8499_retention_min_1024`
- Training tmux: `r59_retention_min_20260516`
- Training log: `output/vc_suda/vc_suda_stage_b_r59_teacher8499_retention_min_1024/logs/train_r59_tmux.log`
- Final checkpoint: `output/vc_suda/vc_suda_stage_b_r59_teacher8499_retention_min_1024/checkpoint_iter_0000250.pth`

## Config Checks

- `data.dataset_root: magformer_datasets/pseudo_real_512`
- `data.train_ann: annotations/instances_source.json`
- `data.val_ann: annotations/instances_val.json`
- fixed LSJ scale `1.0/1.0`
- `model.finetune_weights` set to Teacher8499
- `runtime.resume: null`
- `runtime.ema_enabled: false`
- `runtime.eval_period/checkpoint_period: 250/250`
- `runtime.eval_iou_types: [bbox, segm]`
- `runtime.eval_saves_best: false`
- `runtime.checkpoint_max_keep: null`
- `solver.base_lr: 1.0e-6`
- `solver.max_iter: 250`
- `solver.backbone_multiplier: 0.25`
- `vc_suda.stage: B`
- `vc_suda.target_labeled_weight: 0.05`
- `vc_suda.unsupervised_weight: 0.0`
- `vc_suda.ema_teacher.enabled: false`
- domain adaptation weights set to `0.0`
- no `freeze_modules`

## Test Verification

Command used the existing conda env python:

```bash
/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python -m pytest \
  tests/test_vc_suda_data_protocol.py::test_vc_suda_target_labeled_weight_defaults_and_validates \
  tests/test_vc_suda_trainer_runtime.py::test_stage_b_target_labeled_weight_scales_total_loss \
  tests/test_vc_suda_trainer_runtime.py::test_stage_b_target_labeled_weight_metrics_are_logged \
  tests/test_train_warm_start.py \
  tests/test_optimizer_groups.py \
  tests/test_teacher_first50_protocol.py -q
```

Result: `19 passed`. Warnings were existing pydantic deprecation warnings and missing SHA256 sidecar warnings for test checkpoints / Teacher8499.

## Training

Training ran in tmux on GPUs 4-7 with:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
MAGFORMER_MS_DEFORM_ATTN_BACKEND=cuda \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 \
  tools/train.py \
  --config configs/vc_suda_stage_b_r59_teacher8499_retention_min_1024.yaml \
  --gpus 0,1,2,3 \
  --num-workers 2
```

Training exited `0`. Wall time from `/usr/bin/time` was `8:50.12`. Training log reported warm-start missing/unexpected `0/0` on all ranks. External strict eval load later matched `774/774` keys with missing/unexpected/shape mismatch `0/0/0`. Server RAM stayed below 90%; sampled post-run memory was about `15 GiB / 251 GiB` used.

The training-time built-in 28-image eval is diagnostic only. It reported bbox AP `0.0051` and segm AP `0.0021`.

## External Eval

### Original first50 source

- Protocol checker: `output/diagnostics/r59_retention_min_ckpt0250_original_first50_20260516/protocol_summary.json`
- Eval dir: `output/diagnostics/r59_retention_min_ckpt0250_original_first50_20260516`
- Eval log: `output/diagnostics/r59_retention_min_ckpt0250_original_first50_20260516/eval.log`
- Metrics: `output/diagnostics/r59_retention_min_ckpt0250_original_first50_20260516/metrics.cocoeval.json`
- Predictions: `4287`

| Metric | Value |
|---|---:|
| bbox_AP | `0.4868131007` |
| bbox_AP50 | `0.7650881817` |
| bbox_AP75 | `0.5072273989` |
| segm_AP | `0.4731567886` |
| segm_AP50 | `0.7720989338` |
| segm_AP75 | `0.5051846966` |

### Target_unlabeled200

- Protocol checker: `output/diagnostics/r59_retention_min_ckpt0250_target_unlabeled200_20260516/protocol_summary.json`
- Eval dir: `output/diagnostics/r59_retention_min_ckpt0250_target_unlabeled200_20260516`
- Eval log: `output/diagnostics/r59_retention_min_ckpt0250_target_unlabeled200_20260516/eval.log`
- First failed backend log: `output/diagnostics/r59_retention_min_ckpt0250_target_unlabeled200_20260516/eval_failed_cuda_backend.log`
- Metrics: `output/diagnostics/r59_retention_min_ckpt0250_target_unlabeled200_20260516/metrics.cocoeval.json`
- Predictions: `24196`

The first target eval attempt failed before inference because `MAGFORMER_MS_DEFORM_ATTN_BACKEND=cuda` requested an unavailable MSDeformAttn CUDA extension. No code was changed. The eval was rerun with the same fixed target protocol and `MAGFORMER_MS_DEFORM_ATTN_BACKEND=pytorch`, matching the established external eval path.

| Metric | Value |
|---|---:|
| bbox_AP | `0.0024240086` |
| bbox_AP50 | `0.0086083490` |
| bbox_AP75 | `0.0012376238` |
| segm_AP | `0.0020426278` |
| segm_AP50 | `0.0080003114` |
| segm_AP75 | `0.0012376238` |

## Gate Decision

Hard pass requires both original first50 source segm AP `>= 0.55` and target_unlabeled200 segm AP `>= 0.30`.

R59 gets:

- original first50 source segm AP `0.4731567886`: fail by `0.0768432114`
- target_unlabeled200 segm AP `0.0020426278`: fail by `0.2979573722`

Promotion requires target_unlabeled200 approaching or exceeding R12 `0.320048` while source first50 stays `>= 0.55`. R59 is not promotable.
