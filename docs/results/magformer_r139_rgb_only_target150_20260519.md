# R139 MagFormer RGB-only Target150 Launch - 2026-05-19

Date: 2026-05-19
Host: `4029` (`/home/hdd3/zhanghaonan/magformer`)
Branch: `feature/vc-suda-sim2real`

## Current Inputs

- R138 full 32K source validation finished from the R98 checkpoint: bbox AP `0.7424`, segm AP `0.7839`.
- R139 smoke config: `configs/baseline_supervised_r139_magformer_rgb_only_r98warm_target150_smoke.yaml`.
- R139 smoke output: `output/baseline/r139_magformer_rgb_only_r98warm_target150_smoke_20260519`.
- R139 smoke loaded the R98 warm-start with missing keys `0` and unexpected keys `0`, then completed `20` iterations.

## Formal R139 Config

Formal config: `configs/baseline_supervised_r139_magformer_rgb_only_r98warm_target150.yaml`

Held fixed from the R114 target150 protocol:

- Train annotation: `annotations/instances_target_labeled_r114_balanced_plus125.json`.
- Val annotation: `annotations/instances_val.json`.
- Warm-start checkpoint: `output/baseline/r98_magformer_32k_source_short/checkpoint_iter_0000499.pth`.
- Image size: `1024`.
- RGB photo augmentation disabled.
- Depth noise disabled.
- `model.magformer.depth_backbone.enabled: false`.
- `model.magformer.modality_fusion.enabled: false`.
- `solver.max_iter: 2000`.
- Output: `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519`.

## Launch Command

```bash
CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.run --standalone --nproc_per_node=2 tools/train.py --config configs/baseline_supervised_r139_magformer_rgb_only_r98warm_target150.yaml --gpus 0,1
```

Launch wrapper path after start:

```text
output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/launch_command.sh
```

Run log path after start:

```text
output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/run.log
```

## Post-training Evaluation Target

After the 2000-iteration run completes, evaluate the final R139 checkpoint on the same fixed1024 target protocols used to compare R114 and R137:

- `val28`, fixed1024.
- `remaining75`, fixed1024, using `annotations/instances_target_unlabeled_r114_balanced_minus125.json`.

The comparison should report bbox and segm AP, with segm AP as the primary row for alignment with the R114/R137 target150 tables.

## Formal Fixed1024 Eval

Final checkpoint:

```text
output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0002000.pth
```

Protocol:

- Entry point: `tools/evaluate_1024_backmap.py`.
- Model input: fixed `1024`.
- Backmap to COCO ground-truth size.
- IoU types: `bbox,segm`.
- Score threshold: `0.05`.
- Mask threshold: `0.5`.
- Inference top-k: `200`.
- COCO maxDets: `200`.
- MSDA backend: `MAGFORMER_MS_DEFORM_ATTN_BACKEND=pytorch` with `--force-pytorch-msda`.
- Runtime config confirms `model.magformer.depth_backbone.enabled: false`.
- Runtime config confirms `model.magformer.modality_fusion.enabled: false`.

Eval outputs:

- Val28: `output/diagnostics/r139_rgb_only_val28_1024_backmap_topk200_20260519`.
- Remaining75: `output/diagnostics/r139_rgb_only_remaining75_1024_backmap_topk200_20260519`.
- Each directory contains `launch_command.sh`, `eval.log`, `eval_1024_runtime.yaml`, `coco_instances_results.json`, and `metrics.cocoeval.json`.

Val28 command:

```bash
CUDA_VISIBLE_DEVICES=6,7 MAGFORMER_MS_DEFORM_ATTN_BACKEND=pytorch PYTHONUNBUFFERED=1 \
  /home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python tools/evaluate_1024_backmap.py \
  --base-config configs/baseline_supervised_r139_magformer_rgb_only_r98warm_target150.yaml \
  --dataset-root magformer_datasets/pseudo_real_512 \
  --ann annotations/instances_val.json \
  --split val \
  --weights output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0002000.pth \
  --output-dir output/diagnostics/r139_rgb_only_val28_1024_backmap_topk200_20260519 \
  --image-size 1024 \
  --batch-size 4 \
  --num-workers 2 \
  --score-threshold 0.05 \
  --mask-threshold 0.5 \
  --inference-topk 200 \
  --max-dets 200 \
  --iou-types bbox,segm \
  --force-pytorch-msda
```

Remaining75 command:

```bash
CUDA_VISIBLE_DEVICES=6,7 MAGFORMER_MS_DEFORM_ATTN_BACKEND=pytorch PYTHONUNBUFFERED=1 \
  /home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python tools/evaluate_1024_backmap.py \
  --base-config configs/baseline_supervised_r139_magformer_rgb_only_r98warm_target150.yaml \
  --dataset-root magformer_datasets/pseudo_real_512 \
  --ann annotations/instances_target_unlabeled_r114_balanced_minus125.json \
  --split train \
  --weights output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0002000.pth \
  --output-dir output/diagnostics/r139_rgb_only_remaining75_1024_backmap_topk200_20260519 \
  --image-size 1024 \
  --batch-size 4 \
  --num-workers 2 \
  --score-threshold 0.05 \
  --mask-threshold 0.5 \
  --inference-topk 200 \
  --max-dets 200 \
  --iou-types bbox,segm \
  --force-pytorch-msda
```

Results compared to R114 final fixed1024 formal eval:

| split | metric | R139 | R114 | delta |
| --- | --- | ---: | ---: | ---: |
| val28 | bbox AP | 0.362034 | 0.356494 | +0.005540 |
| val28 | bbox AP50 | 0.714186 | 0.704215 | +0.009971 |
| val28 | bbox AP75 | 0.330822 | 0.321469 | +0.009353 |
| val28 | segm AP | 0.322550 | 0.321831 | +0.000720 |
| val28 | segm AP50 | 0.631782 | 0.630752 | +0.001030 |
| val28 | segm AP75 | 0.293425 | 0.295205 | -0.001780 |
| remaining75 | bbox AP | 0.429653 | 0.430296 | -0.000643 |
| remaining75 | bbox AP50 | 0.766361 | 0.756221 | +0.010140 |
| remaining75 | bbox AP75 | 0.436370 | 0.443207 | -0.006837 |
| remaining75 | segm AP | 0.388223 | 0.392173 | -0.003950 |
| remaining75 | segm AP50 | 0.703940 | 0.703252 | +0.000688 |
| remaining75 | segm AP75 | 0.395624 | 0.397419 | -0.001796 |

Conclusion: R139 RGB-only is essentially tied with R114 on val28 segm AP (`+0.000720`) and slightly lower on remaining75 segm AP (`-0.003950`). It improves val28 bbox AP and AP50 on both splits, but does not improve the primary remaining75 segm AP row.
