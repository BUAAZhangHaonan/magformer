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
