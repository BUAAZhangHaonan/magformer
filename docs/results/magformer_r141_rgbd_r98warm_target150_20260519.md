# R141 MagFormer RGB-D R98-Warm Target150

R141 is the RGB-D enabled matched control for R139.

## Purpose

- R139 RGB-only and R114 full RGB-D were close in formal metrics, but they did not share the same warm start.
- R141 uses the same R98 checkpoint as R139.
- R141 keeps the same target150 recipe as R139 and runs 2000 iterations.
- The only intended training difference from R139 is that `model.magformer.depth_backbone.enabled` and `model.magformer.modality_fusion.enabled` are both enabled.

## Matched Baseline

- R139 config: `configs/baseline_supervised_r139_magformer_rgb_only_r98warm_target150.yaml`
- R141 config: `configs/baseline_supervised_r141_magformer_rgbd_r98warm_target150.yaml`
- R98 checkpoint: `output/baseline/r98_magformer_32k_source_short/checkpoint_iter_0000499.pth`
- R141 final checkpoint: `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0002000.pth`
- Output dir: `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519`

## Formal Fixed1024 Evaluation

Both evals copy the R139 formal `tools/evaluate_1024_backmap.py` protocol and only change config, weights, and output dir to R141.

Eval dirs:

- Val28: `output/diagnostics/r141_rgbd_val28_1024_backmap_topk200_20260519`
- Remaining75: `output/diagnostics/r141_rgbd_remaining75_1024_backmap_topk200_20260519`

Evidence:

- Each eval dir contains `launch_command.sh`, `eval.log`, `eval_1024_runtime.yaml`, `coco_instances_results.json`, and `metrics.cocoeval.json`.
- Runtime config records `model.magformer.depth_backbone.enabled: true` and `model.magformer.modality_fusion.enabled: true`.
- Runtime config and logs load `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0002000.pth` with strict load OK.
- Logs show RGB-D inputs: `images=[..., 3, 1024, 1024]` and `depths=[..., 1, 1024, 1024]`.

Val28 command:

```bash
CUDA_VISIBLE_DEVICES=6,7 MAGFORMER_MS_DEFORM_ATTN_BACKEND=pytorch PYTHONUNBUFFERED=1 \
  /home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python tools/evaluate_1024_backmap.py \
  --base-config configs/baseline_supervised_r141_magformer_rgbd_r98warm_target150.yaml \
  --dataset-root magformer_datasets/pseudo_real_512 \
  --ann annotations/instances_val.json \
  --split val \
  --weights output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0002000.pth \
  --output-dir output/diagnostics/r141_rgbd_val28_1024_backmap_topk200_20260519 \
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
  --base-config configs/baseline_supervised_r141_magformer_rgbd_r98warm_target150.yaml \
  --dataset-root magformer_datasets/pseudo_real_512 \
  --ann annotations/instances_target_unlabeled_r114_balanced_minus125.json \
  --split train \
  --weights output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0002000.pth \
  --output-dir output/diagnostics/r141_rgbd_remaining75_1024_backmap_topk200_20260519 \
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

## Formal Metrics

| split | type | AP | AP50 | AP75 |
|---|---:|---:|---:|---:|
| val28 | bbox | 0.360386 | 0.715124 | 0.336700 |
| val28 | segm | 0.319539 | 0.634925 | 0.288992 |
| remaining75 | bbox | 0.431195 | 0.768302 | 0.446768 |
| remaining75 | segm | 0.388543 | 0.710741 | 0.393523 |

## Matched Delta vs R139 RGB-only

| split | type | AP delta | AP50 delta | AP75 delta |
|---|---:|---:|---:|---:|
| val28 | bbox | -0.001648 | +0.000938 | +0.005878 |
| val28 | segm | -0.003011 | +0.003143 | -0.004433 |
| remaining75 | bbox | +0.001542 | +0.001941 | +0.010398 |
| remaining75 | segm | +0.000320 | +0.006802 | -0.002101 |

## Delta vs R114 RGB-D Anchor

| split | type | AP delta | AP50 delta | AP75 delta |
|---|---:|---:|---:|---:|
| val28 | bbox | +0.003892 | +0.010909 | +0.015231 |
| val28 | segm | -0.002292 | +0.004173 | -0.006213 |
| remaining75 | bbox | +0.000899 | +0.012081 | +0.003561 |
| remaining75 | segm | -0.003630 | +0.007490 | -0.003896 |

## Conclusion

The strict matched R141 RGB-D vs R139 RGB-only comparison does not support a useful depth/fusion gain on the final mask metric.

- Val28 segm AP drops from R139 by `-0.003011`.
- Remaining75 segm AP is nearly tied, with only `+0.000320` over R139, while segm AP75 drops by `-0.002101`.
- Against R114, R141 is lower on segm AP for both val28 and remaining75.

Depth/fusion may raise AP50 slightly, but the formal fixed1024 mask AP/AP75 evidence is neutral to negative.
