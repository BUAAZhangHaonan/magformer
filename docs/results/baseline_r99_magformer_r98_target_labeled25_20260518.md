# R99 MagFormer R98 Target-Labeled25 Finetune - 2026-05-18

Conclusion: stop at the 500 iter gate. R99 target_labeled25 supervised-only fine-tuning clearly improves R98 target_unlabeled200 from near zero, but external 1024 backmap topk/maxDets 200 segm AP is `0.082709`, below the `0.10` continue-to-1000 threshold. This is weak transfer, not a full recovery.

## Config

- Config: `configs/baseline_supervised_r99_magformer_r98ckpt_target_labeled25_1024.yaml`
- Base: R98 supervised-only MagFormer RGB-D config.
- Data root: `magformer_datasets/pseudo_real_512`
- Train ann: `annotations/instances_target_labeled.json`
- Eval ann: `annotations/instances_target_unlabeled.json`
- Warm start: `output/baseline/r98_magformer_32k_source_short/checkpoint_iter_0000499.pth`
- Output dir: `output/baseline/r99_magformer_r98ckpt_target_labeled25_1024`
- VC-SUDA, EMA, offline pseudo labels, contrastive, source retention, and unsupervised losses: disabled.
- Solver: `max_iter=500`, eval/checkpoint period `500`, image size `1024`.

## Pre-Train Sanity

Protocol checker for `pseudo_real_target_unlabeled200`: pass.

- image_count: `200`
- image_size: `1024`
- topk/maxDets: `200/200`
- depth: clip `0.0/2.095623016357422`, norm `minmax`, per-sample norm `true`
- summary: `output/diagnostics/r99_magformer_r98_target_labeled25_20260518/protocol_check_target_unlabeled200.json`

Loader/depth sanity: pass.

| split | images | annotations | raw RGB | raw depth | loader RGB | loader depth | mask count | depth mean/std | nonzero |
| --- | ---: | ---: | --- | --- | --- | --- | --- | --- | --- |
| target_labeled25 | 25 | 1697 | `[512,512,3]` | `[512,512]` | `[3,1024,1024]` | `[1,1024,1024]` | min 50, mean 67.88, max 100 | raw `0.948710/0.217926`, loader `0.948710/0.136446` | yes |
| target_unlabeled200 | 200 | 11750 | `[512,512,3]` | `[512,512]` | `[3,1024,1024]` | `[1,1024,1024]` | min 25, mean 60.95, max 100 | raw `0.948695/0.217847`, loader `0.948695/0.136379` | yes |

Static config and loader smoke: pass.

- `validate_config(strict=True)`: pass.
- Train dataset length: `25`, val dataset length: `200`.
- Train batch: images `[4,3,1024,1024]`, depths `[4,1,1024,1024]`.
- First smoke batch target mask counts: `[50,50,50,50]`.
- Val smoke batch: images `[4,3,512,512]`, depths `[4,1,512,512]`; external eval handles 1024 resize/backmap.

## Training

- tmux session: `r99_magformer_target_labeled25`
- Command shape: `torchrun --nproc_per_node=4 tools/train.py --config configs/baseline_supervised_r99_magformer_r98ckpt_target_labeled25_1024.yaml`
- GPUs: config GPUs `4,5,6,7`
- Warm-start load: `missing=0`, `unexpected=0`
- Train depth sanity: `should_abort=false`, depth min/max/mean/std `-0.002103/1.046874/0.948590/0.136993` after configured train noise.
- Final checkpoint: `output/baseline/r99_magformer_r98ckpt_target_labeled25_1024/checkpoint_iter_0000499.pth`
- Runtime: iter 500 eval logged at `2026-05-18T02:54:23+08:00`.

Built-in eval note: the training-loop eval reported target_unlabeled200 segm AP `0.016815` and bbox AP `0.026957`, but that path uses the trainer's COCO settings and is not the fixed external topk/maxDets 200 gate.

## External Gate Eval

Fixed protocol checker for the R99 checkpoint: pass.

Command shape:

```bash
CUDA_VISIBLE_DEVICES=4 MAGFORMER_MS_DEFORM_ATTN_BACKEND=pytorch \
/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python tools/evaluate_1024_backmap.py \
  --base-config configs/eval_vc_suda_stage_b_1024_teacher8499_segm.yaml \
  --dataset-root magformer_datasets/pseudo_real_512 \
  --ann annotations/instances_target_unlabeled.json \
  --split train \
  --weights output/baseline/r99_magformer_r98ckpt_target_labeled25_1024/checkpoint_iter_0000499.pth \
  --output-dir output/diagnostics/r99_magformer_r98_target_labeled25_iter0499_target_unlabeled200_1024_backmap_20260518 \
  --image-size 1024 \
  --batch-size 4 \
  --num-workers 0 \
  --score-threshold 0.05 \
  --mask-threshold 0.5 \
  --iou-types bbox,segm \
  --inference-topk 200 \
  --max-dets 200 \
  --dump-inference-stats output/diagnostics/r99_magformer_r98_target_labeled25_iter0499_target_unlabeled200_1024_backmap_20260518/inference_stats.json \
  --force-pytorch-msda
```

Metrics:

| eval | images | bbox AP | bbox AP50 | bbox AP75 | segm AP | segm AP50 | segm AP75 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| target_unlabeled200 external 1024 backmap topk/maxDets 200 | 200 | `0.124376` | `0.435706` | `0.031818` | `0.082709` | `0.325090` | `0.007090` |

Inference stats:

- Predictions exported: `22732`
- topk-truncated images: `0/200`
- score-filtered images: `200/200`
- mask-empty-filtered images: `3`

Gate decision: `0.02 <= segm AP 0.082709 < 0.10`, so stop at 500 iter. Do not continue to 1000 iter. Record as weak transfer.

## Overlay / Depth Diagnosis

Artifacts:

- Overlays: `output/diagnostics/r99_magformer_r98_target_labeled25_iter0499_target_unlabeled200_1024_backmap_20260518/overlays`
- Depth/prediction summary: `output/diagnostics/r99_magformer_r98_target_labeled25_iter0499_target_unlabeled200_1024_backmap_20260518/depth_prediction_diagnosis.json`

Summary:

- Depth is not collapsed: mean depth `0.948743`, mean std `0.217837`, minimum nonzero ratio `0.948845`.
- Prediction confidence is not the immediate bottleneck: score p50 `0.716488`, score p90 `0.940221`.
- Prediction density is high: pred count p50 `117`, p90 `138.1`, with no top-k truncation.
- Predicted box area is too large relative to GT: median pred/GT area ratio `1.6087x`.
- This points more to weak localization/scale after target_labeled25 fine-tune than to a depth-loader failure or top-k cap.

## Interpretation

R99 answers the R98 question: a small amount of target supervision moves target_unlabeled200 from near-zero AP to a measurable but weak result. The remaining miss is not protocol, depth shape, or top-k saturation. The main visible issue is coarse/oversized predictions and low AP75, even though AP50 is much better.

## Commits

- Config/docs placeholder: `7063206726229c849972d03b5dfac9b31f66b7b9`
- Final docs-only update: this docs-only commit.
