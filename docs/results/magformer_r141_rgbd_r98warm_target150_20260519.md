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
- Output dir: `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519`

## Completion Standard

Training is only complete after the 2000-iteration run finishes and formal evaluation is run on:

- `val28`
- `remaining75`

The final comparison should be reported directly against R139 formal metrics:

- R139 val28 segm AP: 0.322550
- R139 remaining75 segm AP: 0.388223
