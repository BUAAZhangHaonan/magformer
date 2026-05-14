# VC-SUDA Stage C Final Eval - 2026-05-14

## Conclusion

- Stage C final is degraded. Fresh eval of `checkpoint_iter_0009000.pth` gives `bbox_AP=0.0887` and `segm_AP=0.0665`, far below Stage B post eval (`bbox_AP=0.1637`, `segm_AP=0.1128`).
- Stage C mid at iter 3999 was only slightly below Stage B (`bbox_AP -0.0061`, `segm_AP -0.0046`). The large drop happens between mid and final.
- Among final files, iter 8999 has the best `segm_AP` by a very small margin (`0.06709`), while 9000 and `model_best.pth` produce identical fresh eval metrics and prediction counts.

## Scope

- Agent: CC.
- Host: `WS-4029GP-TRT` via `ssh 4029`.
- Repository: `/home/hdd3/zhanghaonan/magformer`.
- Branch: `feature/vc-suda-sim2real`.
- HEAD during report: `ae519fb`.
- Conda env: `magformer`.
- Stage C output dir: `output/experiments/vc_suda_stage_c_1024_teacher8499_20260514_0734`.
- Fresh eval output dir: `output/experiments/20260514_stage_c_final_eval`.
- Eval config: `configs/eval_vc_suda_stage_b_1024_teacher8499_segm.yaml`.
- Eval IoU types: `bbox`, `segm`.
- Eval GPU: physical GPU 0, `CUDA_VISIBLE_DEVICES=0`. GPU 4-7 were not used.
- No new training was started. No branch or worktree was created. No rollback was done.

## Checkpoints

| File | Size | SHA256 |
| --- | ---: | --- |
| `checkpoint_iter_0008999.pth` | 747.2 MiB | `d18d63f475ccb33e440d1a58fec36be2ae48da9f961dd1902e7b396d1c7355a0` |
| `checkpoint_iter_0009000.pth` | 747.2 MiB | `d4ab9cad086c75001129d20b8b8bccc5e9924edab0f666899e0ef91101137ad1` |
| `model_best.pth` | 747.2 MiB | `9ee955d881057f87fa8c7d6c2f880b9e8f86348a3e5656d8fe4165485455e659` |

Notes: `checkpoint_iter_0009000.pth` and `model_best.pth` have different SHA256 values, so both were evaluated. Their fresh eval outputs are identical.

## Fresh Final Eval

| Checkpoint | bbox AP | bbox AP50 | bbox AP75 | segm AP | segm AP50 | segm AP75 | Predictions | Images |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `checkpoint_iter_0008999` | 0.0874 | 0.2312 | 0.0546 | 0.0671 | 0.1929 | 0.0305 | 665 | 28 |
| `checkpoint_iter_0009000` | 0.0887 | 0.2315 | 0.0548 | 0.0665 | 0.1930 | 0.0303 | 669 | 28 |
| `model_best` | 0.0887 | 0.2315 | 0.0548 | 0.0665 | 0.1930 | 0.0303 | 669 | 28 |

The training-time final eval row in `metrics_log.jsonl` is consistent with the fresh eval: `bbox_AP=0.0890`, `segm_AP=0.0671` at iter 9000.

## Stage Comparison

| Eval | Checkpoint | bbox AP | bbox AP50 | bbox AP75 | segm AP | segm AP50 | segm AP75 | Predictions |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage B post eval | `stage_b checkpoint_iter_0008999` | 0.1637 | 0.4824 | 0.0727 | 0.1128 | 0.3810 | 0.0350 | 1700 |
| Stage C mid eval | `checkpoint_iter_0003999` | 0.1576 | 0.4679 | 0.0710 | 0.1082 | 0.3649 | 0.0356 | 1604 |
| Stage C final checkpoint_iter_0008999 | `checkpoint_iter_0008999` | 0.0874 | 0.2312 | 0.0546 | 0.0671 | 0.1929 | 0.0305 | 665 |
| Stage C final checkpoint_iter_0009000 | `checkpoint_iter_0009000` | 0.0887 | 0.2315 | 0.0548 | 0.0665 | 0.1930 | 0.0303 | 669 |
| Stage C final model_best | `model_best` | 0.0887 | 0.2315 | 0.0548 | 0.0665 | 0.1930 | 0.0303 | 669 |

## Degradation Check

- Stage C mid vs Stage B post: `bbox_AP -0.0061` (-3.7%), `segm_AP -0.0046` (-4.1%).
- Stage C final 9000 vs Stage B post: `bbox_AP -0.0751` (-45.8%), `segm_AP -0.0463` (-41.1%).
- Stage C final 9000 vs Stage C mid: `bbox_AP -0.0690` (-43.8%), `segm_AP -0.0417` (-38.5%).

This is a real final-stage regression, not only noise around the last checkpoint. The final prediction count also dropped from Stage B post eval `1700` to Stage C final `669`, while all 28 validation images still had predictions.

## Training Trend

| Iter window | Samples | loss mean | CE mean | Dice mean | Mask mean | Pseudo total mean | LR first -> last |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0-3000 | 151 | 46.0126 | 0.0638 | 0.1979 | 0.0759 | 26.3783 | 1.000e-07 -> 7.833e-05 |
| 3020-6000 | 150 | 42.2153 | 0.0564 | 0.1866 | 0.0753 | 15.8740 | 7.803e-05 -> 2.688e-05 |
| 6020-8980 | 149 | 39.6107 | 0.0490 | 0.1750 | 0.0722 | 9.5696 | 2.656e-05 -> 5.480e-09 |

- The logged train loss trends down from `46.01` to `39.61` by window average.
- The pseudo total term drops strongly from `26.38` to `9.57`.
- The validation AP still drops sharply by the final checkpoint, so the loss trend does not translate into validation AP improvement.

## Artifacts

- `checkpoint_iter_0008999` eval log: `output/experiments/20260514_stage_c_final_eval/checkpoint_iter_0008999_segm/eval_gpu0.log`
- `checkpoint_iter_0008999` COCO predictions: `output/experiments/20260514_stage_c_final_eval/checkpoint_iter_0008999_segm/coco_instances_results.json`
- `checkpoint_iter_0009000` eval log: `output/experiments/20260514_stage_c_final_eval/checkpoint_iter_0009000_segm/eval_gpu0.log`
- `checkpoint_iter_0009000` COCO predictions: `output/experiments/20260514_stage_c_final_eval/checkpoint_iter_0009000_segm/coco_instances_results.json`
- `model_best` eval log: `output/experiments/20260514_stage_c_final_eval/model_best_segm/eval_gpu0.log`
- `model_best` COCO predictions: `output/experiments/20260514_stage_c_final_eval/model_best_segm/coco_instances_results.json`
- Metrics JSON: `docs/results/vc_suda_stage_c_final_eval_20260514_metrics.json`

## Status

- Fresh bbox+segm eval completed for 8999, 9000, and `model_best.pth`.
- Stage C final is worse than Stage B post eval and worse than Stage C mid eval on both bbox AP and segm AP.
- Residual risk: the validation split is small (`28` images), so exact AP values can move with small prediction changes. The direction of the final regression is large enough to call out clearly.
