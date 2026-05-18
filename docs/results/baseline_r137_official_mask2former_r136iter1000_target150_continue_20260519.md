# R137 official Mask2Former target150 continuation

Date: 2026-05-19
Host: `4029` (`/home/hdd3/zhanghaonan/magformer`)
Branch: `feature/vc-suda-sim2real`
Base commit: `752afa0d732142504183dc22bb0e11953310772e`
Conda env: `mask2former`
TMUX session: `r137_m2f_g67`
Output: `output/baseline/r137_official_m2f_r136iter1000_target150_continue_1000_2000_g67`
Normalized COCO dir: `output/diagnostics/r137_official_m2f_target150_coco`

## Goal

Continue the official RGB Mask2Former target150 labeled-only baseline from R136 iter1000 to the schedule-matched 2000-iteration budget used by R114.

## Repo and checkpoint preflight

- Working tree was clean before launch.
- Proxy `git ls-remote` matched local `HEAD` at `752afa0d732142504183dc22bb0e11953310772e`.
- R136 seed checkpoint existed at `output/baseline/r136_official_m2f_r134iter4500_target150_continue_200_1000_g67/model_final.pth`.
- The copied R137 seed checkpoint had `iteration=999`, trainer iteration `999`, optimizer state, grad scaler state, and scheduler hook state.

## Continuation method

R137 was created as a new output directory seeded from the R136 final checkpoint:

```text
output/baseline/r136_official_m2f_r134iter4500_target150_continue_200_1000_g67/model_final.pth
```

The copied seed was written to `output/baseline/r137_official_m2f_r136iter1000_target150_continue_1000_2000_g67/model_final.pth`, with `last_checkpoint` set to `model_final.pth`. The launch used Detectron2 `--resume`, so trainer state was restored and training started at iteration 1000.

Verified log evidence:

```text
[Checkpointer] Loading from .../r137_official_m2f_r136iter1000_target150_continue_1000_2000_g67/model_final.pth ...
Loading trainer from .../r137_official_m2f_r136iter1000_target150_continue_1000_2000_g67/model_final.pth ...
Loading scheduler from state_dict ...
Starting training from iteration 1000
```

## Schedule

```text
CUDA_VISIBLE_DEVICES=6,7
--num-gpus 2
SOLVER.MAX_ITER 2000
SOLVER.IMS_PER_BATCH 4
SOLVER.CHECKPOINT_PERIOD 200
TEST.EVAL_PERIOD 200
DATASETS.TRAIN ("eccpseudo_real_512_target150_labeled",)
DATASETS.TEST ("eccpseudo_real_512_val28",)
MODEL.MASK_FORMER.NUM_OBJECT_QUERIES 100
TEST.DETECTIONS_PER_IMAGE 100
MODEL.WEIGHTS output/baseline/r136_official_m2f_r134iter4500_target150_continue_200_1000_g67/model_final.pth
```

Exact launch files:

```text
output/baseline/r137_official_m2f_r136iter1000_target150_continue_1000_2000_g67/launch_command.sh
output/baseline/r137_official_m2f_r136iter1000_target150_continue_1000_2000_g67/tmux_command.txt
```

TMUX command:

```bash
tmux new-session -d -s r137_m2f_g67 "cd /home/hdd3/zhanghaonan/magformer && bash output/baseline/r137_official_m2f_r136iter1000_target150_continue_1000_2000_g67/launch_command.sh 2>&1 | tee output/baseline/r137_official_m2f_r136iter1000_target150_continue_1000_2000_g67/tmux_train.log"
```

## GPU evidence

The launch snapshot records:

```text
CUDA_VISIBLE_DEVICES=6,7
logical_num_gpus=2
```

Runtime `nvidia-smi` after launch showed new R137 Python workers on physical GPUs 6 and 7 only. Pre-existing `safa` jobs remained on physical GPUs 4 and 5 and were not interrupted. The post-run snapshot showed GPUs 6 and 7 back at 1 MiB, while the same GPUs 4 and 5 jobs were still present.

## Result

Status: finished.

- Resume start: iteration 1000.
- Final training iter: 1999, so 2000 total iterations in the Detectron2 schedule.
- Final checkpoint: `model_final.pth` with `iteration=1999` and trainer iteration `1999`.
- Checkpoints: `model_0001199.pth`, `model_0001399.pth`, `model_0001599.pth`, `model_0001799.pth`, `model_0001999.pth`, `model_final.pth`.
- Final logged total loss at iter 1999: `17.96`.
- Total training time: `0:10:57`.
- Final val28 built-in eval ran at iteration 2000.

Val28 built-in eval:

| eval iter | bbox AP | bbox AP50 | bbox AP75 | segm AP | segm AP50 | segm AP75 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1199 | 21.4685 | 51.9419 | 13.7216 | 16.8312 | 43.0164 | 10.4208 |
| 1399 | 20.5280 | 49.8945 | 12.2977 | 17.9989 | 42.6292 | 12.2577 |
| 1599 | 20.5428 | 49.9275 | 12.1175 | 15.7735 | 41.1446 | 9.2897 |
| 1799 | 22.6674 | 53.0881 | 15.6654 | 17.9605 | 44.1774 | 11.4671 |
| 2000 | 22.4821 | 52.8471 | 15.5031 | 18.7643 | 44.5556 | 12.8638 |

Compared with R136 final val28 built-in eval (`bbox AP 20.6213`, `segm AP 17.8559`), R137 improves the schedule-matched RGB labeled-only baseline to `bbox AP 22.4821`, `segm AP 18.7643` under the original 800/1333 test-size setting.

## Fixed1024 official eval plus source-RLE replay

A follow-up eval-only pass reran the R137 final checkpoint with the same fixed1024 protocol used for the R136 fair row. It used `CUDA_VISIBLE_DEVICES=6,7`, `--eval-only`, `INPUT.MIN_SIZE_TEST=1024`, `INPUT.MAX_SIZE_TEST=1024`, `MODEL.MASK_FORMER.NUM_OBJECT_QUERIES=100`, and `TEST.DETECTIONS_PER_IMAGE=100`.

Artifacts:

```text
output/diagnostics/r137_official_m2f_val28_fixed1024_eval/eval.log
output/diagnostics/r137_official_m2f_val28_fixed1024_eval/eval_command.sh
output/diagnostics/r137_official_m2f_val28_fixed1024_eval/inference/coco_instances_results.json
output/diagnostics/r137_official_m2f_val28_fixed1024_eval/source_rle_replay_metrics.cocoeval.json
output/diagnostics/r137_official_m2f_remaining75_fixed1024_eval/eval.log
output/diagnostics/r137_official_m2f_remaining75_fixed1024_eval/eval_command.sh
output/diagnostics/r137_official_m2f_remaining75_fixed1024_eval/inference/coco_instances_results.json
output/diagnostics/r137_official_m2f_remaining75_fixed1024_eval/source_rle_replay_metrics.cocoeval.json
```

The eval logs show `resume=False, eval_only=True`, fixed `MIN_SIZE_TEST: 1024`, fixed `MAX_SIZE_TEST: 1024`, and `Start inference`. Grep found no `Starting training` line in either fixed1024 eval log. The eval commands pin physical GPUs 6 and 7 through `CUDA_VISIBLE_DEVICES=6,7`; pre/post `nvidia-smi` snapshots show the pre-existing `safa` jobs still on physical GPUs 4 and 5 and GPUs 6 and 7 back at 1 MiB after eval.

Built-in fixed1024 official eval:

| split | images | bbox AP | bbox AP50 | bbox AP75 | segm AP | segm AP50 | segm AP75 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| val28 | 28 | 27.1572 | 59.4833 | 20.4662 | 24.9466 | 54.0895 | 20.2101 |
| remaining75 | 75 | 34.2840 | 68.2626 | 29.8093 | 32.2779 | 62.4539 | 29.8498 |

Source-RLE replay of the saved COCO JSON, with empty masks dropped and bboxes recomputed from RLE masks in memory:

| split | input preds | dropped empty | kept preds | maxDets | bbox AP | bbox AP50 | bbox AP75 | segm AP | segm AP50 | segm AP75 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| val28 | 2800 | 393 | 2407 | 100 | 27.1572 | 59.4833 | 20.4662 | 25.3215 | 54.1955 | 20.4993 |
| val28 | 2800 | 393 | 2407 | 200 | 27.1572 | 59.4833 | 20.4662 | 25.3215 | 54.1955 | 20.4993 |
| remaining75 | 7500 | 989 | 6511 | 100 | 34.2840 | 68.2626 | 29.8093 | 32.4278 | 63.4791 | 29.7099 |
| remaining75 | 7500 | 989 | 6511 | 200 | 34.2840 | 68.2626 | 29.8093 | 32.4278 | 63.4791 | 29.7099 |

Current RGB baseline gap versus R136 and R114:

| split | model/protocol | bbox AP | segm AP | segm AP delta vs R136 replay | segm AP gap vs R114 |
| --- | --- | ---: | ---: | ---: | ---: |
| val28 | R136 fixed1024 source-RLE replay maxDets100/200 | 24.1935 | 22.6913 | reference | -9.4918 |
| val28 | R137 fixed1024 source-RLE replay maxDets100/200 | 27.1572 | 25.3215 | +2.6303 | -6.8615 |
| val28 | R114 MagFormer 1024 backmap topk200 | 35.6494 | 32.1831 | n/a | reference |
| remaining75 | R136 fixed1024 source-RLE replay maxDets100/200 | 30.7500 | 29.0370 | reference | -10.1802 |
| remaining75 | R137 fixed1024 source-RLE replay maxDets100/200 | 34.2840 | 32.4278 | +3.3907 | -6.7895 |
| remaining75 | R114 MagFormer 1024 backmap topk200 | 43.0296 | 39.2173 | n/a | reference |

R137 narrows the official RGB labeled-only gap, but it remains below R114 on both target splits. Replay `maxDets=200` is still identical to `maxDets=100` because the official prediction JSON is capped at 100 detections per image.
