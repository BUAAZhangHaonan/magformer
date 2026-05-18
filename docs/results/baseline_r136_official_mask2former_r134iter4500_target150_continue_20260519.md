# R136 official Mask2Former target150 continuation

Date: 2026-05-19
Host: `4029` (`/home/hdd3/zhanghaonan/magformer`)
Branch: `feature/vc-suda-sim2real`
Conda env: `mask2former`
TMUX session: `r136_m2f_g67`
Output: `output/baseline/r136_official_m2f_r134iter4500_target150_continue_200_1000_g67`
Normalized COCO dir: `output/diagnostics/r136_official_m2f_target150_coco`

## Goal

Continue R135 from a 200-iter smoke into a finite, more useful RGB labeled-only baseline checkpoint. This run uses the same target150 train split and val28 eval as R135. No remaining75 eval was launched during training; the follow-up eval-only remaining75 run is recorded below.

## Continuation method

R135 was not overwritten. R136 was created as a new output dir seeded with:

```text
output/baseline/r135_official_m2f_r134iter4500_target150_smoke_g67/model_final.pth
```

The R135 checkpoint contains `iteration=199` and trainer state under `trainer._trainer`, including optimizer and grad scaler state. R136 has its own `last_checkpoint` pointing to the copied `model_final.pth`, and the launch uses Detectron2 `--resume`.

Verified log evidence:

```text
[Checkpointer] Loading from .../r136_official_m2f_r134iter4500_target150_continue_200_1000_g67/model_final.pth ...
Loading trainer from .../r136_official_m2f_r134iter4500_target150_continue_200_1000_g67/model_final.pth ...
Loading scheduler from state_dict ...
Starting training from iteration 200
```

## Schedule

```text
CUDA_VISIBLE_DEVICES=6,7
--num-gpus 2
SOLVER.MAX_ITER 1000
SOLVER.IMS_PER_BATCH 4
SOLVER.CHECKPOINT_PERIOD 200
TEST.EVAL_PERIOD 200
DATASETS.TRAIN ("eccpseudo_real_512_target150_labeled",)
DATASETS.TEST ("eccpseudo_real_512_val28",)
MODEL.WEIGHTS /home/hdd3/zhanghaonan/magformer/output/baseline/r135_official_m2f_r134iter4500_target150_smoke_g67/model_final.pth
```

Exact launch files:

```text
output/baseline/r136_official_m2f_r134iter4500_target150_continue_200_1000_g67/launch_command.sh
output/baseline/r136_official_m2f_r134iter4500_target150_continue_200_1000_g67/tmux_command.txt
```

TMUX command:

```bash
tmux new-session -d -s r136_m2f_g67 "cd /home/hdd3/zhanghaonan/magformer && bash output/baseline/r136_official_m2f_r134iter4500_target150_continue_200_1000_g67/launch_command.sh 2>&1 | tee output/baseline/r136_official_m2f_r134iter4500_target150_continue_200_1000_g67/tmux_train.log"
```

## GPU evidence

The environment snapshot records:

```text
CUDA_VISIBLE_DEVICES=6,7
logical_num_gpus=2
```

Runtime `nvidia-smi` after launch showed the R136 Python workers on physical GPUs 6 and 7 only. Separate pre-existing jobs were present on GPUs 4 and 5 and were not interrupted.

## Result

Status: finished.

- Resume start: iteration 200.
- Final training iter: 999, so 1000 total iterations in the Detectron2 schedule.
- Checkpoints: `model_0000399.pth`, `model_0000599.pth`, `model_0000799.pth`, `model_0000999.pth`, `model_final.pth`.
- Final logged total loss at iter 999: `20.6802`.
- Total training time: `0:08:34`.
- Final val28 eval ran at iteration 1000.

Val28 built-in eval:

| eval iter | bbox AP | bbox AP50 | bbox AP75 | segm AP | segm AP50 | segm AP75 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 399 | 19.8356 | 50.0702 | 12.3068 | 15.0090 | 38.9332 | 8.2441 |
| 599 | 19.3703 | 49.8989 | 11.4645 | 16.4421 | 41.5983 | 9.7960 |
| 799 | 21.1991 | 50.9829 | 14.0884 | 16.6258 | 41.9193 | 9.4113 |
| 1000 | 20.6213 | 51.2346 | 11.4854 | 17.8559 | 44.1671 | 11.0509 |

Compared with R135 smoke val28 (`bbox AP 16.9757`, `segm AP 12.9465`), R136 improves the finite RGB labeled-only baseline signal while keeping the same target150 and val28 protocol.

## Remaining75 eval-only

Follow-up eval-only run used the R136 final checkpoint and the official wrapper style. It is still official built-in COCO eval with `MODEL.MASK_FORMER.NUM_OBJECT_QUERIES=100` and `TEST.DETECTIONS_PER_IMAGE=100`, so it is not the later fair topk200/maxDets200 wrapper result.

Command shape:

```bash
cd /home/hdd3/zhanghaonan/magformer
source ~/anaconda3/etc/profile.d/conda.sh
conda activate mask2former
CUDA_VISIBLE_DEVICES=6,7 PYTHONUNBUFFERED=1 python -u - <<PYCODE
# fork shim around baselines/run_official_mask2former_ecc.py
# passthrough includes --eval-only --num-gpus 2
# DATASETS.TEST=("eccpseudo_real_512_remaining75",)
# MODEL.WEIGHTS=output/baseline/r136_official_m2f_r134iter4500_target150_continue_200_1000_g67/model_final.pth
PYCODE
```

Full argv and logs are in:

```text
output/diagnostics/r136_official_m2f_remaining75_eval/eval.log
output/diagnostics/r136_official_m2f_remaining75_eval/config.yaml
output/diagnostics/r136_official_m2f_remaining75_eval/inference/coco_instances_results.json
output/diagnostics/r136_official_m2f_remaining75_eval/inference/instances_predictions.pth
output/diagnostics/r136_official_m2f_remaining75_coco/instances_target_unlabeled_r114_balanced_minus125.67fe4a45ce.normalized.json
```

GPU and no-training evidence:

```text
CUDA_VISIBLE_DEVICES=6,7
CONDA_DEFAULT_ENV=mask2former
eval_only=1
logical_num_gpus=2
Command Line Args: ... resume=False, eval_only=True, num_gpus=2 ...
Start inference on 38 batches
```

The eval log contains no `Starting training` line. The mask probe file confirms logical devices under `CUDA_VISIBLE_DEVICES=6,7` map to physical GPUs 6 and 7, while pre-existing jobs stayed on physical GPUs 4 and 5:

```text
output/diagnostics/r136_official_m2f_remaining75_eval/gpu_mask_probe_nvidia_smi.txt
```

Remaining75 built-in eval:

| split | images | annotations | bbox AP | bbox AP50 | bbox AP75 | segm AP | segm AP50 | segm AP75 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| remaining75 | 75 | 4364 | 26.9038 | 60.0196 | 20.3872 | 24.0292 | 53.4401 | 18.1336 |


## Fixed1024 official eval plus source-RLE replay

A follow-up eval-only pass reran the R136 final checkpoint with fixed test input size and then replayed the saved official COCO JSON against the original source RLE annotation files.

Fixed eval overrides:

```text
CUDA_VISIBLE_DEVICES=6,7
--eval-only
INPUT.MIN_SIZE_TEST 1024
INPUT.MAX_SIZE_TEST 1024
MODEL.WEIGHTS output/baseline/r136_official_m2f_r134iter4500_target150_continue_200_1000_g67/model_final.pth
```

Artifacts:

```text
output/diagnostics/r136_official_m2f_val28_fixed1024_eval/eval.log
output/diagnostics/r136_official_m2f_val28_fixed1024_eval/eval_command.sh
output/diagnostics/r136_official_m2f_val28_fixed1024_eval/inference/coco_instances_results.json
output/diagnostics/r136_official_m2f_val28_fixed1024_eval/source_rle_replay_metrics.cocoeval.json
output/diagnostics/r136_official_m2f_remaining75_fixed1024_eval/eval.log
output/diagnostics/r136_official_m2f_remaining75_fixed1024_eval/eval_command.sh
output/diagnostics/r136_official_m2f_remaining75_fixed1024_eval/inference/coco_instances_results.json
output/diagnostics/r136_official_m2f_remaining75_fixed1024_eval/source_rle_replay_metrics.cocoeval.json
```

GPU and no-training evidence is in each `env_snapshot_pre.txt` / `env_snapshot_post.txt`. The logs show `resume=False, eval_only=True`, `MIN_SIZE_TEST: 1024`, `MAX_SIZE_TEST: 1024`, and `Start inference`; grep found no `Starting training` line. Physical GPUs 4 and 5 kept their pre-existing jobs, while the launcher used `CUDA_VISIBLE_DEVICES=6,7`.

Built-in fixed1024 official eval:

| split | images | bbox AP | bbox AP50 | bbox AP75 | segm AP | segm AP50 | segm AP75 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| val28 | 28 | 24.1935 | 57.0211 | 15.7493 | 22.6573 | 50.8427 | 16.8662 |
| remaining75 | 75 | 30.7500 | 65.8459 | 25.0719 | 29.1094 | 60.1029 | 24.8333 |

Source-RLE replay of the saved COCO JSON, with empty masks dropped and bbox recomputed from RLE masks in memory:

| split | input preds | dropped empty | kept preds | maxDets | bbox AP | bbox AP50 | bbox AP75 | segm AP | segm AP50 | segm AP75 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| val28 | 2800 | 723 | 2077 | 100 | 24.1935 | 57.0211 | 15.7493 | 22.6913 | 52.1415 | 15.5754 |
| val28 | 2800 | 723 | 2077 | 200 | 24.1935 | 57.0211 | 15.7493 | 22.6913 | 52.1415 | 15.5754 |
| remaining75 | 7500 | 2102 | 5398 | 100 | 30.7500 | 65.8459 | 25.0719 | 29.0370 | 60.3990 | 24.6425 |
| remaining75 | 7500 | 2102 | 5398 | 200 | 30.7500 | 65.8459 | 25.0719 | 29.0370 | 60.3990 | 24.6425 |

Caveat: this closes the fixed-test-size and source-RLE replay mismatch, but it is still a 100-query / 100-detection official Mask2Former export. Replay `maxDets=200` is identical to `maxDets=100` because the saved prediction JSON has at most 100 predictions per image. It is therefore fairer than the old 800-short-edge row, but it is not a true topk200 candidate-pool eval.

## Current RGB baseline gap versus R114

R114 MagFormer final external 1024 backmap topk200 rows remain stronger than the R136 fixed1024 official RGB rows. The R136 source-RLE replay closes the old fixed-size and annotation-format mismatch, but it is still capped at 100 predictions per image.

| split | model/protocol | bbox AP | segm AP | segm AP gap vs R114 |
| --- | --- | ---: | ---: | ---: |
| val28 | R136 fixed1024 built-in maxDets100 | 24.1935 | 22.6573 | -9.5258 |
| val28 | R136 fixed1024 source-RLE replay maxDets100/200 | 24.1935 | 22.6913 | -9.4918 |
| val28 | R114 MagFormer 1024 backmap topk200 | 35.6494 | 32.1831 | reference |
| remaining75 | R136 fixed1024 built-in maxDets100 | 30.7500 | 29.1094 | -10.1079 |
| remaining75 | R136 fixed1024 source-RLE replay maxDets100/200 | 30.7500 | 29.0370 | -10.1803 |
| remaining75 | R114 MagFormer 1024 backmap topk200 | 43.0296 | 39.2173 | reference |

Decision: the fixed1024 R136 remaining75 result is better than its val28 score but still far below R114. These metrics do not argue for launching an immediate R136 continuation to 2000.
