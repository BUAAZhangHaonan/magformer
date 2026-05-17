# R97 Official Mask2Former 32K Cache Short Training

Date: 2026-05-18
Host: `4029` (`WS-4029GP-TRT`)
Repository: `/home/hdd3/zhanghaonan/magformer`
Branch: `feature/vc-suda-sim2real`
Conda env: `mask2former`
TMUX session: `r97_official_m2f_32k_short`
Output: `output/baseline/r97_official_m2f_32k_cache_short`
Dataset root: `magformer_datasets/20260318_1K_32254`
Train cache: `cache/coco_loader/instances_train.sqlite`
Register id: `20260318_1K_32254`

## Goal

Run the official Mask2Former RGB-only source-supervised short training on the
32K SQLite train cache for 1500 iterations, with evaluation at 500, 1000, and
1500 iterations.

## Status Audit

- Initial audit at 2026-05-18 00:08 CST found the tmux session
  `r97_official_m2f_32k_short` active.
- GPU 4, 5, 6, and 7 had R97 Mask2Former Python worker processes attached.
- `ps` showed the launcher `/tmp/r97_official_m2f_32k_short_launch.sh` and the
  official `baselines/run_official_mask2former_ecc.py` command.
- `git status --short --branch` showed branch
  `feature/vc-suda-sim2real...origin/feature/vc-suda-sim2real`, with untracked
  `output/baseline/` and `output/logs/` only.
- The run completed without interruption. After completion, the R97 tmux session
  was gone and no R97 Mask2Former training process remained.

## Command

The observed tmux process launched the official runner with GPU 4-7 and these
key options:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
python -u -c '... baselines/run_official_mask2former_ecc.py ...' \
  --register 20260318_1K_32254 \
  --dataset-root magformer_datasets/20260318_1K_32254 \
  --train-cache cache/coco_loader/instances_train.sqlite \
  --normalized-ann-dir output/diagnostics/r97_official_mask2former_coco \
  -- --num-gpus 4 \
  --config-file configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml \
  OUTPUT_DIR /home/hdd3/zhanghaonan/magformer/output/baseline/r97_official_m2f_32k_cache_short \
  DATASETS.TRAIN '("ecc20260318_1k_32254_train",)' \
  DATASETS.TEST '("ecc20260318_1k_32254_val",)' \
  MODEL.WEIGHTS detectron2://ImageNetPretrained/torchvision/R-50.pkl \
  MODEL.SEM_SEG_HEAD.NUM_CLASSES 1 \
  MODEL.MASK_FORMER.NUM_OBJECT_QUERIES 100 \
  TEST.DETECTIONS_PER_IMAGE 100 \
  INPUT.IMAGE_SIZE 1024 \
  SOLVER.IMS_PER_BATCH 4 \
  SOLVER.MAX_ITER 1500 \
  SOLVER.CHECKPOINT_PERIOD 500 \
  TEST.EVAL_PERIOD 500 \
  DATALOADER.NUM_WORKERS 2
```

## Metrics

Metrics were read from
`output/baseline/r97_official_m2f_32k_cache_short/metrics.json`.

| Eval | Metrics iteration | Checkpoint | bbox AP | bbox AP50 | bbox AP75 | segm AP | segm AP50 | segm AP75 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 500 | 499 | `model_0000499.pth` | 0.0000 | 0.0000 | 0.0000 | 12.8163 | 29.0234 | 10.4390 |
| 1000 | 999 | `model_0000999.pth` | 0.0000 | 0.0000 | 0.0000 | 29.8435 | 50.4509 | 31.1410 |
| 1500 | 1500 | `model_final.pth` | 0.0000 | 0.0000 | 0.0000 | 36.2643 | 55.3776 | 39.9021 |

Best checkpoint by segm AP: `model_final.pth` with segm AP `36.2643`.
`last_checkpoint` also points to `model_final.pth`.

## Artifacts

- Checkpoints were generated in the output directory and were not committed:
  `model_0000499.pth`, `model_0000999.pth`, `model_0001499.pth`, and
  `model_final.pth`.
- Predictions were generated under
  `output/baseline/r97_official_m2f_32k_cache_short/inference/`:
  `instances_predictions.pth` and `coco_instances_results.json`.
- Strict log scan found no `Traceback`, `Exception`, `ERROR`, `RuntimeError`,
  CUDA OOM, or NaN tokens in the R97 logs.

## R98 Gate

Pass. Segm AP improved monotonically from `12.8163` to `29.8435` to `36.2643`.
