# R97 Official Mask2Former 32K Cache Short Training

Date: 2026-05-18
Host: `4029` (`/home/hdd3/zhanghaonan/magformer`)
Branch: `feature/vc-suda-sim2real`
Conda env: `mask2former`
TMUX session: `r97_official_m2f_32k_cache_short`
Output: `output/baseline/r97_official_m2f_32k_cache_short`
Dataset root: `magformer_datasets/20260318_1K_32254`
Train cache: `magformer_datasets/20260318_1K_32254/cache/coco_loader/instances_train.sqlite`
Register id: `20260318_1K_32254`

## Goal

Run the official RGB-only Mask2Former baseline on the 32K source train split for a short 1500-iteration gate after R96 validated SQLite train-cache registration.

The gate for R98 is segmentation AP. The expected pass line was roughly 8-10 segm AP; bbox AP is recorded, but it is not the main target for this baseline gate.

## Command and environment

The run used the same official wrapper path as R96, with GPU 4-7 and the SQLite train-cache registration path. The observed wrapper-level command shape was:

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

The tmux log confirms train and val dataset registration:

```text
[official-mask2former] registered train dataset: ecc20260318_1k_32254_train
[official-mask2former] registered val/test dataset: ecc20260318_1k_32254_val
```

Detectron2 reported this runtime environment:

- Python: `3.11.0`
- PyTorch: `2.5.1+cu121`
- Detectron2: `0.6`
- CUDA compiler: `12.1`
- GPU world size: `4`
- Visible GPUs in the process: `NVIDIA GeForce RTX 3090`

The trainer args recorded in `log.txt` / tmux log were:

```text
--num-gpus 4
--config-file configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml
OUTPUT_DIR /home/hdd3/zhanghaonan/magformer/output/baseline/r97_official_m2f_32k_cache_short
DATASETS.TRAIN ("ecc20260318_1k_32254_train",)
DATASETS.TEST ("ecc20260318_1k_32254_val",)
MODEL.WEIGHTS detectron2://ImageNetPretrained/torchvision/R-50.pkl
MODEL.SEM_SEG_HEAD.NUM_CLASSES 1
MODEL.MASK_FORMER.NUM_OBJECT_QUERIES 100
TEST.DETECTIONS_PER_IMAGE 100
INPUT.IMAGE_SIZE 1024
SOLVER.IMS_PER_BATCH 4
SOLVER.MAX_ITER 1500
SOLVER.CHECKPOINT_PERIOD 500
TEST.EVAL_PERIOD 500
DATALOADER.NUM_WORKERS 2
```

## Metrics

Completed evals from `metrics.json` / `log.txt`:

| Iteration | Checkpoint | bbox AP | bbox AP50 | bbox AP75 | segm AP | segm AP50 | segm AP75 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 499 | `model_0000499.pth` | 0.0000 | 0.0000 | 0.0000 | 12.8163 | 29.0234 | 10.4390 |
| 999 | `model_0000999.pth` | 0.0000 | 0.0000 | 0.0000 | 29.8435 | 50.4509 | 31.1410 |
| 1500 | `model_final.pth` | 0.0000 | 0.0000 | 0.0000 | 36.2643 | 55.3776 | 39.9021 |

Training loss at the last training row, iter `1499`:

| total_loss | loss_ce | loss_mask | loss_dice | lr |
|---:|---:|---:|---:|---:|
| 20.2654 | 0.4112 | 0.1757 | 1.3604 | 0.0001 |

Checkpoint notes:

- Saved checkpoints: `model_0000499.pth`, `model_0000999.pth`, `model_0001499.pth`, `model_final.pth`.
- `last_checkpoint` points to `model_final.pth`.
- No `best` checkpoint marker appears in `log.txt`.
- Best segm AP by recorded eval is `36.2643` at iter `1500`, represented by `model_final.pth` for this run.
- Best bbox AP is tied at `0.0000`; it is not useful for selecting a checkpoint here.

## Error scan

Scanned `output/baseline/r97_official_m2f_32k_cache_short/log.txt` for:

- `Traceback`: 0
- `OOM`: 0
- `RuntimeError`: 0
- `Killed`: 0
- `NaN`: 0
- `non-finite`: 0

The run reached iter `1499`, saved `model_final.pth`, and completed the final val eval.

## Inference result sanity

Checked `output/baseline/r97_official_m2f_32k_cache_short/inference/coco_instances_results.json`:

- Exists: yes
- Size: `142397766` bytes
- Prediction count: `327600`
- JSON type: list
- Fields present in sampled predictions: `bbox`, `segmentation`, `score`, `category_id`, `image_id`
- Required fields present in the first 1000 predictions:
  - `bbox`: yes
  - `segmentation`: yes
  - `score`: yes
  - `category_id`: yes
- All `327600` prediction bbox entries are `[0.0, 0.0, 0.0, 0.0]`.
- All `327600` prediction bbox entries have non-positive width/height.
- Segmentation RLEs are present, and final segm AP is non-zero and high for this gate.

## bbox AP 0 judgment

The bbox AP 0 result is more likely a prediction bbox export / bbox-field convention problem than ordinary poor box quality.

Reason: the inference JSON contains valid-looking segmentation predictions with scores and category ids, and the segm AP reaches `36.2643`. At the same time, every prediction bbox is exactly `[0.0, 0.0, 0.0, 0.0]`. If boxes were merely low quality, the bbox values would normally vary and at least some boxes would have positive width and height. This points to the bbox field being unset or serialized with a wrong path for this official Mask2Former instance output.

No code was changed in this audit.

## Conclusion

Pass for R98 gate.

R97 reached final segm AP `36.2643`, clearly above the 8-10 AP gate. The official RGB-only 32K source-supervised Mask2Former short run is learning useful masks on the source-val protocol.

bbox AP remains `0.0000` and needs a separate diagnosis before using bbox AP as a quality target. It does not block continuing with segm AP as the main objective.
