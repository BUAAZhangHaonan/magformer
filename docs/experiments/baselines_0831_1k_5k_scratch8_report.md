# 0831_1K / 5K (Scratch-8) Baseline Suite Report

This report corresponds to the **scratch-only** 8-model suite:

- Experiment ID: `0831_1k_5k_scratch8`
- Dataset: `magformer_datasets/0831_1K` (train 886 / val 110)
- Budget alignment:
  - Detectron2 / MagFormer / MGM / MSMFormer / UOAIS: `MAX_ITER=5000`, `IMS_PER_BATCH=8`
  - UCN / YOLOv8-seg: `epochs=45`, `batch=8`
- Initialization: **from scratch** (no ImageNet / no task-level warm-start)
- Metric: **COCOeval** `segm/bbox` AP (`AP/AP50/AP75/APs/APm/APl`)

## How To Run

Dry-run (print commands only):

```bash
cd magformer
bash scripts/experiments/run_0831_1k_5k_scratch8_all.sh --dry-run
```

Run (serial on `GPU0`):

```bash
cd magformer
bash scripts/experiments/run_0831_1k_5k_scratch8_all.sh --run
```

Summarize:

```bash
cd magformer
python scripts/experiments/summarize_0831_1k_5k_scratch8.py \
  --output-root output/experiments/0831_1k_5k_scratch8 \
  --write
```

## Outputs (Canonical Paths)

All outputs are written under:

- `magformer/output/experiments/0831_1k_5k_scratch8/`

Per-model directories:

- `magformer_scratch/`
- `mgm_mask2former_scratch/`
- `msmformer_scratch/`
- `uoais_scratch/`
- `ucn_scratch/`
- `official_mask2former_scratch/`
- `maskrcnn_scratch/`
- `yolov8_seg_scratch/`

Suite summary JSON:

- `summary_0831_1k_5k_scratch8.json`

## Results (Fill After Runs)

### Segm AP (COCOeval)

| Model | segm/AP | AP50 | AP75 | APs | APm | APl |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| magformer_scratch | TBD | TBD | TBD | TBD | TBD | TBD |
| mgm_mask2former_scratch | TBD | TBD | TBD | TBD | TBD | TBD |
| msmformer_scratch | TBD | TBD | TBD | TBD | TBD | TBD |
| uoais_scratch | TBD | TBD | TBD | TBD | TBD | TBD |
| ucn_scratch | TBD | TBD | TBD | TBD | TBD | TBD |
| official_mask2former_scratch | TBD | TBD | TBD | TBD | TBD | TBD |
| maskrcnn_scratch | TBD | TBD | TBD | TBD | TBD | TBD |
| yolov8_seg_scratch | TBD | TBD | TBD | - | - | - |

### BBox AP (COCOeval)

| Model | bbox/AP | AP50 | AP75 | APs | APm | APl |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| magformer_scratch | TBD | TBD | TBD | TBD | TBD | TBD |
| mgm_mask2former_scratch | TBD | TBD | TBD | TBD | TBD | TBD |
| msmformer_scratch | TBD | TBD | TBD | TBD | TBD | TBD |
| uoais_scratch | TBD | TBD | TBD | TBD | TBD | TBD |
| ucn_scratch | TBD | TBD | TBD | TBD | TBD | TBD |
| official_mask2former_scratch | TBD | TBD | TBD | TBD | TBD | TBD |
| maskrcnn_scratch | TBD | TBD | TBD | TBD | TBD | TBD |
| yolov8_seg_scratch | TBD | TBD | - | - | - | - |

### Params + Wall Time

| Model | params_trainable (M) | wall_time (sec) |
| --- | ---: | ---: |
| magformer_scratch | TBD | TBD |
| mgm_mask2former_scratch | TBD | TBD |
| msmformer_scratch | TBD | TBD |
| uoais_scratch | TBD | TBD |
| ucn_scratch | TBD | TBD |
| official_mask2former_scratch | TBD | TBD |
| maskrcnn_scratch | TBD | TBD |
| yolov8_seg_scratch | TBD | TBD |

## Notes / Known Caveats

- UOAIS is designed for amodal/occlusion supervision. In this suite, it is adapted to standard COCO **modal** masks and
  trains only the `A` branch (`MODEL.PREDICTION_ORDER=["A"]`) to avoid division-by-zero issues when no occlusion labels exist.
- RGB-only baselines ignore depth by design (Mask R-CNN / official Mask2Former / YOLOv8-seg).

