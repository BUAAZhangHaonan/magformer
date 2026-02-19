# 0831_1K / 5K Iter Baseline Suite Report (Template)

Date: February 19, 2026

## Goal

Run a small-but-meaningful baseline suite on `magformer_datasets/0831_1K` (train=886, val=110) to compare:

- MagFormer (RGBD)
- MGM_Mask2Former (modified, Detectron2-based, RGBD)
- Official Mask2Former (facebookresearch/Mask2Former, RGB-only)
- Mask R-CNN (Detectron2, RGB-only)
- YOLOv8-seg (ultralytics, RGB-only)

Budget alignment:

- Detectron2/MagFormer: `MAX_ITER=5000`, `IMS_PER_BATCH=8`
- YOLOv8-seg: `epochs=45` (approx. matches 5000 iters @ batch=8)

Main metric target: COCO `segm/AP` (also record `AP50/AP75/APs/APm/APl` and bbox table).

## How To Run (One Command)

From `magformer/` repo root:

```bash
bash scripts/experiments/run_0831_1k_5k_all.sh --run
```

Dry-run:

```bash
bash scripts/experiments/run_0831_1k_5k_all.sh --dry-run
```

## Outputs

Default output root:

- `output/experiments/0831_1k_5k/`

Per-model subdirs:

- `output/experiments/0831_1k_5k/magformer/`
- `output/experiments/0831_1k_5k/mgm_mask2former/`
- `output/experiments/0831_1k_5k/official_mask2former/`
- `output/experiments/0831_1k_5k/maskrcnn/`
- `output/experiments/0831_1k_5k/yolov8_seg/`

Unified summary json:

- `output/experiments/0831_1k_5k/summary_0831_1k_5k.json`

Regenerate summary:

```bash
python scripts/experiments/summarize_0831_1k_5k.py \
  --output-root output/experiments/0831_1k_5k \
  --write
```

## Visualization (Triptych)

Prerequisites:

- MagFormer evaluation now dumps `coco_instances_results.json` under its output dir.
- Detectron2 evaluators typically dump `coco_instances_results.json` under their output dir (after the first eval).

Triptych (GT / MagFormer / MGM_Mask2Former):

```bash
python scripts/compare_predictions.py \
  --dataset-root ../magformer_datasets/0831_1K \
  --ann-file annotations/instances_val.json \
  --split val \
  --magformer-results output/experiments/0831_1k_5k/magformer/coco_instances_results.json \
  --mask2former-results output/experiments/0831_1k_5k/mgm_mask2former/coco_instances_results.json \
  --output-dir output/experiments/0831_1k_5k/visualizations/mag_vs_mgm \
  --num-images 50 \
  --score-threshold 0.5
```

## Results (To Fill After Runs)

### Metrics (Segmentation)

| Model | AP | AP50 | AP75 | APs | APm | APl |
|---|---:|---:|---:|---:|---:|---:|
| MagFormer | 33.51 | 59.49 | 32.82 | 6.01 | 40.22 | 19.03 |
| MGM_Mask2Former | 21.01 | 43.23 | 17.76 | 1.91 | 23.27 | 14.21 |
| Official Mask2Former | 50.12 | 72.92 | 54.90 | 1.83 | 50.39 | 81.56 |
| Mask R-CNN | 62.69 | 88.50 | 71.84 | 5.51 | 63.98 | 85.36 |
| YOLOv8-seg | 36.88 | 57.46 | - | - | - | - |

### Metrics (BBox)

| Model | AP | AP50 | AP75 | APs | APm | APl |
|---|---:|---:|---:|---:|---:|---:|
| MagFormer | 14.87 | 30.18 | 11.81 | 11.93 | 22.95 | 5.07 |
| MGM_Mask2Former | - | - | - | - | - | - |
| Official Mask2Former | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Mask R-CNN | 73.28 | 92.94 | 83.35 | 25.70 | 76.03 | 86.51 |
| YOLOv8-seg | 44.04 | 60.13 | - | - | - | - |

### Efficiency / Params (To Fill)

| Model | Params (M) | Train wall time (sec) | Notes |
|---|---:|---:|---|
| MagFormer | 79.716 | 9564 | RGBD |
| MGM_Mask2Former | 79.718 | 6101 | RGBD; bbox metrics not emitted in `metrics.json` |
| Official Mask2Former | 43.950 | 14333 | RGB-only; bbox metrics are all 0.0 (needs investigation) |
| Mask R-CNN | 43.696 | 2485 | RGB-only |
| YOLOv8-seg | 3.264 | 744 | RGB-only; Ultralytics metrics (not COCOeval) |

## Notes / Caveats

- RGB-only baselines ignore depth modality by design; results are not directly comparable to RGBD models in terms of input information content.
- YOLOv8 metrics in `results.csv` are Ultralytics-reported; a follow-up step will re-evaluate all models via a single COCOeval path for strict parity.
- YOLOv8-seg required `polars-runtime-compat==1.38.1` on this machine to avoid `illegal instruction`; runner forces `POLARS_FORCE_PKG=compat`.
