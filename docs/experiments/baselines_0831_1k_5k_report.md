# 0831_1K / 5K Iter Baseline Suite Report (Template)

Date: February 18, 2026

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
| MagFormer |  |  |  |  |  |  |
| MGM_Mask2Former |  |  |  |  |  |  |
| Official Mask2Former |  |  |  |  |  |  |
| Mask R-CNN |  |  |  |  |  |  |
| YOLOv8-seg |  |  |  |  |  |  |

### Metrics (BBox)

| Model | AP | AP50 | AP75 | APs | APm | APl |
|---|---:|---:|---:|---:|---:|---:|
| MagFormer |  |  |  |  |  |  |
| MGM_Mask2Former |  |  |  |  |  |  |
| Official Mask2Former |  |  |  |  |  |  |
| Mask R-CNN |  |  |  |  |  |  |
| YOLOv8-seg |  |  |  |  |  |  |

### Efficiency / Params (To Fill)

| Model | Params (M) | Train wall time (sec) | Notes |
|---|---:|---:|---|
| MagFormer |  |  |  |
| MGM_Mask2Former |  |  |  |
| Official Mask2Former |  |  |  |
| Mask R-CNN |  |  |  |
| YOLOv8-seg |  |  |  |

## Notes / Caveats

- RGB-only baselines ignore depth modality by design; results are not directly comparable to RGBD models in terms of input information content.
- YOLOv8 metrics in `results.csv` are Ultralytics-reported; a follow-up step will re-evaluate all models via a single COCOeval path for strict parity.

