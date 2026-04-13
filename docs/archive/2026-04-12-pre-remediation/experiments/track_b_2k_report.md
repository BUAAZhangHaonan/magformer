# Track B 2k High-AP Reproduction Report

Date: February 18, 2026  
Scope: M9 (`exp(track-b): reproduce high-ap 2k recipe with unified eval`)

## 1) Experiment Setup

- Dataset: `magformer_datasets/0909_512_0.12K`
- Hardware: single GPU (`GPU0`)
- MagFormer config: `configs/magformer_track_b_2k.yaml`
- Mask2Former config: `baselines/MGM_Mask2Former/configs/mgm_swin_convnext_tiny.yaml`
- Iterations: `2,000`
- Track-B recipe highlights:
  - warm-start enabled
  - `BASE_LR=1e-4`
  - `WARMUP_ITERS=100`
  - `PRIOR.COMPUTE_ON=full`

## 2) Run Commands

- Unified run script:
  - `bash scripts/experiments/run_track_b_2k.sh --run`
- MagFormer offline export/eval (for visualization parity):
  - `conda run --no-capture-output -n magformer python tools/evaluate.py --config-file configs/magformer_track_b_2k.yaml --dataset-root magformer_datasets/0909_512_0.12K --weights output/experiments/track_b_2k/magformer/model_best.pth --output output/experiments/track_b_2k/magformer/eval_best --batch-size 1 --num-workers 2`
- Triptych visualization:
  - `conda run --no-capture-output -n magformer python scripts/compare_predictions.py --dataset-root magformer_datasets/0909_512_0.12K --ann-file annotations/instances_val.json --split val --magformer-results output/experiments/track_b_2k/magformer/eval_best/coco_instances_results.json --mask2former-results output/experiments/track_b_2k/mask2former/inference/coco_instances_results.json --output-dir output/experiments/track_b_2k/visualizations --num-images 12 --score-threshold 0.5`

## 3) Unified Metrics (COCO segm, percentage scale)

Summary file: `output/experiments/track_b_2k/track_b_2k_summary.json`

| Metric | MagFormer | Mask2Former | Gap (Mag - M2F) |
|---|---:|---:|---:|
| AP | 75.05 | 86.23 | -11.18 |
| AP50 | 91.70 | 95.99 | -4.29 |
| AP75 | 82.37 | 91.95 | -9.59 |
| APs | 44.23 | 52.74 | -8.51 |
| APm | 83.36 | 94.84 | -11.48 |
| APl | N/A | N/A | N/A |

## 4) Parity and Consistency Checks

- Parameter parity (grouped):
  - Saved: `output/experiments/track_b_2k/parameter_comparison.txt`
  - Total params:
    - MagFormer: `79,715,549`
    - Mask2Former: `79,718,429`
    - Difference: `-2,880` (`~0.00%`)
- Online/offline evaluation consistency (MagFormer, same checkpoint):
  - Online val at iter `1999` (`metrics_log.jsonl`): `segm_AP=0.7505172` (75.05%)
  - Offline eval from `model_best.pth` (`tools/evaluate.py`): `segm_AP=0.7484340` (74.84%)
  - Absolute difference: `0.208 pp`

## 5) Visualization Deliverables

- Triptych (GT / MagFormer / Mask2Former):
  - Directory: `output/experiments/track_b_2k/visualizations/triptych`
  - Example: `output/experiments/track_b_2k/visualizations/triptych/triptych_0000_id1.png`
- Single-model overlays retained:
  - MagFormer: `output/experiments/track_b_2k/visualizations/magformer_single`
  - Mask2Former: `output/experiments/track_b_2k/visualizations/mask2former_single`

## 6) Track B @2k Conclusion

- Under high-AP warm-start recipe at `2k`, Mask2Former leads by `11.18 AP`.
- Compared to Track A (`+22.74 AP` for MagFormer), Track B clearly shifts advantage to the reference-style warm-start path.
- The next step is full `20k` dual-track validation to determine whether MagFormer can close this gap under longer training.
