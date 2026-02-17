# Track A 2k Strict-Parity Report

Date: February 18, 2026  
Scope: M8 (`exp(track-a): run 2k strict-parity benchmark and report`)

## 1) Experiment Setup

- Dataset: `/home/k100/zhn/electronic-components-grasp-and-segment/magformer_datasets/0909_512_0.12K`
- Hardware: single GPU (`GPU0`)
- MagFormer config: `configs/magformer_track_a_2k.yaml`
- Mask2Former config: `/home/k100/zhn/electronic-components-grasp-and-segment/mask2former/MGM_Mask2Former/configs/mgm_aligned_comparison.yaml`
- Iterations: `2,000`
- Warm-start: disabled on both sides (`MODEL.FINETUNE_WEIGHTS=''`, `model.finetune_weights=null`)

## 2) Run Commands

- MagFormer:
  - `conda run -n magformer python tools/train.py --config configs/magformer_track_a_2k.yaml --dataset-root /home/k100/zhn/electronic-components-grasp-and-segment/magformer_datasets/0909_512_0.12K --output-dir output/experiments/track_a_2k/magformer --num-workers 4`
- Mask2Former:
  - `conda run -n magformer python train_net_mgm_0831.py --num-gpus 1 --config-file configs/mgm_aligned_comparison.yaml INPUT.DATASET_ROOT /home/k100/zhn/electronic-components-grasp-and-segment/magformer_datasets/0909_512_0.12K OUTPUT_DIR /home/k100/zhn/electronic-components-grasp-and-segment/magformer/output/experiments/track_a_2k/mask2former MODEL.FINETUNE_WEIGHTS '' MODEL.WEIGHTS ''`

## 3) Unified Metrics (COCO segm, percentage scale)

Summary file: `output/experiments/track_a_2k/track_a_2k_summary.json`

| Metric | MagFormer | Mask2Former | Gap (Mag - M2F) |
|---|---:|---:|---:|
| AP | 52.42 | 29.68 | +22.74 |
| AP50 | 74.65 | 50.74 | +23.91 |
| AP75 | 59.90 | 33.00 | +26.90 |
| APs | 20.72 | 2.29 | +18.43 |
| APm | 61.31 | 37.33 | +23.98 |
| APl | N/A | N/A | N/A |

## 4) Parity and Consistency Checks

- Parameter parity (grouped):
  - Saved: `output/experiments/track_a_2k/parameter_comparison.txt`
  - Total params:
    - MagFormer: `79,715,549`
    - Mask2Former: `79,718,429`
    - Difference: `-2,880` (`~0.00%`)
- Online/offline evaluation consistency (MagFormer, same checkpoint):
  - Online val at iter `1999` (`metrics_log.jsonl`): `segm_AP=0.5242479` (52.42%)
  - Offline eval from `checkpoint_iter_0001999.pth`: `segm_AP=0.5218342` (52.18%)
  - Absolute difference: `0.241 pp`

## 5) Visualization Deliverables

- Triptych (GT / MagFormer / Mask2Former):
  - Directory: `output/experiments/track_a_2k/visualizations/triptych`
  - Example: `output/experiments/track_a_2k/visualizations/triptych/triptych_0000_id1.png`
- Single-model overlays retained:
  - MagFormer: `output/experiments/track_a_2k/visualizations/magformer_single`
  - Mask2Former: `output/experiments/track_a_2k/visualizations/mask2former_single`

## 6) Current Conclusion for Track A @2k

- Under strict no-warm-start parity at `2k`, MagFormer currently leads Mask2Former by `+22.74 AP` on this dataset split.
- This result is explicitly the Track A setting (strict fairness), not the historical high-AP warm-start recipe.
