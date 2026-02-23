# 0831_1K / 20 Epoch (Scratch-8) Baseline Suite Report

- Experiment ID: `0831_1k_20ep_scratch8`
- Dataset: `magformer_datasets/0831_1K` (train 886 / val 110)
- Training budget:
  - Iter-based models: `MAX_ITER=2220` (20 epoch equivalent @ batch=8)
  - Epoch-based models: `epochs=20` (YOLOv8-seg / UCN)
- Initialization: scratch only
- Metrics: unified `pycocotools COCOeval` (`segm` + `bbox`)

## Run Commands

```bash
cd magformer
bash scripts/experiments/run_0831_1k_20ep_scratch8_all.sh --dry-run
bash scripts/experiments/run_0831_1k_20ep_scratch8_all.sh --run
python scripts/experiments/summarize_0831_1k_20ep_scratch8.py --output-root output/experiments/0831_1k_20ep_scratch8 --write
python scripts/experiments/visualize_0831_1k_20ep_scratch8.py --dataset-root /home/k100/zhn/electronic-components-grasp-and-segment/magformer_datasets/0831_1K --output-root output/experiments/0831_1k_20ep_scratch8
```

## Outputs

- Root: `output/experiments/0831_1k_20ep_scratch8/`
- Summary: `output/experiments/0831_1k_20ep_scratch8/summary_0831_1k_20ep_scratch8.json`
- Visualizations:
  - Overlay: `output/experiments/0831_1k_20ep_scratch8/visualizations/<model>/overlay/`
  - Triptych: `output/experiments/0831_1k_20ep_scratch8/visualizations/triptych_gt_magformer_mgm/`

## Segm AP (COCOeval)

| Model | AP | AP50 | AP75 | APs | APm | APl |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| magformer_scratch | TBD | TBD | TBD | TBD | TBD | TBD |
| mgm_mask2former_scratch | TBD | TBD | TBD | TBD | TBD | TBD |
| msmformer_scratch | TBD | TBD | TBD | TBD | TBD | TBD |
| uoais_scratch | TBD | TBD | TBD | TBD | TBD | TBD |
| ucn_scratch | TBD | TBD | TBD | TBD | TBD | TBD |
| official_mask2former_scratch | TBD | TBD | TBD | TBD | TBD | TBD |
| maskrcnn_scratch | TBD | TBD | TBD | TBD | TBD | TBD |
| yolov8_seg_scratch | TBD | TBD | TBD | TBD | TBD | TBD |

## BBox AP (COCOeval)

| Model | AP | AP50 | AP75 | APs | APm | APl |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| magformer_scratch | TBD | TBD | TBD | TBD | TBD | TBD |
| mgm_mask2former_scratch | TBD | TBD | TBD | TBD | TBD | TBD |
| msmformer_scratch | TBD | TBD | TBD | TBD | TBD | TBD |
| uoais_scratch | TBD | TBD | TBD | TBD | TBD | TBD |
| ucn_scratch | TBD | TBD | TBD | TBD | TBD | TBD |
| official_mask2former_scratch | TBD | TBD | TBD | TBD | TBD | TBD |
| maskrcnn_scratch | TBD | TBD | TBD | TBD | TBD | TBD |
| yolov8_seg_scratch | TBD | TBD | TBD | TBD | TBD | TBD |

## Params + Wall Time

| Model | params_trainable (M) | wall_time_sec |
| --- | ---: | ---: |
| magformer_scratch | TBD | TBD |
| mgm_mask2former_scratch | TBD | TBD |
| msmformer_scratch | TBD | TBD |
| uoais_scratch | TBD | TBD |
| ucn_scratch | TBD | TBD |
| official_mask2former_scratch | TBD | TBD |
| maskrcnn_scratch | TBD | TBD |
| yolov8_seg_scratch | TBD | TBD |
