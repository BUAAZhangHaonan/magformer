# 0831_1K / 1024 / 20 Epoch Final Report

- Dataset: `0831_1K`
- Resolution: `1024`
- Primary result root: `output/experiments/0831_1k_20ep_1024_depth_revisit`
- Summary source: `output/experiments/0831_1k_20ep_1024_depth_revisit/summary_0831_1k_20ep_1024_depth_revisit.json`

## Final Ranking

| Model | Best segm AP | Last segm AP | Best iter/epoch | Last iter/epoch |
|---|---:|---:|---:|---:|
| magformer_depthnorm_on | 77.500 | 77.339 | 4217 | -1 |
| mgm_mask2former_depthnorm_on | 77.393 | 77.346 | 3995 | -1 |
| official_mask2former_pretrained | 73.050 | 73.050 | 2220 | -1 |
| magformer_nodpth_ref | 70.489 | 70.332 | 4217 | -1 |
| maskrcnn_pretrained | 62.166 | 62.164 | 2108 | -1 |
| mgm_mask2former_nodpth_ref | 54.642 | 54.486 | 4217 | -1 |
| yolov8_seg_pretrained | 51.683 | 49.850 | 20 | -1 |
| maskrcnn_scratch | 48.483 | 48.441 | 2108 | -1 |
| yolov8_seg_scratch | 35.343 | 32.225 | 20 | -1 |
| official_mask2former_scratch | 28.719 | 28.425 | 2108 | -1 |
| uoais_scratch | 16.092 | 15.852 | 2108 | -1 |
| unet_semantic_inst | 11.019 | 9.053 | 13 | 20 |
| unet_boundary_inst | 7.699 | 2.949 | 17 | 20 |
| unetpp_boundary_inst | 6.671 | 5.929 | 19 | 20 |
| ucn_scratch | 0.003 | 0.003 | -1 | -1 |

## Key Findings

- `magformer_depthnorm_on` and `mgm_mask2former_depthnorm_on` are the strongest models and remain effectively tied at ~77.4–77.5 AP.
- `magformer_depthnorm_on` improves over `magformer_nodpth_ref` by roughly `+7.01 AP`.
- `mgm_mask2former_depthnorm_on` improves over `mgm_mask2former_nodpth_ref` by roughly `+22.75 AP`; this no-depth MGM reference is not directly comparable to the MagFormer no-depth reference because its initialization route is weaker.
- In the fairer pretrained subset, `official_mask2former_pretrained` reaches `73.050 AP`, `maskrcnn_pretrained` reaches `62.166 AP`, and `yolov8_seg_pretrained` reaches `51.683 AP`.
- `unet_semantic_inst` is the strongest U-Net-style baseline (`11.019 AP`) and visually produces cleaner boundaries than the instance-style U-Net variants, but instance separation remains the limiting factor.

## Fairness Notes

- `magformer_*` and `mgm_mask2former_*` use warm-start / pretrained initialization in this benchmark.
- To improve fairness, additional pretrained baselines were added: `maskrcnn_pretrained`, `official_mask2former_pretrained`, `yolov8_seg_pretrained`.
- `uoais_scratch`, `ucn_scratch`, and the U-Net family remain scratch baselines in this run.

## Preserved Artifacts

- Per-model directories retain final metrics, exported COCO predictions, selected checkpoints, run logs, metadata, and local `visualizations/overlay/` outputs.
- Suite-level outputs are preserved in `output/experiments/0831_1k_20ep_1024_depth_revisit/visualizations/` plus `revisit_comparison.json` and `revisit_comparison.csv`.
