# Pretrained Subset Report

- Source summary: `output/experiments/0831_1k_20ep_1024_depth_revisit/summary_0831_1k_20ep_1024_depth_revisit.json`

## Ranking

| Model | Best segm AP |
|---|---:|
| magformer_depthnorm_on | 77.500 |
| mgm_mask2former_depthnorm_on | 77.393 |
| official_mask2former_pretrained | 73.050 |
| maskrcnn_pretrained | 62.166 |
| yolov8_seg_pretrained | 51.683 |

## Interpretation Notes

- `mgm_mask2former_nodpth_ref` is included as a diagnostic reference only.
- It is **not directly comparable** to `mgm_mask2former_depthnorm_on` as a symmetric no-depth ablation, because it is initialized from a generic public COCO checkpoint with depth/MGM/DPE disabled rather than from a dedicated 0831 no-depth reference model.
- Current `mgm_mask2former_nodpth_ref` best AP: `54.642`.
