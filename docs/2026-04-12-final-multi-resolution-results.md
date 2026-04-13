# MAGFormer Final Multi-Resolution Results

## Report Metadata
- **Data sources:**
  - `output/analysis/2026-04-10-all-models-metrics-1024-512-sorted-by-segm-ap.csv`
  - `output/analysis/2026-04-10-live-metrics-manifest-fresh.json`
- **Repository commit:** `a9592766da4495253329c99d45a56792511d5b4b`
- **Repository tag:** `none`
- **Generation date:** `2026-04-12`
- **Evaluation protocol:** The live CSV and live manifest were used as the only source of truth. FPS values come from existing `inference_speed.json` / `inference_speed_clean.json` artifacts produced by the benchmark scripts, and the report excludes 256-resolution rows.

## Lineage
- **Dataset lineage:** `20260318_1K_1566`
- **Live experiment family:** 20-epoch 1024 and 512 runs under the `20260318_1k_1566_20ep_*` and `20260406_1k_1566_20ep_*` artifact trees
- **Scope:** 1024 and 512 only; 256 is not part of the supported publication set

## Training Recipe
- The published live artifacts come from the `20260318_1K_1566` lineage and the paired 1024 / 512 experiment trees.
- The artifact paths encode the 20-epoch recipe used for the live rows.
- This report does not add any new metrics beyond the two source files above.

## Evaluation Protocol
- Use the live CSV row values for `segm_AP`, `bbox_AP`, and `fps`.
- Use the refreshed manifest to keep the live artifact view consistent with the CSV.
- Keep rows that already carry reliability notes, and flag weak or degenerate results directly in the table notes.

### 1024 results

| model_id | segm_ap | bbox_ap | fps | note |
|---|---:|---:|---:|---|
| mgm_mask2former_depthnorm_on | 72.798 | 61.529 | 6.557 | — |
| magformer_depthnorm_on | 68.418 | 62.281 | 2.138 | — |
| magformer_lightdepth_convnextlite_spatialgate_edge_validhole | 65.453 | 57.649 | 2.433 | — |
| magformer_lightdepth_mobilenetv3_spatialgate_edge_validhole | 64.740 | 57.626 | 2.455 | — |
| magformer_lightdepth_mobilenetv3_sagate_edge_validhole | 64.503 | 57.654 | 2.500 | — |
| magformer_nodpth_ref | 59.523 | 52.580 | 2.556 | multiple live artifact candidates found; using experiments/20260406_1k_1566_20ep_1024_full19/magformer_nodpth_ref_fair; no inference-profile artifact saved |
| mask2former | 58.755 | 50.600 | 15.037 | — |
| maskrcnn | 54.101 | 59.488 | 32.894 | — |
| yolov8_seg_l | 40.563 | 61.733 | 40.819 | multiple live artifact candidates found; using experiments/20260318_1k_1566_20ep_1024_full19/yolov8_seg_l_pretrained |
| yolov8_seg_x | 40.324 | 61.989 | 42.283 | multiple live artifact candidates found; using experiments/20260318_1k_1566_20ep_1024_full19/yolov8_seg_x_pretrained |
| yolov8_seg_m | 40.079 | 60.890 | 55.455 | multiple live artifact candidates found; using experiments/20260318_1k_1566_20ep_1024_full19/yolov8_seg_m_pretrained |
| mgm_mask2former_nodpth_ref | 39.609 | 41.079 | 12.316 | — |
| yolov8_seg_s | 36.450 | 56.850 | 62.289 | multiple live artifact candidates found; using experiments/20260318_1k_1566_20ep_1024_full19/yolov8_seg_s_pretrained |
| yolov8_seg_n | 32.237 | 51.322 | 52.102 | multiple live artifact candidates found; using experiments/20260318_1k_1566_20ep_1024_full19/yolov8_seg_n_pretrained |
| uoais | 17.674 | 31.497 | 21.342 | — |
| unet_boundary_inst | 13.599 | 12.628 | 73.631 | multiple live artifact candidates found; using experiments/20260318_1k_1566_20ep_1024_full19/unet_boundary_inst; no inference-profile artifact saved; no training-memory artifact saved |
| unetpp_boundary_inst | 10.883 | 10.046 | 52.999 | multiple live artifact candidates found; using experiments/20260318_1k_1566_20ep_1024_full19/unetpp_boundary_inst; no inference-profile artifact saved; no training-memory artifact saved |
| unet_semantic_inst | 3.567 | 3.338 | 73.870 | multiple live artifact candidates found; using experiments/20260318_1k_1566_20ep_1024_full19/unet_semantic_inst; no inference-profile artifact saved; no training-memory artifact saved |
| ucn | 1.033 | 2.080 | 2.740 | weak results; artifact coverage incomplete; no inference-profile artifact saved; no training-memory artifact saved |
| msmformer | 0.000 | 0.000 | 2.925 | 0 AP; treat as unreliable; multiple live artifact candidates found; using 20260318_1k_1566_20ep_1024_full19/_backup_fix_20260324_normrepair/msmformer |

### 512 results

| model_id | segm_ap | bbox_ap | fps | note |
|---|---:|---:|---:|---|
| mgm_mask2former_depthnorm_on | 69.904 | 53.168 | 6.684 | no inference-profile artifact saved |
| magformer_lightdepth_mobilenetv3_spatialgate_edge_validhole | 60.364 | 58.427 | 2.487 | no inference-profile artifact saved |
| magformer_depthnorm_on | 59.910 | 49.447 | 2.194 | no inference-profile artifact saved |
| magformer_lightdepth_mobilenetv3_sagate_edge_validhole | 59.753 | 55.009 | 2.517 | no inference-profile artifact saved |
| magformer_lightdepth_convnextlite_spatialgate_edge_validhole | 59.736 | 54.909 | 2.498 | no inference-profile artifact saved |
| magformer_nodpth_ref | 53.875 | 49.265 | 2.584 | no inference-profile artifact saved |
| mask2former | 43.085 | 40.982 | 20.533 | no inference-profile artifact saved |
| yolov8_seg_x | 42.928 | 63.858 | 29.787 | no inference-profile artifact saved |
| yolov8_seg_l | 42.682 | 63.671 | 32.582 | no inference-profile artifact saved |
| yolov8_seg_m | 42.105 | 62.334 | 34.212 | no inference-profile artifact saved |
| yolov8_seg_s | 40.213 | 59.981 | 56.985 | no inference-profile artifact saved |
| maskrcnn | 38.780 | 45.671 | 35.433 | no inference-profile artifact saved |
| yolov8_seg_n | 35.884 | 54.244 | 59.485 | no inference-profile artifact saved |
| mgm_mask2former_nodpth_ref | 32.219 | 34.876 | 12.153 | no inference-profile artifact saved |
| unetpp_boundary_inst | 10.179 | 9.492 | 53.215 | no inference-profile artifact saved; no training-memory artifact saved |
| msmformer | 8.570 | 8.345 | 2.216 | no inference-profile artifact saved |
| uoais | 6.164 | 15.431 | 27.860 | no inference-profile artifact saved |
| unet_boundary_inst | 5.787 | 4.588 | 73.812 | no inference-profile artifact saved; no training-memory artifact saved |
| unet_semantic_inst | 3.254 | 3.148 | 73.885 | no inference-profile artifact saved; no training-memory artifact saved |
| ucn | 0.049 | 0.372 | 2.811 | weak results; artifact coverage incomplete; no inference-profile artifact saved; no training-memory artifact saved |
