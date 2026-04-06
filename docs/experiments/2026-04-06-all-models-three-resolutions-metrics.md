| Resolution | Model | Training mode | Status | segm AP | AP50 | AP75 | bbox AP | bbox AP50 | bbox AP75 | P@50 | R@50 | F1@50 | Params | Train sec | Train mem MB | Infer ms | Infer mem MB | FPS | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1024 | magformer_depthnorm_on | fine-tuned | ok | 68.4177 | 87.9741 | 77.7181 | 62.2810 | 82.9465 | 69.2393 | 96.6455 | 88.0000 | 92.1203 | 79696062 | 9453.0000 | 29975.9300 | 467.7866 | 3399.0015 | 2.1377 |  |
| 1024 | magformer_lightdepth_convnextlite_spatialgate_edge_validhole | fine-tuned | ok | 65.4526 | 86.8597 | 74.6540 | 57.6493 | 81.9553 | 66.3226 | 95.9131 | 86.0000 | 90.6864 | 50237888 | 8602.0000 | 15569.7500 | 410.9326 | 2455.9868 | 2.4335 |  |
| 1024 | magformer_lightdepth_mobilenetv3_sagate_edge_validhole | fine-tuned | ok | 64.5032 | 85.8691 | 73.6259 | 57.6543 | 82.0998 | 66.2676 | 95.9349 | 85.0000 | 90.1370 | 49958798 | 8989.0000 | 14138.1800 | 399.9963 | 2453.9731 | 2.5000 |  |
| 1024 | magformer_lightdepth_mobilenetv3_spatialgate_edge_validhole | fine-tuned | ok | 64.7399 | 85.9040 | 73.6656 | 57.6262 | 82.0185 | 65.5521 | 96.7993 | 85.0000 | 90.5168 | 49912530 | 8932.0000 | 14136.6400 | 407.3516 | 2453.7969 | 2.4549 |  |
| 1024 | magformer_nodpth_ref | fine-tuned | ok | 48.7786 | 74.9858 | 54.3229 | 44.6381 | 74.8140 | 47.7525 | 91.2694 | 73.0000 | 81.1188 | 79628990 | 8854.0000 | 14059.8000 | 396.4242 | 2568.5430 | 2.5226 |  |
| 1024 | mask2former | fine-tuned | ok | 58.7554 | 80.5301 | 66.0488 | 50.5996 | 77.8286 | 58.2976 | 93.4138 | 79.0000 | 85.6044 | 44056196 | 6944.0000 | 35761.0000 | 66.5025 | 2711.4526 | 15.0370 |  |
| 1024 | maskrcnn | fine-tuned | ok | 54.1012 | 77.6090 | 61.4442 | 59.4877 | 83.6752 | 67.7998 | 94.8226 | 76.0000 | 84.3743 | 44024278 | 1850.0000 | 9476.0000 | 30.4008 | 1579.5366 | 32.8938 |  |
| 1024 | mgm_mask2former_depthnorm_on | fine-tuned | ok | 72.7981 | 87.9188 | 78.6997 | 61.5294 | 83.5279 | 69.8334 | 96.8018 | 87.0000 | 91.6395 | 79696062 | 11053.0000 | 61279.0000 | 152.5003 | 2777.1172 | 6.5574 |  |
| 1024 | mgm_mask2former_nodpth_ref | fine-tuned | ok | 39.6095 | 68.6068 | 40.9181 | 41.0790 | 74.4927 | 41.2291 | 87.0296 | 66.0000 | 75.0698 | 79628990 | 8176.0000 | 29857.0000 | 81.1941 | 2778.3604 | 12.3162 |  |
| 1024 | msmformer | from-scratch | ok | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 50745922 | 12356.0000 | 77197.0000 | 341.9223 | 3891.8457 | 2.9246 | multiple live artifact candidates found; using 20260318_1k_1566_20ep_1024_full19/_backup_fix_20260324_normrepair/msmformer |
| 1024 | ucn | from-scratch | missing |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | live artifact missing in current tree; rerun required |
| 1024 | unet_boundary_inst | from-scratch | ok | 13.5987 | 35.0566 | 8.7096 | 12.6285 | 36.6316 | 6.2889 | 62.9120 | 40.0000 | 48.9055 | 1942594 | 6626.0000 |  |  |  |  | multiple live artifact candidates found; using experiments/20260318_1k_1566_20ep_1024_full19/unet_boundary_inst; no inference-profile artifact saved; no training-memory artifact saved |
| 1024 | unet_semantic_inst | from-scratch | ok | 3.5670 | 4.2916 | 3.8042 | 3.3375 | 4.0071 | 3.2454 | 65.3963 | 4.0000 | 7.5389 | 1942594 | 2693.0000 |  |  |  |  | multiple live artifact candidates found; using experiments/20260318_1k_1566_20ep_1024_full19/unet_semantic_inst; no inference-profile artifact saved; no training-memory artifact saved |
| 1024 | unetpp_boundary_inst | from-scratch | ok | 10.8825 | 27.8168 | 6.8792 | 10.0462 | 30.0110 | 5.1084 | 59.8582 | 32.0000 | 41.7048 | 529026 | 6203.0000 |  |  |  |  | multiple live artifact candidates found; using experiments/20260318_1k_1566_20ep_1024_full19/unetpp_boundary_inst; no inference-profile artifact saved; no training-memory artifact saved |
| 1024 | uoais | from-scratch | ok | 17.6743 | 50.3255 | 5.9743 | 31.4972 | 70.2212 | 22.9806 | 77.3475 | 49.0000 | 59.9937 | 81686167 | 5011.0000 | 25226.0000 | 46.8559 | 1976.7129 | 21.3420 |  |
| 1024 | yolov8_seg_l | fine-tuned | ok | 40.5626 | 78.6088 | 39.8740 | 61.7331 | 83.2272 | 70.2146 | 95.8427 | 76.0000 | 84.7757 | 45936819 | 1739.0000 | 26726.4000 | 24.4985 | 932.1382 | 40.8188 | multiple live artifact candidates found; using experiments/20260318_1k_1566_20ep_1024_full19/yolov8_seg_l_pretrained |
| 1024 | yolov8_seg_m | fine-tuned | ok | 40.0791 | 77.5599 | 39.1804 | 60.8902 | 83.0663 | 68.9139 | 95.4942 | 75.0000 | 84.0153 | 27240227 | 1718.0000 | 22732.8000 | 18.0327 | 929.5957 | 55.4547 | multiple live artifact candidates found; using experiments/20260318_1k_1566_20ep_1024_full19/yolov8_seg_m_pretrained |
| 1024 | yolov8_seg_n | fine-tuned | ok | 32.2371 | 69.1427 | 24.8334 | 51.3223 | 79.0336 | 58.3787 | 88.6830 | 66.0000 | 75.6784 | 3263811 | 1698.0000 | 14745.6000 | 19.1931 | 806.0615 | 52.1020 | multiple live artifact candidates found; using experiments/20260318_1k_1566_20ep_1024_full19/yolov8_seg_n_pretrained |
| 1024 | yolov8_seg_s | fine-tuned | ok | 36.4499 | 75.0820 | 31.5606 | 56.8499 | 81.7390 | 65.2856 | 94.0311 | 72.0000 | 81.5539 | 11790483 | 1711.0000 | 19353.6000 | 16.0542 | 852.2231 | 62.2891 | multiple live artifact candidates found; using experiments/20260318_1k_1566_20ep_1024_full19/yolov8_seg_s_pretrained |
| 1024 | yolov8_seg_x | fine-tuned | ok | 40.3242 | 78.7571 | 38.7892 | 61.9894 | 83.3218 | 70.8741 | 95.2083 | 77.0000 | 85.1415 | 71751811 | 1822.0000 | 28569.6000 | 23.6503 | 1019.9507 | 42.2827 | multiple live artifact candidates found; using experiments/20260318_1k_1566_20ep_1024_full19/yolov8_seg_x_pretrained |
| 512 | magformer_depthnorm_on | fine-tuned | missing |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | live artifact missing in current tree; rerun required |
| 512 | magformer_lightdepth_convnextlite_spatialgate_edge_validhole | fine-tuned | missing |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | live artifact missing in current tree; rerun required |
| 512 | magformer_lightdepth_mobilenetv3_sagate_edge_validhole | fine-tuned | missing |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | live artifact missing in current tree; rerun required |
| 512 | magformer_lightdepth_mobilenetv3_spatialgate_edge_validhole | fine-tuned | missing |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | live artifact missing in current tree; rerun required |
| 512 | magformer_nodpth_ref | fine-tuned | missing |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | live artifact missing in current tree; rerun required |
| 512 | mask2former | fine-tuned | missing |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | live artifact missing in current tree; rerun required |
| 512 | maskrcnn | fine-tuned | missing |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | live artifact missing in current tree; rerun required |
| 512 | mgm_mask2former_depthnorm_on | fine-tuned | missing |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | live artifact missing in current tree; rerun required |
| 512 | mgm_mask2former_nodpth_ref | fine-tuned | missing |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | live artifact missing in current tree; rerun required |
| 512 | msmformer | from-scratch | missing |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | live artifact missing in current tree; rerun required |
| 512 | ucn | from-scratch | missing |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | live artifact missing in current tree; rerun required |
| 512 | unet_boundary_inst | from-scratch | missing |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | live artifact missing in current tree; rerun required |
| 512 | unet_semantic_inst | from-scratch | missing |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | live artifact missing in current tree; rerun required |
| 512 | unetpp_boundary_inst | from-scratch | missing |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | live artifact missing in current tree; rerun required |
| 512 | uoais | from-scratch | missing |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | live artifact missing in current tree; rerun required |
| 512 | yolov8_seg_l | fine-tuned | missing |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | live artifact missing in current tree; rerun required |
| 512 | yolov8_seg_m | fine-tuned | missing |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | live artifact missing in current tree; rerun required |
| 512 | yolov8_seg_n | fine-tuned | missing |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | live artifact missing in current tree; rerun required |
| 512 | yolov8_seg_s | fine-tuned | missing |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | live artifact missing in current tree; rerun required |
| 512 | yolov8_seg_x | fine-tuned | missing |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | live artifact missing in current tree; rerun required |
| 256 | magformer_depthnorm_on | fine-tuned | missing |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | live artifact missing in current tree; rerun required |
| 256 | magformer_lightdepth_convnextlite_spatialgate_edge_validhole | fine-tuned | missing |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | live artifact missing in current tree; rerun required |
| 256 | magformer_lightdepth_mobilenetv3_sagate_edge_validhole | fine-tuned | missing |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | live artifact missing in current tree; rerun required |
| 256 | magformer_lightdepth_mobilenetv3_spatialgate_edge_validhole | fine-tuned | missing |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | live artifact missing in current tree; rerun required |
| 256 | magformer_nodpth_ref | fine-tuned | missing |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | live artifact missing in current tree; rerun required |
| 256 | mask2former | fine-tuned | missing |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | live artifact missing in current tree; rerun required |
| 256 | maskrcnn | fine-tuned | missing |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | live artifact missing in current tree; rerun required |
| 256 | mgm_mask2former_depthnorm_on | fine-tuned | missing |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | live artifact missing in current tree; rerun required |
| 256 | mgm_mask2former_nodpth_ref | fine-tuned | missing |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | live artifact missing in current tree; rerun required |
| 256 | msmformer | from-scratch | missing |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | live artifact missing in current tree; rerun required |
| 256 | ucn | from-scratch | missing |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | live artifact missing in current tree; rerun required |
| 256 | unet_boundary_inst | from-scratch | missing |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | live artifact missing in current tree; rerun required |
| 256 | unet_semantic_inst | from-scratch | missing |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | live artifact missing in current tree; rerun required |
| 256 | unetpp_boundary_inst | from-scratch | missing |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | live artifact missing in current tree; rerun required |
| 256 | uoais | from-scratch | missing |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | live artifact missing in current tree; rerun required |
| 256 | yolov8_seg_l | fine-tuned | missing |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | live artifact missing in current tree; rerun required |
| 256 | yolov8_seg_m | fine-tuned | missing |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | live artifact missing in current tree; rerun required |
| 256 | yolov8_seg_n | fine-tuned | missing |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | live artifact missing in current tree; rerun required |
| 256 | yolov8_seg_s | fine-tuned | missing |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | live artifact missing in current tree; rerun required |
| 256 | yolov8_seg_x | fine-tuned | missing |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | live artifact missing in current tree; rerun required |
