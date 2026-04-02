# 2026-03-29 All-Resolution Instance Segmentation Results

结论先说清楚：

- 现在不需要做一轮全量重训，也不需要把三档分辨率全部重评一遍。现有 `512` 和 `256` 结果本身就是单卡最终导出结果，`1024` 的大部分结果也已经落成可复核的最终 artifact。
- 但 `1024` 里的 MAGFormer 家族有一个边界要说清楚：这些行现在能当“最终 checkpoint 导出指标”来用，不能当“修复后 best-checkpoint 指标”来用。原始 DDP 训练当时没有可靠的在线 best 选择，所以如果要发布“修复后 best 模型”数字，还得把这 5 个 `1024` MAGFormer 变体重训。
- `1024` 的 `msmformer` 和三档 `ucn` 是后续单独补跑并并回来的，不该再被旧 `summary` 里的失效行覆盖。

## Whether Metrics Need To Move

- `512` 和 `256`：不用重训，也不用重评。当前表里的数直接来自现有最终 artifact。
- `1024` 非 MAGFormer-DPP 行：不用重训。当前表里的数可以直接复核。
- `1024` MAGFormer 家族：当前表保留现有最终 checkpoint 导出指标。如果后续目标是“修复后 best-checkpoint 排名”，那就需要重训，不是只跑一次离线评估就能补回来。

## Final Summary Table

| Resolution | Model | Status | segm AP | AP50 | AP75 | bbox AP | bbox AP50 | bbox AP75 | P@50 | R@50 | F1@50 | Params | Train sec | Train mem MB | Infer ms | Infer mem MB | FPS | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1024 | magformer_depthnorm_on | ok | 68.4177 | 87.9741 | 77.7181 | 62.2810 | 82.9465 | 69.2393 | 96.6455 | 88.0000 | 92.1203 | 79696062 | 9453.0000 | 29975.9300 | 467.7866 | 3399.0015 | 2.1377 | final checkpoint export; original DDP run had no trustworthy online best-checkpoint selection |
| 1024 | magformer_lightdepth_convnextlite_spatialgate_edge_validhole | ok | 65.4526 | 86.8597 | 74.6540 | 57.6493 | 81.9553 | 66.3226 | 95.9131 | 86.0000 | 90.6864 | 50237888 | 8602.0000 | 15569.7500 | 410.9326 | 2455.9868 | 2.4335 | final checkpoint export; original DDP run had no trustworthy online best-checkpoint selection |
| 1024 | magformer_lightdepth_mobilenetv3_sagate_edge_validhole | ok | 64.5032 | 85.8691 | 73.6259 | 57.6543 | 82.0998 | 66.2676 | 95.9349 | 85.0000 | 90.1370 | 49958798 | 8989.0000 | 14138.1800 | 399.9963 | 2453.9731 | 2.5000 | final checkpoint export; original DDP run had no trustworthy online best-checkpoint selection |
| 1024 | magformer_lightdepth_mobilenetv3_spatialgate_edge_validhole | ok | 64.7399 | 85.9040 | 73.6656 | 57.6262 | 82.0185 | 65.5521 | 96.7993 | 85.0000 | 90.5168 | 49912530 | 8932.0000 | 14136.6400 | 407.3516 | 2453.7969 | 2.4549 | final checkpoint export; original DDP run had no trustworthy online best-checkpoint selection |
| 1024 | magformer_nodpth_ref | ok | 48.7786 | 74.9858 | 54.3229 | 44.6381 | 74.8140 | 47.7525 | 91.2694 | 73.0000 | 81.1188 | 79628990 | 8854.0000 | 14059.8000 | 396.4242 | 2568.5430 | 2.5226 | final checkpoint export; original DDP run had no trustworthy online best-checkpoint selection |
| 1024 | mask2former | ok | 58.7554 | 80.5301 | 66.0488 | 0.0000 | 0.0000 | 0.0000 | 93.4138 | 79.0000 | 85.6044 | 44056196 | 6944.0000 | 35761.0000 | 66.5025 | 2711.4526 | 15.0370 |  |
| 1024 | maskrcnn | ok | 54.3322 | 78.4134 | 61.4813 | 59.4577 | 83.6528 | 67.8010 | 94.8226 | 76.0000 | 84.3743 | 44024278 | 1850.0000 | 9476.0000 | 30.4008 | 1579.5366 | 32.8938 |  |
| 1024 | mgm_mask2former_depthnorm_on | ok | 72.8081 | 87.9172 | 78.6858 |  |  |  | 96.8018 | 87.0000 | 91.6395 | 79696062 | 11053.0000 | 61279.0000 | 152.5003 | 2777.1172 | 6.5574 |  |
| 1024 | mgm_mask2former_nodpth_ref | ok | 39.6413 | 68.6351 | 40.8917 |  |  |  | 87.0296 | 66.0000 | 75.0698 | 79628990 | 8176.0000 | 29857.0000 | 81.1941 | 2778.3604 | 12.3162 |  |
| 1024 | msmformer | ok | 28.3180 | 31.6794 | 30.6859 | 24.9071 | 31.6387 | 29.3970 | 99.8993 | 31.0000 | 47.3170 | 52521730 | 162557.0000 |  |  |  |  | separate repaired 1024 run; no inference-profile artifact saved |
| 1024 | ucn | ok | 6.7927 | 21.0988 | 2.5057 | 9.2446 | 25.3330 | 5.4762 | 51.8383 | 40.0000 | 45.1562 | 42635008 | 8935.0000 |  |  |  |  | separate repair run; no training-memory or inference-profile artifact saved |
| 1024 | unet_boundary_inst | ok | 13.5987 | 35.0566 | 8.7096 | 12.6285 | 36.6316 | 6.2889 | 62.9120 | 40.0000 | 48.9055 | 1942594 | 6626.0000 |  |  |  |  |  |
| 1024 | unet_semantic_inst | ok | 4.0611 | 5.2063 | 4.0612 | 3.7383 | 4.8881 | 4.0718 | 65.3963 | 4.0000 | 7.5389 | 1942594 | 2693.0000 |  |  |  |  |  |
| 1024 | unetpp_boundary_inst | ok | 11.0563 | 30.3917 | 6.3648 | 10.6254 | 32.7919 | 4.9186 | 59.8582 | 32.0000 | 41.7048 | 529026 | 6203.0000 |  |  |  |  |  |
| 1024 | uoais | ok | 17.4346 | 50.1926 | 5.7692 | 31.3865 | 69.8880 | 22.7097 | 77.3475 | 49.0000 | 59.9937 | 81686167 | 5011.0000 | 25226.0000 | 46.8559 | 1976.7129 | 21.3420 |  |
| 1024 | yolov8_seg_l | ok | 51.8850 | 67.0280 |  | 56.3960 | 66.4250 |  | 95.8427 | 76.0000 | 84.7757 | 45936819 | 1739.0000 | 26726.4000 | 24.4985 | 932.1382 | 40.8188 |  |
| 1024 | yolov8_seg_m | ok | 51.3570 | 66.4530 |  | 55.6230 | 65.9650 |  | 95.4942 | 75.0000 | 84.0153 | 27240227 | 1718.0000 | 22732.8000 | 18.0327 | 929.5957 | 55.4547 |  |
| 1024 | yolov8_seg_n | ok | 44.5110 | 61.4640 |  | 48.9290 | 62.0920 |  | 88.6830 | 66.0000 | 75.6784 | 3263811 | 1698.0000 | 14745.6000 | 19.1931 | 806.0615 | 52.1020 |  |
| 1024 | yolov8_seg_s | ok | 48.7960 | 64.3950 |  | 53.0890 | 64.5220 |  | 94.0311 | 72.0000 | 81.5539 | 11790483 | 1711.0000 | 19353.6000 | 16.0542 | 852.2231 | 62.2891 |  |
| 1024 | yolov8_seg_x | ok | 52.3590 | 67.4290 |  | 56.8610 | 66.6510 |  | 95.2083 | 77.0000 | 85.1415 | 71751811 | 1822.0000 | 28569.6000 | 23.6503 | 1019.9507 | 42.2827 |  |
| 512 | magformer_depthnorm_on | ok | 56.2581 | 80.8099 | 65.0626 | 54.7114 | 79.0792 | 62.9734 | 95.3357 | 80.0000 | 86.9972 | 79696062 | 5206.0000 | 16326.2900 | 95.3639 | 1088.3413 | 10.4861 |  |
| 512 | magformer_lightdepth_convnextlite_spatialgate_edge_validhole | ok | 49.6199 | 77.5751 | 57.4966 | 48.4337 | 77.7504 | 54.6896 | 93.7167 | 76.0000 | 83.9336 | 50237888 | 4250.0000 | 8955.8200 | 120.8917 | 764.4854 | 8.2719 |  |
| 512 | magformer_lightdepth_mobilenetv3_sagate_edge_validhole | ok | 45.9682 | 74.5452 | 52.3479 | 46.0311 | 77.0270 | 50.5030 | 91.6503 | 74.0000 | 81.8848 | 49958798 | 4286.0000 | 8231.4600 | 136.3089 | 763.6997 | 7.3363 |  |
| 512 | magformer_lightdepth_mobilenetv3_spatialgate_edge_validhole | ok | 46.2109 | 75.3178 | 52.3239 | 46.4748 | 77.8465 | 50.8607 | 93.3738 | 73.0000 | 81.9394 | 49912530 | 4382.0000 | 8231.7800 | 123.7388 | 763.2954 | 8.0815 |  |
| 512 | magformer_nodpth_ref | ok | 29.7106 | 60.7768 | 26.5193 | 33.9887 | 69.3259 | 29.8921 | 82.9945 | 59.0000 | 68.9699 | 79628990 | 4354.0000 | 8184.6100 | 63.4500 | 878.3945 | 15.7604 |  |
| 512 | mask2former | missing |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | train_net.py missing; no finished artifact |
| 512 | maskrcnn | ok | 39.8201 | 67.1736 | 43.7757 | 44.6155 | 73.5282 | 48.6551 | 90.6812 | 66.0000 | 76.3966 | 44024278 | 1093.0000 | 6379.0000 | 29.8273 | 492.0513 | 33.5263 |  |
| 512 | mgm_mask2former_depthnorm_on | ok | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 79696062 | 5302.0000 | 21423.0000 | 61.8716 | 930.3682 | 16.1625 |  |
| 512 | mgm_mask2former_nodpth_ref | ok | 24.4955 | 54.7578 | 18.9561 | 30.3763 | 67.3244 | 23.6369 | 77.1864 | 54.0000 | 63.5442 | 79628990 | 4211.0000 | 13184.0000 | 43.3179 | 930.0293 | 23.0851 |  |
| 512 | msmformer | ok | 23.4265 | 31.6461 | 28.4795 | 22.1260 | 31.6733 | 26.8826 | 98.8575 | 31.0000 | 47.1992 | 52521730 | 19307.0000 | 69666.0000 | 448.8246 | 5253.7456 | 2.2280 |  |
| 512 | ucn | ok | 4.7175 | 19.0162 | 0.3902 | 6.6656 | 22.2386 | 1.8452 | 49.1415 | 38.0000 | 42.8585 | 42635008 | 2307.0000 |  |  |  |  | separate repair run; no training-memory or inference-profile artifact saved |
| 512 | unet_boundary_inst | ok | 18.3631 | 42.3709 | 13.5030 | 18.3265 | 46.3027 | 11.6764 | 74.2559 | 45.0000 | 56.0394 | 1942594 | 1651.0000 | 587.4683 | 13.5814 | 587.4683 | 73.6302 |  |
| 512 | unet_semantic_inst | ok | 3.3070 | 4.3205 | 3.2749 | 3.2781 | 4.0519 | 3.2964 | 66.3176 | 4.0000 | 7.5449 | 1942594 | 580.0000 | 587.4683 | 13.5583 | 587.4683 | 73.7554 |  |
| 512 | unetpp_boundary_inst | ok | 20.4649 | 47.8383 | 14.4009 | 20.7485 | 51.1938 | 13.3403 | 74.6182 | 50.0000 | 59.8774 | 529026 | 1673.0000 | 666.0771 | 18.9730 | 666.0771 | 52.7065 |  |
| 512 | uoais | ok | 8.6873 | 33.2150 | 0.9685 | 19.1145 | 54.5079 | 7.8205 | 53.3198 | 36.0000 | 42.9807 | 81686167 | 1773.0000 | 9744.0000 | 31.6082 | 739.8848 | 31.6374 |  |
| 512 | yolov8_seg_l | missing |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | Ultralytics import error; no finished artifact |
| 512 | yolov8_seg_m | missing |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | Ultralytics import error; no finished artifact |
| 512 | yolov8_seg_n | missing |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | Ultralytics import error; no finished artifact |
| 512 | yolov8_seg_s | missing |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | Ultralytics import error; no finished artifact |
| 512 | yolov8_seg_x | missing |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | Ultralytics import error; no finished artifact |
| 256 | magformer_depthnorm_on | ok | 36.9747 | 68.6798 | 38.8638 | 39.2586 | 73.1055 | 39.6616 | 87.9502 | 68.0000 | 76.6990 | 79696062 | 4076.0000 | 6161.2600 | 57.5104 | 504.6279 | 17.3882 |  |
| 256 | magformer_lightdepth_convnextlite_spatialgate_edge_validhole | ok | 25.6417 | 58.2669 | 19.0901 | 30.3259 | 67.3388 | 22.2124 | 82.0482 | 57.0000 | 67.2680 | 50237888 | 3364.0000 | 3924.8200 | 65.6482 | 340.5454 | 15.2327 |  |
| 256 | magformer_lightdepth_mobilenetv3_sagate_edge_validhole | ok | 20.2808 | 51.2659 | 11.9473 | 25.4783 | 63.0557 | 15.5193 | 74.6629 | 51.0000 | 60.6035 | 49958798 | 3425.0000 | 3732.2400 | 64.1295 | 339.5317 | 15.5935 |  |
| 256 | magformer_lightdepth_mobilenetv3_spatialgate_edge_validhole | ok | 20.3533 | 51.8105 | 11.9549 | 26.0809 | 64.5847 | 15.8840 | 79.2583 | 50.0000 | 61.3178 | 49912530 | 3406.0000 | 3743.0000 | 57.6821 | 339.3555 | 17.3364 |  |
| 256 | magformer_nodpth_ref | ok | 13.3019 | 38.0674 | 6.3806 | 18.5915 | 53.2373 | 8.7476 | 57.1703 | 41.0000 | 47.7534 | 79628990 | 3388.0000 | 3807.9500 | 54.6146 | 452.6641 | 18.3101 |  |
| 256 | mask2former | missing |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | not rerun at this resolution |
| 256 | maskrcnn | ok | 22.4546 | 52.6769 | 15.7490 | 28.0743 | 65.1423 | 20.1136 | 73.6613 | 52.0000 | 60.9637 | 44024278 | 667.0000 | 4273.0000 | 27.5066 | 393.7295 | 36.3549 |  |
| 256 | mgm_mask2former_depthnorm_on | ok | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 79696062 | 53372.0000 | 11397.0000 | 59.6806 | 466.1812 | 16.7559 |  |
| 256 | mgm_mask2former_nodpth_ref | ok | 10.2470 | 32.1353 | 3.9993 | 15.6422 | 47.8489 | 6.6038 | 48.2392 | 38.0000 | 42.5118 | 79628990 | 3329.0000 | 9006.0000 | 42.7827 | 465.9043 | 23.3739 |  |
| 256 | msmformer | ok | 12.0576 | 24.2800 | 10.6154 | 12.8011 | 25.5218 | 10.5370 | 94.6205 | 24.0000 | 38.2884 | 52521730 | 4228.0000 | 34692.0000 | 121.3253 | 1409.2456 | 8.2423 |  |
| 256 | ucn | ok | 0.8764 | 4.9696 | 0.0254 | 2.0854 | 9.2329 | 0.2123 | 24.4534 | 19.0000 | 21.3845 | 42635008 | 653.0000 |  |  |  |  | separate repair run; no training-memory or inference-profile artifact saved |
| 256 | unet_boundary_inst | ok | 18.3659 | 43.3551 | 12.9739 | 20.8953 | 50.5473 | 14.7987 | 75.9319 | 50.0000 | 60.2960 | 1942594 | 675.0000 | 587.4683 | 13.7461 | 587.4683 | 72.7479 |  |
| 256 | unet_semantic_inst | ok | 2.4128 | 3.2680 | 2.4894 | 2.3646 | 3.2327 | 2.4584 | 60.9504 | 3.0000 | 5.7185 | 1942594 | 188.0000 | 587.4683 | 13.6755 | 587.4683 | 73.1236 |  |
| 256 | unetpp_boundary_inst | ok | 24.1087 | 52.8477 | 18.8725 | 26.2888 | 59.3234 | 20.0294 | 78.9552 | 56.0000 | 65.5253 | 529026 | 690.0000 | 666.0771 | 18.9974 | 666.0771 | 52.6388 |  |
| 256 | uoais | ok | 0.5311 | 3.1123 | 0.0127 | 2.7123 | 12.4112 | 0.3688 | 12.3521 | 19.0000 | 14.9713 | 81686167 | 852.0000 | 5855.0000 | 34.3776 | 583.1094 | 29.0887 |  |
| 256 | yolov8_seg_l | missing |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | not rerun at this resolution |
| 256 | yolov8_seg_m | missing |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | not rerun at this resolution |
| 256 | yolov8_seg_n | missing |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | not rerun at this resolution |
| 256 | yolov8_seg_s | missing |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | not rerun at this resolution |
| 256 | yolov8_seg_x | missing |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | not rerun at this resolution |

## Coverage Notes

- 表里所有 `P@50 / R@50 / F1@50` 都直接按保存下来的 `coco_instances_results.json` 重新算过一遍，避免不同来源文件口径不一。
- `1024` 的 `msmformer` 和三档 `ucn` 没有现成的推理测速 artifact，所以这些行的 `Infer ms / Infer mem MB / FPS` 暂时留空。
- 同样，`ucn` 和修复版 `1024 msmformer` 没有统一保存训练峰值显存，所以 `Train mem MB` 留空，不做猜测。

## Artifact Sources

- `1024 full19`: `output/experiments/20260318_1k_1566_20ep_1024_full19/summary_20260318_1k_1566_20ep_1024_full19.json`
- `1024 repaired msmformer`: `.worktrees/ucn-msmformer-repair/output/experiments/20260325_repaired_pretrained_20260318_1k_1566_20ep_1024/msmformer/metrics.cocoeval.json`
- `1024 repaired ucn`: `.worktrees/ucn-msmformer-repair/output/experiments/20260325_repaired_pretrained_20260318_1k_1566_20ep_1024/ucn/metrics.cocoeval.json`
- `512 full19`: `.worktrees/ucn-msmformer-repair/output/experiments/20260326_repaired_pretrained_20260318_1k_1566_20ep_512_full19/summary_20260326_repaired_pretrained_20260318_1k_1566_20ep_512_full19.json`
- `512 repaired ucn`: `.worktrees/ucn-msmformer-repair/output/experiments/20260326_repaired_pretrained_20260318_1k_1566_20ep_512_multires/ucn/metrics.cocoeval.json`
- `256 full19`: `.worktrees/ucn-msmformer-repair/output/experiments/20260326_repaired_pretrained_20260318_1k_1566_20ep_256_full19/summary_20260326_repaired_pretrained_20260318_1k_1566_20ep_256_full19.json`
- `256 repaired ucn`: `.worktrees/ucn-msmformer-repair/output/experiments/20260326_repaired_pretrained_20260318_1k_1566_20ep_256_multires/ucn/metrics.cocoeval.json`
