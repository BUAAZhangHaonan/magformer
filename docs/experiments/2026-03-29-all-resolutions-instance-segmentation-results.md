# 2026-03-29 全部分辨率实例分割结果汇总

本表汇总了当前仓库里已经跑完并产出最终 `COCOEval` 指标的实例分割实验结果，覆盖 `1024`、`512`、`256` 三个分辨率。

说明：
- `1024` 的 `msmformer` 使用修复后补跑结果，替换了旧 `extended_metrics_table.md` 里失效的 `0.0000` 行。
- `UCN` 不在原始 `full19` 总表中，这里将其三套补跑结果单独并入。
- `512` 与 `256` 只纳入已经产出最终 `metrics.cocoeval.json` 的完成实验；失败或未完成条目单独列在文末，不混入排名。
- 当前导出的 `summary_*.json`、`extended_metrics_table.*` 与 `metrics.cocoeval.json` 都没有统一保存 `AR100`，因此本文只汇总可直接复核的最终 `AP` 指标。

## 最佳配置概览

| 分辨率 | 最佳模型 | segm AP | segm AP50 | segm AP75 |
| --- | --- | ---: | ---: | ---: |
| 1024 | mgm_mask2former_depthnorm_on | 72.8081 | 87.9172 | 78.6858 |
| 512 | magformer_depthnorm_on | 56.2581 | 80.8099 | 65.0626 |
| 256 | magformer_depthnorm_on | 36.9747 | 68.6798 | 38.8638 |

全局最佳结果是 `1024 / mgm_mask2former_depthnorm_on`，`segm AP = 72.8081`。

## 1024 已完成结果

| Model | segm AP | AP50 | AP75 | bbox AP | bbox AP50 | bbox AP75 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mgm_mask2former_depthnorm_on | 72.8081 | 87.9172 | 78.6858 | - | - | - |
| magformer_depthnorm_on | 68.4177 | 87.9741 | 77.7181 | 62.2810 | 82.9465 | 69.2393 |
| magformer_lightdepth_convnextlite_spatialgate_edge_validhole | 65.4526 | 86.8597 | 74.6540 | 57.6493 | 81.9553 | 66.3226 |
| magformer_lightdepth_mobilenetv3_spatialgate_edge_validhole | 64.7399 | 85.9040 | 73.6656 | 57.6262 | 82.0185 | 65.5521 |
| magformer_lightdepth_mobilenetv3_sagate_edge_validhole | 64.5032 | 85.8691 | 73.6259 | 57.6543 | 82.0998 | 66.2676 |
| mask2former | 58.7554 | 80.5301 | 66.0488 | 0.0000 | 0.0000 | 0.0000 |
| maskrcnn | 54.3322 | 78.4134 | 61.4813 | 59.4577 | 83.6528 | 67.8010 |
| yolov8_seg_x | 52.3590 | 67.4290 | - | 56.8610 | 66.6510 | - |
| yolov8_seg_l | 51.8850 | 67.0280 | - | 56.3960 | 66.4250 | - |
| yolov8_seg_m | 51.3570 | 66.4530 | - | 55.6230 | 65.9650 | - |
| yolov8_seg_s | 48.7960 | 64.3950 | - | 53.0890 | 64.5220 | - |
| magformer_nodpth_ref | 48.7786 | 74.9858 | 54.3229 | 44.6381 | 74.8140 | 47.7525 |
| yolov8_seg_n | 44.5110 | 61.4640 | - | 48.9290 | 62.0920 | - |
| mgm_mask2former_nodpth_ref | 39.6413 | 68.6351 | 40.8917 | - | - | - |
| msmformer | 28.3180 | 31.6794 | 30.6859 | 24.9071 | 32.4959 | 25.9942 |
| uoais | 17.4346 | 50.1926 | 5.7692 | 31.3865 | 69.8880 | 22.7097 |
| unet_boundary_inst | 13.5987 | 35.0566 | 8.7096 | 12.6285 | 36.6316 | 6.2889 |
| unetpp_boundary_inst | 11.0563 | 30.3917 | 6.3648 | 10.6254 | 32.7919 | 4.9186 |
| ucn | 6.7927 | 21.0988 | 2.5057 | 9.2446 | 27.0517 | 4.8243 |
| unet_semantic_inst | 4.0611 | 5.2063 | 4.0612 | 3.7383 | 4.8881 | 4.0718 |

## 512 已完成结果

| Model | segm AP | AP50 | AP75 | bbox AP | bbox AP50 | bbox AP75 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| magformer_depthnorm_on | 56.2581 | 80.8099 | 65.0626 | 54.7114 | 79.0792 | 62.9734 |
| magformer_lightdepth_convnextlite_spatialgate_edge_validhole | 49.6199 | 77.5751 | 57.4966 | 48.4337 | 77.7504 | 54.6896 |
| magformer_lightdepth_mobilenetv3_spatialgate_edge_validhole | 46.2109 | 75.3178 | 52.3239 | 46.4748 | 77.8465 | 50.8607 |
| magformer_lightdepth_mobilenetv3_sagate_edge_validhole | 45.9682 | 74.5452 | 52.3479 | 46.0311 | 77.0270 | 50.5030 |
| maskrcnn | 39.8201 | 67.1736 | 43.7757 | 44.6155 | 73.5282 | 48.6551 |
| magformer_nodpth_ref | 29.7106 | 60.7768 | 26.5193 | 33.9887 | 69.3259 | 29.8921 |
| mgm_mask2former_nodpth_ref | 24.4955 | 54.7578 | 18.9561 | 30.3763 | 67.3244 | 23.6369 |
| msmformer | 23.4265 | 31.6461 | 28.4795 | 22.1260 | 31.6733 | 26.8826 |
| unetpp_boundary_inst | 20.4649 | 47.8383 | 14.4009 | 20.7485 | 51.1938 | 13.3403 |
| unet_boundary_inst | 18.3631 | 42.3709 | 13.5030 | 18.3265 | 46.3027 | 11.6764 |
| uoais | 8.6873 | 33.2150 | 0.9685 | 19.1145 | 54.5079 | 7.8205 |
| ucn | 4.7175 | 19.0162 | 0.3902 | 6.6656 | 24.2326 | 1.7751 |
| unet_semantic_inst | 3.3070 | 4.3205 | 3.2749 | 3.2781 | 4.0519 | 3.2964 |
| mgm_mask2former_depthnorm_on | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## 256 已完成结果

| Model | segm AP | AP50 | AP75 | bbox AP | bbox AP50 | bbox AP75 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| magformer_depthnorm_on | 36.9747 | 68.6798 | 38.8638 | 39.2586 | 73.1055 | 39.6616 |
| magformer_lightdepth_convnextlite_spatialgate_edge_validhole | 25.6417 | 58.2669 | 19.0901 | 30.3259 | 67.3388 | 22.2124 |
| unetpp_boundary_inst | 24.1087 | 52.8477 | 18.8725 | 26.2888 | 59.3234 | 20.0294 |
| maskrcnn | 22.4546 | 52.6769 | 15.7490 | 28.0743 | 65.1423 | 20.1136 |
| magformer_lightdepth_mobilenetv3_spatialgate_edge_validhole | 20.3533 | 51.8105 | 11.9549 | 26.0809 | 64.5847 | 15.8840 |
| magformer_lightdepth_mobilenetv3_sagate_edge_validhole | 20.2808 | 51.2659 | 11.9473 | 25.4783 | 63.0557 | 15.5193 |
| unet_boundary_inst | 18.3659 | 43.3551 | 12.9739 | 20.8953 | 50.5473 | 14.7987 |
| magformer_nodpth_ref | 13.3019 | 38.0674 | 6.3806 | 18.5915 | 53.2373 | 8.7476 |
| msmformer | 12.0576 | 24.2800 | 10.6154 | 12.8011 | 25.5218 | 10.5370 |
| mgm_mask2former_nodpth_ref | 10.2470 | 32.1353 | 3.9993 | 15.6422 | 47.8489 | 6.6038 |
| unet_semantic_inst | 2.4128 | 3.2680 | 2.4894 | 2.3646 | 3.2327 | 2.4584 |
| ucn | 0.8764 | 4.9696 | 0.0254 | 2.0854 | 8.4171 | 0.0716 |
| uoais | 0.5311 | 3.1123 | 0.0127 | 2.7123 | 12.4112 | 0.3688 |
| mgm_mask2former_depthnorm_on | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## 关键趋势

- `msmformer` 修复后不再是 `0.0000`，三档分辨率分别达到 `28.3180 / 23.4265 / 12.0576`。
- `UCN` 三档分辨率分别达到 `6.7927 / 4.7175 / 0.8764`，明显弱于当前主力实例分割 baseline。
- `magformer_depthnorm_on` 在 `512` 和 `256` 分辨率下都是最佳模型。
- `mgm_mask2former_depthnorm_on` 在 `1024` 最强，但在 `512` 与 `256` 上都塌到 `0.0000`。
- 分辨率下降会显著拉低绝大多数模型的 `segm AP`，但 `magformer` 系列的相对稳定性明显好于 `UCN` 与 `uoais`。

## 未纳入排名的失败或未完成实验

| 分辨率 | Model | 状态 | 原因 |
| --- | --- | --- | --- |
| 512 | official_mask2former_pretrained | failed | `baselines/Mask2Former/train_net.py` 缺失，无法启动 |
| 512 | yolov8_seg_x_pretrained | failed | `ImportError: cannot import name '__version__' from 'ultralytics'` |
| 512 | yolov8_seg_l_pretrained | failed | `ImportError: cannot import name '__version__' from 'ultralytics'` |
| 512 | yolov8_seg_m_pretrained | failed | `ImportError: cannot import name '__version__' from 'ultralytics'` |
| 512 | yolov8_seg_s_pretrained | failed | `ImportError: cannot import name '__version__' from 'ultralytics'` |
| 512 | yolov8_seg_n_pretrained | failed | `ImportError: cannot import name '__version__' from 'ultralytics'` |
| 256 | maskrcnn_pretrained | failed | polygon mask 标注格式与 Detectron2 `mask_format='polygon'` 不兼容 |

## 结果来源

- `1024 full19`：`output/experiments/20260318_1k_1566_20ep_1024_full19/summary_20260318_1k_1566_20ep_1024_full19.json`
- `1024 repaired msmformer`：`.worktrees/ucn-msmformer-repair/output/experiments/20260325_repaired_pretrained_20260318_1k_1566_20ep_1024/msmformer/metrics.cocoeval.json`
- `1024 ucn`：`.worktrees/ucn-msmformer-repair/output/experiments/20260325_repaired_pretrained_20260318_1k_1566_20ep_1024/ucn/metrics.cocoeval.json`
- `512 full19`：`.worktrees/ucn-msmformer-repair/output/experiments/20260326_repaired_pretrained_20260318_1k_1566_20ep_512_full19/summary_20260326_repaired_pretrained_20260318_1k_1566_20ep_512_full19.json`
- `512 ucn`：`.worktrees/ucn-msmformer-repair/output/experiments/20260326_repaired_pretrained_20260318_1k_1566_20ep_512_multires/ucn/metrics.cocoeval.json`
- `256 full19`：`.worktrees/ucn-msmformer-repair/output/experiments/20260326_repaired_pretrained_20260318_1k_1566_20ep_256_full19/summary_20260326_repaired_pretrained_20260318_1k_1566_20ep_256_full19.json`
- `256 ucn`：`.worktrees/ucn-msmformer-repair/output/experiments/20260326_repaired_pretrained_20260318_1k_1566_20ep_256_multires/ucn/metrics.cocoeval.json`
