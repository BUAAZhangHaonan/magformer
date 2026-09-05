# MAGFormer 基线结果全集（三口径）

> 评测口径说明：segm AP = COCO mask mAP[.5:.95]，maxDets [1,10,100]。
> 口径一（1566）：train 1,261 / val 149，20ep，1024 与 512 双分辨率。
> 口径二（32254 全量）：train 25,654 / val 3,276，官方实现、官方合理预算、全量 val。
> 口径三：自研探索（非基线），仅供消融参考，**不得**作为基线引用。

## 口径一 · 1566（2026-04-12 正式收官表 + 100ep 修复版）

| model_id | segm AP @1024 | @512 | FPS@1024 |
|---|---:|---:|---:|
| mgm_mask2former_depthnorm | **72.80** | 69.90 | 6.6 |
| **magformer_depthnorm_on** | **68.42** | 59.91 | 2.1 |
| magformer_lightdepth_convnextlite | 65.45 | 59.74 | 2.4 |
| magformer_lightdepth_mobilenetv3(spatialgate) | 64.74 | 60.36 | 2.5 |
| magformer_lightdepth_mobilenetv3(sagate) | 64.50 | 59.75 | 2.5 |
| magformer_nodpth_ref | 59.52 | 53.88 | 2.6 |
| mask2former (44M) | 58.76 | 43.08 | 15.0 |
| maskrcnn (44M) | 54.10 | 38.78 | 32.9 |
| yolov8_seg_l / x / m / s / n | 40.56 / 40.32 / 40.08 / 36.45 / 32.24 | 42.68 / 42.93 / 42.11 / 40.21 / 35.88 | 40-62 |
| mgm_mask2former_nodpth_ref | 39.61 | 32.22 | 12.3 |
| uoais | 17.67 | 6.16 | 21.3 |
| unetpp_boundary_inst | 10.88 | 10.18 | 53.0 |
| unet_boundary_inst | 13.60 | 5.79 | 73.6 |
| unet_semantic_inst | 3.57 | 3.25 | 73.9 |
| ucn | 1.03 | 0.05 | 2.7 |
| msmformer | 0.00（不可靠） | 8.57 | 2.9 |
| **cellpose（100ep 修复版）** | 43.08 | **49.88** | — |
| **stardist（100ep 修复版）** | **56.70** | 45.82 | — |

## 口径二 · 32254 全量（GISEC 等预算协议，2026-08-30~09-06）

| 模型 | 参数 | segm AP | 备注 |
|---|---:|---:|---|
| **magformer_17m_mbv3l_dual_tower_dccg** | 17.45M | **0.7088** | 本项目参数效率版（AP50 0.8871） |
| yolov8s-seg（官方 ultralytics 8.4.14，COCO 预训练 FT） | 11.79M | 0.6989 | AP50 0.8737 / APs 0.047；ultralytics 自家口径 0.4119 |
| maskrcnn r18（torchvision） | 16.99M | 0.6638 | |
| mask2former r18（HF） | 16.54M | 0.4305 | query 范式欠拟合 |
| panoptic-deeplab r50（官方 d2） | 59.64M | 0.2518 | AP50 0.545 / AP75 0.198：找得到切不准 |
| uois-net zero-shot（官方 TOD 权重） | ~81M | 0.0003 | 渲染域零迁移失败对照 |
| cellpose 3.1.1.1（官方，files 磁盘通道） | 6.60M | 训练中 | RAM 红线战役详见 baselines 修复提交 |
| cellpose transformer（mit_b5+MAnet） | 92.2M | 排队 | 官方大模型版 |
| stardist 0.9.2 | 1.41M | 训练中 | |
| ucn（官方，from scratch） | ~14M | 训练中（6401） | 官方权重 2023 年死链 |

## 口径三 · 自研探索（非基线）

| 方法（新名） | 旧代号 | 口径 | segm AP |
|---|---|---|---:|
| m2f_47m_rgb_concat_finetune | d2m2f concat90/E23 | 32254 全量 @265K | **0.9069**（全场最高，47M） |
| depth_only | E24 | subset-1000 | 0.7894 |
| depth_trunk_distilled_rgb_sidepath | DPTD | subset-1000 | 0.7817 |
| rgb_trunk_distilled_depth_sidepath | hybrid_ptd | subset-1000 | 0.7539 |
| …full / …shallow / …no_distillation / naive_concat_control | W1/W4/W3/W2 | subset-1000 @40K | 0.8441 / 0.8344 / 0.8367 / 0.8296 |
| 蒸馏塔零训练直评 | zeroshot_cdti | subset-1000 | 0.8491 |
| v317_merge_m2f_rgb_backbone | c0_300k | subset 轨迹 / 全量补测 | 86.03@112.5K / 0.7565 |

GISEC（姊妹项目，16.85M，32254 全量）：canonical E26b offw0 **0.87617**——见 GISEC 仓 docs/BASELINE_ATLAS.md。
