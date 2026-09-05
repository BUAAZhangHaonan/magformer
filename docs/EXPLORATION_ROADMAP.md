# MAGFormer 探索路线图（全谱系，2026-03 至 2026-09）

> 命名约定（2026-09-06 起）：方法一律用描述性名称，历史字母代号（CDTI/DPTD/W1-W4/E18-E24/B16M/c0）仅作为别名保留在映射表中。
> 所有权重与实验产物统一归档于 4029 `~/magformer/archive_20260906/`（staging_4029 / staging_6401 / backup）。git 只入代码与文档。

## 总览

| 时代 | 时间 | 数据集 | 主线 | 最终结果 |
|---|---|---|---|---|
| 一、奠基收官 | 2026-03~04 | 20260318_1K_1566（train 1,261 / val 149） | 双塔架构定型 + 轻深度塔扫描 + 19 基线收官 | **magformer_depthnorm：segm AP 68.42 / mAP50 87.97 @1024**（全项目最高 mAP50） |
| 二、vc-suda-sim2real | 2026-05 | 0831_1K + pseudo_real_512 | r100→r142（41 个配置版本）少量真实标注迁移 | **失败**：无合适真实数据（分支保留，不合 master） |
| 三、v317 大模型主线 | 2026-07 | 32254 | c0_corrected_300k（4028 训练，141K 步被杀） | subset 轨迹 85.16→85.71→86.03@112.5K；全量补测见下 |
| 四、M2F 融合探索 | 2026-08 | 32254 subset-1000 | E18-E24（塔蒸馏→warmstart→native recipe→融合十臂） | m2f_47m_rgb_concat_finetune **0.9069** 全量（自研探索，非基线） |
| 五、交叉蒸馏审计 | 2026-08-28~09-02 | 32254 subset-1000 @40K | 四臂审计 | **0.8441**（蒸馏注入胜出，蒸馏贡献 +0.74） |
| 六、参数效率最终版 | 2026-08-24~30 | 32254 全量（GISEC 等预算 64K/bs8） | 17.45M 双 MBV3-L 塔 + DCCG | **segm AP 0.7088**（AP50 0.8871） |

## 时代一 · 奠基收官（1566，正式发表口径）

正式报告：`docs/2026-04-12-final-multi-resolution-results.md`（commit a9592766）。
`magformer_depthnorm_on` = Swin-RGB + depth-norm + 门控融合，79.7M。
轻深度塔扫描（lightdepth_roster.json）：convnextlite 65.45 > **mobilenetv3 64.74/64.50** > nodpth 59.52；参数 79.7M→49.9M（-37%）代价 ~3.7pt。
深度贡献 = +8.9pt（68.42 vs 59.52）。

## 时代二 · vc-suda-sim2real（r100-r142，判失败）

`feature/vc-suda-sim2real` 分支完整保留 41 个配置版本与 E18/E19 塔蒸馏提交（已推 origin）。
终版基线 r141 RGB-D / r139 RGB-only（target150）。产物在 archive（各留 best）。

## 时代三 · v317 大模型主线（c0_corrected_300k）

- 定名：`v317_merge_m2f_rgb_backbone`（RGB 主干取自 M2F task ckpt，解码器 v316e 血统）
- 4028 训练，best = iter 112,500（141K 处机器任务被杀，"300k" 为计划步数）
- subset-1000 轨迹：85.16@37.5K → 85.71@75K → **86.03@112.5K**
- **全量 3276 补测（2026-09-06，E18 源码树 + 严格装载校验 882/882 对齐，dpe/agpe 开启）：segm AP 0.7565 / mAP50 0.9292 / AP75 0.8450**
- 权重：archive/pretrained（c0_corrected_300k_seed42_best.pth，630M）

## 时代四 · M2F 融合探索（E18-E24 → 描述名）

| 旧代号 | 新名 | 结果 (subset-1000) |
|---|---|---|
| E18/E19 | distill_towers_from_pretrained（rdi_lite / rdi_mbv3l 两塔） | 塔蒸馏 |
| E20 | warmstart_from_distilled_towers | rdi_lite 0.8444 / mbv3l 0.8427 |
| E21 | refine_query_masks_with_rgbd_points | 0.8284 |
| E22 a-d | native_recipe_coco_init（lr/nobox 扫描） | — |
| E23 | m2f_47m_rgb_concat_finetune | 0.8988；**全量 0.9069**（全场最高，自研探索） |
| E24 十臂 | fusion_arms：depth_trunk_distilled_rgb_sidepath（DPTD）/ rgb_trunk_distilled_depth_sidepath（hybrid_ptd）/ depth_only / concat4ch / mgc2 / sgf … | depth_only 0.7894 > dptd 0.7817 > hybrid 0.7539 |

机理结论：**深度是主导模态**——depth_only 胜过一切融合（Round-5 pivot）；DPTD 把获胜配方镜像（深度主干+蒸馏 RGB 侧路）。

## 时代五 · 交叉蒸馏审计（四臂）

定名：`coco4ch_base_distilled_depth_tower_film_injection`
- W1 full（四阶段注入）**0.8441** > W3 no_distillation 0.8367 > W4 shallow 0.8344 > W2 naive_concat_control 0.8296
- 蒸馏塔零训练直评（zeroshot）0.8491
- 结论：蒸馏注入本身有效（+1.45 vs 朴素 concat），其中塔的蒸馏占 +0.74

## 时代六 · 参数效率最终版

定名：`magformer_17m_mbv3l_dual_tower_dccg`
- MBV3-Large 双塔（RGB 2.97M + Depth 2.97M）+ DCCG 融合（v283 血统，5.65M）+ 解码器 = **17,449,177 参数**
- 32254 全量、GISEC 等预算（64K iter × bs8）：**segm AP 0.7088 / AP50 0.8871 / AP75 0.7932 / bbox 0.6645**
- 事故史（STATUS.md）：focal-bug 修复 + lr 隐式乘数发散修复（mgm_multiplier 默认 2.0 过热 DCCG）

## 版本号体系说明

r100-r142 = 时代二配置号；融合模块内部版本至 v283；v316e/v317 = 时代三大模型合并版本。内部迭代总数远超用户印象的 v160。

## 命名映射总表

| 旧代号 | 正式名 |
|---|---|
| CDTI / W1 / W4 | coco4ch_base_distilled_depth_tower_film_injection (_full / _shallow) |
| W3 | …_no_distillation |
| W2 | naive_rgb_depth_concat_control |
| DPTD | depth_trunk_distilled_rgb_sidepath |
| hybrid_ptd | rgb_trunk_distilled_depth_sidepath |
| E23 / concat90 | m2f_47m_rgb_concat_finetune |
| B16M | magformer_17m_mbv3l_dual_tower_dccg |
| c0_corrected_300k | v317_merge_m2f_rgb_backbone |
| E18-E24 | 见时代四表 |
| r1xx | vc-suda 历史配置号（分支内保留原名） |
