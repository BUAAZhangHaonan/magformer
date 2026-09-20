# 调研A：小目标掩码边界/内容质量

日期：2026-09-21（调研员 A）
检索工具：arXiv API 全文摘要 + ar5iv 全文抽取（WebSearch/webReader 本周配额耗尽，已用 arXiv 直检替代；Semantic Scholar API 限流未用）。
主题：小目标掩码**内容/边界质量**的监督方法（损失、采样、标签、掩码头），面向 Mask2Former 系实例分割的移植。

## 1. 问题画像（我们的数字）

- Mask2Former 系单类工业零件实例分割：200 queries、1024² 输入、512² 掩码画布（stride-2）、双线性×2 + 0.5 二值化。
- 已检出 <64px 小目标的预测掩码换成"GT 穿生产渲染链"版本：AP_s 0.2925 → 0.5636（+27.1pt）——掩码内容质量是最大单项缺口；matched IoU 中位 0.635 vs 链可达 0.817。
- 症状：bbox 定位正确、掩码内部/边界糊；±1px 膨胀/侵蚀都更差（边界抖动，非整体偏移）；高 IoU 阈值段崩塌（AP@0.75=0.247，AP@0.9=0.018）。
- 主因：掩码损失在 12544 个均匀采样点上算 BCE+Dice，10px 目标边界只落 ~4-47 个点且不专门落在边界上——监督对边界无约束力；次因：512 画布上 <32px² 目标仅 2-3 cell（信息论不足）；hires 组 20× lr 噪声写在边界上。
- 约束：训练开销 +10-20% ms/it 以内；推理零/近零开销；与 DETR/M2F set-to-set 匹配兼容。

## 2. 方法总表

（收益数字均注出处；"定性"= 摘要/检索可见层面无具体数字或数字在正文未取到）

| # | 方法 | 年份/venue | 机制一句话 | 报告收益（场景+数字） | 训练开销 | 推理开销 | 移植到 M2F 的改动面 | 兼容性风险 |
|---|------|-----------|-----------|----------------------|---------|---------|--------------------|-----------|
| 1 | Boundary Loss（Kervadec 等，distance-map/level-set）[1] | MIDL'19 / MedIA'21 | 将 L2 轮廓距离写成基于 GT 距离图的区域积分，逐像素加权进 softmax 损失，梯度 ∝ 到边界距离 | 医学小病灶 ISLES：GDL 0.511→0.659 DSC（+14.8pt）；WMH 0.768→0.818，HD95 3.634→1.702mm [1] | 低（GT 距离图离线/collate 现算） | 零 | 小：逐点 BCE 乘 w(p)=DT(p)，完全兼容点采样损失 | 单独用会塌成空掩码，需配区域损失 + α ramp [1]；对 CE 基线无增益（WMH UNet-CE 0.757→0.756）[1] |
| 2 | Active Boundary Loss（ABL）[2] | AAAI'22 | 从当前预测检测边界，把"边界对齐"变成可微方向向量回归，迭代把预测边界推向 GT | Cityscapes DeepLabV3 79.5→80.5 mIoU；OCR-HRNetW48 82.2→82.9；ADE20K UperNet-SwinT +1.07/+1.77；BF(3px) 77.5→78.9；WMH GDL DSC 0.727→0.768 [2] | 低 | 零 | 小-中：在 512 掩码 logits 上检测边界 + 方向回归 | 去掉 detach 会失稳（76.0 vs 80.5，-4.5）[2]；语义分割出身，逐 mask 用需适配 |
| 3 | Conditional Boundary Loss（CBL）[3] | ECCV'22 | 从注意力图生成随语义/尺度条件化的"软边界带"，在带内加权分割损失 | 定性（ECCV'22 报告 Cityscapes/Pascal 上 mIoU 提升；正文数字未取到） | 低 | 零 | 小：条件化的带内加权，与点采样可结合 | 无 arXiv 版，复现需读 Springer 正文 |
| 4 | Boundary DoU Loss [4] | MICCAI'23 | 直接优化"差集/(差集∪部分交集)"的边界版 DoU，带宽随目标尺寸自适应 | 定性（UNet/TransUNet/Swin-UNet 在 ACDC/Synapse 上 Dice 超 Dice/Tversky 基线；数字在正文） | 低 | 零 | 小：替换/增补 per-mask Dice（M2F 本来就有 per-mask dice） | 点采样近似 vs 全分辨率精算需消融 |
| 5 | Boundary Loss for RS（可微边界度量代理）[5] | 2019（遥感） | 把边界检测 F 度量做成可微损失，惩罚 BCE/IoU/Dice 罚不到的边界错位 | 定性（遥感数据集 mIoU 提升） | 低 | 零 | 小 | 老方法，效果后被 ABL/CBL 覆盖 |
| 6 | Gated-SCNN（shape stream）[6] | NeurIPS'19 | 双流架构：门控 shape 流在图像级分辨率专职边界，高层激活过滤低层噪声 | 定性+（摘要：~2% mIoU、~4% boundary F，Cityscapes SOTA） | 中（额外分支） | 中（额外分支） | 大：需加分支，超出改动预算 | 语义分割架构手术，与 512 画布 stride-2 预算冲突 |
| 7 | PointRend（自适应点采样/细分渲染）[7] | CVPR'20 | 粗预测→双线性上采样→只对最不确定的 N 点用细特征重预测，迭代细分；训练用不确定性偏置随机采样 | COCO Mask R-CNN R50-1x mask AP 35.2→36.1-36.3；3x 37.2→38.2；X101 39.5→40.9；Cityscapes DLv3 77.8→78.4；224² 掩码省 30× 计算 [7] | 低 | 低-中（细分 MLP） | 中：M2F 已吸收其"点采样损失"思想；升级为推理期细分需加模块 | 推理期细分与"零开销"约束冲突 |
| 8 | Pointly-Supervised IS / Implicit PointRend [8] | CVPR'22 Oral | 每目标 ~10 个标注点 + 隐式表示，可达接近全掩码监督的效果 | ~10 点/目标 = 全掩码监督的 94-98% [8] | 低 | 零（监督侧） | 小（思想层面：点的"位置"比"数量"重要） | 证明点采样密度足够、位置不足是我们的真问题 |
| 9 | Mask2Former 点采样损失（基线锚点）[9] | CVPR'22 | 匹配损失用 K=12544 均匀点；最终损失用逐对的 importance sampling（不确定性采样） | COCO ins R50 43.7 / Swin-L 50.1 AP；点损失省 3× 训练显存（18GB→6GB）且不掉点 [9] | — | — | — | 我们的生产配置若用均匀采样，等于放弃 M2F 原文的重要性采样收益 |
| 10 | Mask Transfiner（quadtree 边界精修）[10] | ECCV'22 | 找"不相干区域"（下采样丢边界信息的稀疏点）组 quadtree，transformer 联合精修后粗到细传播 | COCO +3.0 mask AP（R50 37.5→39.4；query 系 41.6）；Boundary AP COCO 21.2→26.0；Cityscapes AP^B 11.4→18.0（+6.6）[10] | 中 | 中-高（7.1 vs 9.6 FPS，-35%） | 大：加 quadtree 精修头 | 推理 -35% FPS 超预算；但"不相干区域=边界"的思想可只用于训练 |
| 11 | RefineMask（多阶段粗到细融合）[11] | CVPR'21 | P2 语义分支 + 多阶段融合掩码头，后期只在边界带内训练/预测 | COCO 34.7→37.3（+2.6）；AP* 大目标 52.0→58.0（+6.0），小目标 22.6→24.1（+1.5）；11.4 vs 15.7 FPS [11] | 中 | 中（-27% FPS） | 大 | 收益集中在大目标；推理超预算 |
| 12 | DCT-Mask（频域掩码头）[12] | NeurIPS'21 | GT 128² 掩码 2D-DCT 取 300 低频系数，头回归 DCT 向量（L1），逆 DCT 重建高分辨率掩码 | COCO 35.2→36.5（+1.3），**AP@75 37.5→39.6（+2.1）**；LVIS +2.1；AP_S 17.2→17.7（+0.5）；22 vs 23 FPS（近零） [12] | 低-中 | 近零（-4% FPS） | 中：每个 query 加回归分支 + L1 | COCO 上小目标增益有限（+0.5），需与边界损失叠加 |
| 13 | PatchDCT（DCT 掩码 patch 精修）[13] | ICLR'23 | 对 DCT 掩码按 patch 分类+回归精修 | 超 DCT-Mask +0.7/+1.1/+1.3 AP（COCO/LVIS/Cityscapes）[13] | 中 | 低 | 中 | 同上，定位为第二阶段 |
| 14 | DynaMask（逐实例动态分辨率）[14] | CVPR'23 | r-FPN 逐级细化 + Mask Switch 模块按实例难度在 14²-112² 间选分辨率，避免固定高分辨率退化 | COCO 34.7→37.6（+2.9）；**AP75 +3.3-3.6；AP_S 18.3→20.7（+2.4）**；1.4G vs 8.0G FLOPs；8.3 vs 12.4 FPS [14] | 中 | 中（-33% FPS） | 大 | 掩码头类里小目标证据最强，但推理超预算 |
| 15 | BPR（边界 patch 后处理精修）[15] | CVPR'21 | 沿预测边界取小 patch，专用网络高分辨率精修，模型无关后处理 | 定性（Cityscapes 榜首组合 SegFix+PolyTransform+BPR）[15] | 无 | 高（逐实例 patch CNN 前传） | 小（后处理） | 违反零推理开销约束 |
| 16 | Mask Scoring R-CNN（MaskIoU 校准）[16] | CVPR'19 | 回归 mask IoU 作为质量分，校准分类置信度与掩码质量的错位 | 定性（COCO 一致小幅 AP 提升，≈+1pt 级；正文数字未取到）[16] | 低 | 低 | 小：加打分头 | 只改排序不改掩码本身，救不了 AP@0.9 的内容崩塌 |
| 17 | Mask DINO（去噪 + 统一检测分割）[17] | CVPR'23 | DINO 框架 + 掩码分支，query 去噪训练 + 联合数据集 | COCO Swin-L 54.5 AP vs Mask2Former 50.1（+4.4）[17][9] | 中 | 近零 | 大（换框架/训练策略） | 工程量大；可作为整体升级路径而非补丁 |
| 18 | HQ-SAM（高质量输出 token）[18] | NeurIPS'23 | 可学习 HQ output token + 融合 ViT 早期/最终特征，仅 44K 精细掩码微调 | 定性（10 个数据集零样本全面超 SAM；训练仅 8GPU×4h）[18] | 低（微调） | 零（同 SAM） | 小-中（若引入 SAM 作教师/辅助监督） | 引入 SAM 依赖；作为教师蒸馏最合适 |
| 19 | NWD（高斯 Wasserstein 距离）[19] + AI-TOD-v2/NWD-RKA [20] | ICCV'21 / TPAMI'23 | 小框建模为 2D 高斯，用归一化 Wasserstein 距离替代 IoU 度量/标签分配 | AI-TOD +6.0 AP；AI-TOD-v2 +4.3 AP（DetectoRS）[19][20] | 低 | 零 | 小（只动匹配代价/分配） | bbox 级度量，不直接产生掩码内容增益；可迁移到 M2F 匹配代价的小目标加权 |
| 20 | 植物 M2F + Focal+Boundary Loss [21] | 2025（PhenoBench 竞赛） | Mask2Former 全景框架上加 focal + distance-map boundary loss，分层监督植物/叶片/杂草 | **小目标杂草 PQ_weed 49.17→51.16（+2.0，纯 boundary loss）**；+focal 51.56；总 PQ 76.77→77.18；推理零开销 [21] | 低 | 零 | 小：直接在 M2F 损失上加项（最贴近我们场景的证据） | 农业遥感场景，单类增益≈2pt 量级 |
| 21 | O2Former（M2F + 查询/方向增强）[22] | 2025（SAR 舰船） | M2F 上改 query 生成（多尺度原型）+ 方向感知卷积/极坐标编码 | SSDD 近岸 AP 52.5→63.2（+10.7），**AP75 近岸 59.9→81.2（+21.3）**；HRSID 近岸 17.9→43.1；AP_S 近岸 51.4→59.5 [22] | 中 | 中 | 大 | 证明 M2F 系小目标+模糊边界有巨大可挖空间；但模块改动大 |
| 22 | Boundary IoU / Boundary AP 度量 [23] | CVPR'21 | 按目标尺寸自适应 trimap 带内算 IoU，衍生 Boundary AP/PQ | 定性（度量论文：对大目标边界误差远比 Mask IoU 敏感，且不过度惩罚小目标）[23] | — | — | 小（评测侧） | 用它做我们的验收指标而非训练信号 |
| 23 | BoundarySqueeze [24] | 2021 | 把分割视为边界挤压（warp），含可微边界注意力 | 定性（精度/速度优于 PointRend）[24] | 低 | 低 | 中 | 老方法，可选 |
| 24 | EffSeg（结构保持稀疏精修）[25] | 2023 | 只在结构保持的稀疏位置精修细粒度掩码 | 定性（vs RefineMask：FLOPs -71%、FPS +29%）[25] | 中 | 低-中 | 中 | 掩码头类，收益仍偏大目标 |
| 25 | SAM 辅助遥感损失 [26] / BCKD 边界蒸馏 [27] / MAL 自动标注 [28] | 2023-2024 | 用 SAM/ViT 教师产生对象/边界约束或伪标签，蒸馏边界区域 | 定性（三者均报告边界区域掩码质量/伪标签质量提升）[26][27][28] | 中（教师离线推理一次性） | 零 | 中：离线产软标签/边界图 | 教师域差（SAM 对工业零件未必准）；但我们有自己的"渲染链 oracle"可替代教师 |
| 26 | Focaler-IoU [29] | 2024 | 对 IoU 损失做 focal 式难样本聚焦 | 定性（bbox 回归，COCO/VOC 提升；数字在正文）[29] | 低 | 零 | 小 | bbox 系思想；可借"难阈值段聚焦"思想到 mask 损失权重 |

## 3. Top-5 可移植候选

判据：直击主因（12544 均匀点对边界无约束）> 次因（<32px² 目标 2-3 cell）；训练 ≤+10-20% ms/it；推理零开销；不动匹配机制（或只加权不动结构）。上限参照：oracle 缺口 27.1pt（AP_s 0.2925→0.5636）、matched IoU 中位 0.635→0.817、AP@0.75 0.247 / @0.9 0.018。

### 候选 1：边界带过采样 + 不确定性点采样（重写 M2F 点采样策略）
- **为什么攻主因**：我们的诊断就是"点不在边界上"。M2F 原文自己在最终损失里就用了 importance sampling（不确定性采样），并证明点损失不掉点还省 3× 显存 [9]；PointSup 证明 ~10 点/目标即可达全监督 94-98% [8]——**点的位置密度决定边界质量，点的总量不缺**。把采样分布改为 均匀 U + GT 边界距离带（|DT|<r，r 随目标尺度自适应）过采样 + 预测不确定性偏置，一个小目标边界的有效监督点可从 ~4-47 提升 5-10 倍，纯 loss 侧改动。
- **预期回收**：无同场景直接数字（定性：本项单位开销收益最高）。旁证：植物 M2F 仅加一个边界加权损失就拿到小目标 PQ +2.0、零推理开销 [21]。合理预期先吃掉 27.1pt 缺口中的"边界约束缺失"份额（结合候选 2，参照医学小结构 +14.8pt DSC 的量级 [1]）。
- **最便宜验证**：只改 `point_sample` 的索引分布与逐点权重；5k it 微调现有 ckpt（不动结构）；指标：AP_s、matched IoU 中位、AP@0.75/0.9、1/3px 边界 F。扫边界带采样比例 {0.25, 0.5, 0.75}×r=max(2, s/16)。开销：512² 距离变换每图 <5ms（collate 线程），GPU 采样不变，训练开销 ≈+2-5%。

### 候选 2：Kervadec 距离图边界损失（逐点加权并入 BCE）
- **为什么攻主因**：把"边界无约束"变成"梯度幅值 ∝ 到 GT 边界距离"——w(p)=signed DT(p) 直接乘进现有逐点 BCE，与 M2F 点采样损失**天然兼容、零结构改动**。最强的小结构证据全在医学：ISLES 小病灶 GDL 0.511→0.659（+14.8pt DSC）、HD95 减半 [1]；M2F 系直接证据：植物杂草（小目标）PQ +2.0、推理零开销 [21]。
- **预期回收**：+14.8pt 是 3D 小病灶 Dice；迁移到我们 AP_s 上保守估计可回收缺口的 1/4-1/3（~7-9pt），并显著抬 AP@0.75（0.247→）。
- **最便宜验证**：与候选 1 同一实验框架（可同扫）；α 从 0 线性升 1（前 2k it），必须配区域损失防塌缩 [1]；监控空掩码率与 hires 组边界噪声（20× lr 组是已知噪声源，边界加权可能放大——需单独看该组）。

### 候选 3：Boundary DoU（尺度自适应边界 DoU）替换/增补 per-mask Dice
- **为什么攻主因+次因**：M2F 的 per-mask Dice 分母小但仍无边界几何；Boundary DoU 优化"差集/(差集∪部分交集)"，**带宽随目标尺寸自适应**——<32px² 目标（我们次因：2-3 cell）的边界带相对面积更大、监督占比自动上调 [4]。区域集合运算可微、即插即用、论文明确卖点"无需额外损失、训练稳定"[4]。
- **预期回收**：ACDC/Synapse 上 Dice 超 Dice/Tversky 基线（定性，正文数字未取到）[4]；与候选 1/2 正交可叠加。
- **最便宜验证**：先在 512 全分辨率 GT 掩码上精算（训练时 GT 本来就有），每 mask 一次集合运算 ≈+3-5% ms/it；消融 点级近似 vs 全分辨率。

### 候选 4：渲染链软标签监督（自蒸馏 / oracle 软边界做教师）
- **为什么最有针对性**：27.1pt 缺口就是"GT 穿渲染链"版本定义的——把 BCE 的 0/1 目标换成**渲染链产生的软边界标签**（离线批量生成，已有全部管线），教网络输出"过链后仍高 IoU"的掩码形态；±1px 膨胀/侵蚀变差说明模型缺的正是"软边界的正确形状"。旁证：MAL 用 ViT 自动标注达到≈全监督质量 [28]；BCKD 边界区域蒸馏提升学生边界 [27]；HQ-SAM 仅 44K 高质掩码 + 8GPU×4h 就大幅提升边界 [18]——高质标签的边际收益极大。
- **预期回收**：直接对齐 oracle 上限 0.5636；首要指标是 AP@0.75/0.9 崩塌段（0.247/0.018）与 matched IoU 中位（0.635→趋近 0.817）。
- **最便宜验证**：第一阶段"oracle 软标签替换硬标签"（纯数据侧，训练开销≈0）；第二阶段 teacher=EMA 自模型做 KL 蒸馏。风险：软标签与评测用硬 GT 的偏差——保留小权重硬 BCE 混合。

### 候选 5：DCT-Mask 频域掩码头（辅助回归分支）
- **为什么**：唯一同时攻"内容质量整体"与"高 IoU 崩塌"的头侧方法：AP@75 +2.1（37.5→39.6）且推理近零开销（22 vs 23 FPS）[12]；PatchDCT 再 +0.7-1.3 [13]。300-dim DCT 强迫掩码输出结构化高频，正对"内部糊"。
- **预期回收**：COCO AP_S 仅 +0.5 [12]——单用不够；作为候选 1-4 收敛后的第二阶段，目标 AP@0.9>0。
- **最便宜验证**：每个 query 加 3FC 回归 + L1（128² 目标画布可降到 64² 省算）；训练 ≈+10%；推理 iDCT 每 mask 微秒级。

Runner-up（超推理预算或改动过大，思想可借）：Mask Transfiner（quadtree"不相干区域=边界"判定可只用于训练期采样 [10]）、DynaMask（掩码头类中 AP_S +2.4 最强 [14]）、O2Former（M2F 系小目标 AP75 +21.3 的存在性证明 [22]）、NWD 度量进匹配代价（小目标标签分配 [19][20]）、Boundary AP 做验收指标 [23]。

## 4. 反面教训

1. **精修头收益集中在大目标**：RefineMask AP* 大目标 +6.0 vs 小目标 +1.5 [11]；DCT-Mask AP_S 仅 +0.5 [12]；DynaMask AP_S +2.4 但 AP_L +3.1 [14]——直接搬 COCO 精修头救不了 <64px 桶，loss/标签侧必须先行。
2. **朴素加分辨率反而退化**：Mask R-CNN 固定掩码 28²→112² AP 34.7→32.5（8G FLOPs）[14]——512 画布上对小目标无脑放大不是出路；DynaMask 需逐实例选择性才有效。
3. **Boundary loss 单独用会塌**：Kervadec 明确报告单独边界损失塌成空预测，需 ℒR+αℒB 且 α 走 ramp [1]；对纯 CE 基线可能无增益（WMH UNet-CE 0.757→0.756）[1]——要与我们 BCE+Dice 联合消融。
4. **ABL 去掉 detach 失稳 -4.5 mIoU** [2]：边界检测算子里的梯度截断是稳定性关键，移植时保留。
5. **固定 trimap 宽度的边界度量/损失会过度惩罚小目标**：Boundary IoU 论文自己的分析（固定带宽对小目标退化为普通 IoU 甚至失真），必须尺度自适应带宽 [23][4]。
6. **后处理精修违反零推理约束**：BPR 逐实例 patch 前传 [15]、SegFix 额外分支——推理开销不可忽略。
7. **分数校准不等于掩码变好**：Mask Scoring R-CNN 只修排序 [16]，AP@0.9=0.018 的内容崩塌救不了。
8. **bbox 级小目标度量不能直接当掩码监督**：NWD/DotD 证据强（+6.0 AP）但作用在框高斯上 [19][20]，掩码内容需另配；只能借到匹配代价加权。
9. **Focal 类损失对极小类/极小目标可能崩**：2026 年 VLM 细粒度损伤分割报告 focal loss 把 tiny-damage 检测塌到 0 [30]——对我们单类场景，盲目加 focal 项需谨慎（植物场景 focal+boundary 组合才有效 [21]）。
10. **SAM 当推理期精修器不可行**（推理成本+提示工程），SAM 系只以教师/标签角色入场（HQ-SAM、SAM 辅助损失 [18][26]）；且 SAM 对工业零件域差未知，我们自有 oracle 更可靠。

## 5. 引用列表

[1] Kervadec et al., *Boundary loss for highly unbalanced segmentation*, MIDL 2019 / MedIA 2021. https://arxiv.org/abs/1812.07032 （数字取自 ar5iv 全文）
[2] *Active Boundary Loss for Semantic Segmentation*, AAAI 2022. https://arxiv.org/abs/2102.02696 （数字取自 ar5iv 全文）
[3] *Conditional Boundary Loss for Semantic Segmentation*, ECCV 2022（Springer；无 arXiv 版）. 检索入口：https://scholar.google.com/scholar?q=%22Conditional+Boundary+Loss%22+segmentation
[4] Sun, Luo, Li, *Boundary Difference Over Union Loss for Medical Image Segmentation*, MICCAI 2023. https://arxiv.org/abs/2308.00220 代码 https://github.com/sunfan-bvb/BoundaryDoULoss
[5] *Boundary Loss for Remote Sensing Imagery Semantic Segmentation*, 2019. https://arxiv.org/abs/1905.07852
[6] Takikawa et al., *Gated-SCNN*, NeurIPS 2019. https://arxiv.org/abs/1907.05740
[7] Kirillov et al., *PointRend: Image Segmentation as Rendering*, CVPR 2020. https://arxiv.org/abs/1912.08193 （数字取自 ar5iv 全文）
[8] Li et al., *Pointly-Supervised Instance Segmentation*, CVPR 2022. https://arxiv.org/abs/2104.06404
[9] Cheng et al., *Mask2Former*, CVPR 2022. https://arxiv.org/abs/2112.01527 （点采样/重要性采样细节取自 ar5iv 全文）
[10] Wang et al., *Mask Transfiner*, ECCV 2022. https://arxiv.org/abs/2111.13673 （数字取自 ar5iv 全文）
[11] Shen et al., *RefineMask*, CVPR 2021. https://arxiv.org/abs/2104.08569 （数字取自 ar5iv 全文）
[12] Shen et al., *DCT-Mask*, NeurIPS 2021. https://arxiv.org/abs/2011.09876 （数字取自 ar5iv 全文）
[13] *PatchDCT: Patch Refinement for High Quality Instance Segmentation*, ICLR 2023. https://arxiv.org/abs/2302.02693
[14] *DynaMask: Dynamic Mask Selection for Instance Segmentation*, CVPR 2023. https://arxiv.org/abs/2303.07868 （数字取自 ar5iv 全文）
[15] *Look Closer to Segment Better: Boundary Patch Refinement*, CVPR 2021. https://arxiv.org/abs/2104.05239
[16] Huang et al., *Mask Scoring R-CNN*, CVPR 2019. https://arxiv.org/abs/1903.00241
[17] *Mask DINO*, CVPR 2023. https://arxiv.org/abs/2206.02777
[18] Ke et al., *Segment Anything in High Quality (HQ-SAM)*, NeurIPS 2023. https://arxiv.org/abs/2306.01567
[19] Wang et al., *A Normalized Gaussian Wasserstein Distance for Tiny Object Detection*, 2021. https://arxiv.org/abs/2110.13389
[20] *Detecting tiny objects in aerial images: NWD-RKA + AI-TOD-v2*, 2022. https://arxiv.org/abs/2206.13996
[21] *Exploiting Boundary Loss for Hierarchical Panoptic Segmentation of Plants and Leaves*, 2025. https://arxiv.org/abs/2501.00527 （数字取自 ar5iv 全文）
[22] *O2Former: SAR Ship Instance Segmentation*, 2025. https://arxiv.org/abs/2506.11913 （数字取自 ar5iv 全文）
[23] Cheng et al., *Boundary IoU*, CVPR 2021. https://arxiv.org/abs/2103.16562
[24] *BoundarySqueeze: Image Segmentation as Boundary Squeezing*, 2021. https://arxiv.org/abs/2105.11668
[25] *EffSeg: Efficient Fine-Grained Instance Segmentation*, 2023. https://arxiv.org/abs/2307.01545
[26] *SAM-Assisted Remote Sensing Semantic Segmentation with Object and Boundary Constraints*, 2023. https://arxiv.org/abs/2312.02464
[27] *BCKD: Towards Customized Knowledge Distillation*（边界/上下文蒸馏）, 2024. https://arxiv.org/abs/2401.13174
[28] *Vision Transformers Are Good Mask Auto-Labelers (MAL)*, 2023. https://arxiv.org/abs/2301.03992
[29] *Focaler-IoU*, 2024. https://arxiv.org/abs/2401.10525
[30] *Grounding Agentic VLMs with Dedicated Segmentation for Fine-Grained Vehicle Damage Assessment*, 2026. https://arxiv.org/abs/2608.02470
（另：BLO-Inst YOLO+SAM 双层优化 https://arxiv.org/abs/2601.22061 ；SAM 核实例边界引导精修 https://arxiv.org/abs/2603.28027 ——均为定性，作教师/精修思想参考。）
