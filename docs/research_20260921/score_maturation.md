# 调研C：分数成型与 copy-paste 增密

调研员 C ｜ 2026-09-20 ｜ 主题：DETR 系（Mask2Former 单类实例分割）训练期分数成型 / 校准 + copy-paste 增密后的训练行为
检索方式：WebSearch 配额耗尽（9-25 重置），改用 arXiv API + arXiv abs 页逐篇核对数字；所有"收益"列数字除注明 ~ 者外均出自论文摘要/官网。

---

## 1. 问题画像（我们的数字）

- 打分公式：`score = cls_logit × maskness`（二值掩码内平均概率）。小目标两个因子结构性双低：<64px 桶 TP 分数中位 **0.29** vs 大目标 **0.80**。
- 已证伪的事后路线：分位重标定 **-13pt**、score=IoU 替换 **-11.9pt**。原因：小目标检测池 **80.5% 是 hard-FP**，任何抬分变换都让 FP 搭车。结论：分数必须在**训练期**解决。
- 稀疏正样本：batch=4 每步仅 **0.85 个可见 <64px GT**（小目标占实例 4.94%、像素 0.124%、损失点份额 0.1-0.4%）→ focal 平衡被中大目标主导。
- copy-paste 通道已修复（深度同步贴入），文献锚 Ghiasi CVPR21（+1.2 AP COCO / +2.4-3.7 LVIS / 小目标 +3-7pt），拟以 bank 2000+ / max_paste 12-20 / prefill 启用。
- 深度通道弱对比度：小目标内外深度差 p50≈1.5σ 噪声（贴入质量上限约束）。
- 约束：训练开销 +10-20% 内；推理零开销优先；EMA 0.9999 在用。

文献视角下，这恰好对应两个有成熟先例的子问题：
1. **分类-定位/掩码质量错位**（cls 分数不携带可达质量信息）——IoU-aware / ranking 系一线；
2. **稀疏监督增密**（每步 0.85 个正样本喂不饱 query 系匹配）——one-to-many 辅助 + DEIM 的"增密必须配质量感知损失"结论。

---

## 2. 方法总表

收益列数字除标 ~ 者均为摘要可查；T=训练期改动，I=推理期改动。

| # | 方法 | 年份/venue | 机制一句话 | 收益(数字) | 训练开销 | 推理开销 | 移植改动面 | 兼容性风险 |
|---|------|-----------|-----------|-----------|---------|---------|-----------|-----------|
| 1 | VarifocalNet (IACS+VFL) | CVPR21 | 分类目标改为"IoU-aware 分类分数"，正样本按质量加权、负样本 focal 式压梯度的非对称损失 | COCO +2.0 AP over FCOS+ATSS 基线（最佳 55.1） | ≈0（只改损失） | 0（去掉了 IoU 分支反而更省） | 小：换 cls 损失一行级 | 低；需给 matched query 提供质量标量 |
| 2 | GFL (QFL) | NeurIPS20 | 分类分支回归"cls×IoU 联合软标签"，BCE 推广为 quality-target 形式 | 单类提升 ~+1 AP 级 | ≈0 | 0 | 小 | 低 |
| 3 | GFLv2 (DGQP) | arXiv20 | 用框分布统计（峰值锐度）生成质量估计，替代脆弱的质量预测头 | R101 46.2 vs ATSS 43.6（+2.6） | 轻量预测头 | 近 0 | 中：需要分布式回归头，Mask2Former 无对应物 | 中；掩码版需自设计 |
| 4 | TOOD (TAL+T-Head) | ICCV21 oral | 任务对齐：按 cls×IoU 联合度量选正样本 + 对齐头交互两任务 | 51.1 vs ATSS 47.7 / GFL 48.2 / PAA 49.0 | 头部小改 | 近 0 | 中：动分配+头 | 中；与我们已有 matcher 交互 |
| 5 | OTA | CVPR21 | 标签分配=最优运输，全局考虑锚-GT 匹配而非逐 GT 贪心 | FCOS-R50 40.7 (1x)，超各分配法 | Sinkhorn 迭代，小 | 0 | 中：换分配器 | 中；小目标匹配改动已另线验证过 |
| 6 | RS Loss | ICCV21 oral | 排序损失：正样本排所有负样本之上、正样本之间按 IoU 排序；identity update 可微 | Faster R-CNN +3 box AP；LVIS+RFS +3.5 mask AP（稀有类 ~+7）；免辅助头/免采样启发式 | ≈0~小 | 0（还省掉 centerness/IoU 头） | 中：换分类损失+去辅助项 | 低；对类不平衡稳健（论文卖点） |
| 7 | Rank-DETR | NeurIPS23 | rank-oriented 架构+排序损失：促正抑负，损失/匹配代价偏向定位更准的预测 | 在 R50/Swin-T/Swin-L 上改进 H-DETR 与 DINO（高 IoU 阈值 AP 提升更明显） | 小 | 0 | 中：加排序损失与 query 选择 | 低-中 |
| 8 | Align-DETR | BMVC24 | 对齐损失联合纠正分类-回归与跨层错位，中间层用 many-to-one | R50 49.3（+0.6 over H-DETR），2x 51.7 | 小 | 0 | 中 | 中 |
| 9 | H-DETR | CVPR23 | 训练时加一条 one-to-many 辅助支路，推理时丢弃 | Deformable/PETR 系一致提升（R50 档 ~+2 AP 量级） | +一条支路（~10-20% 步时） | 0 | 中：加辅助 head+分支 | 中；分数分布会被 o2m 污染需处理 |
| 10 | Group DETR | ICCV23 | 多组 query 各自组内 one-to-one，等价安全增密 | 各 DETR 变体加速收敛+提升 | +query 组 | 0 | 中 | 中 |
| 11 | Co-DETR | ICCV23 | 并行多个 one-to-many 辅助头（ATSS/RCNN 等）协同训练 | ViT-L 66.0 AP COCO test-dev | 大（多辅助头） | 0 | 大：超我们 +10-20% 预算 | 高 |
| 12 | MS-DETR | arXiv24 | 混合监督：o2m 直接加在主 decoder query 上，无需额外分支，并用解耦损失处理重复 | 超 DN-DETR/Hybrid/Group DETR | 小-中 | 0 | 中 | 中 |
| 13 | NMS Strikes Back | arXiv22 | 证明 o2m+NMS 一致优于 o2o；给 DETR 加 NMS | +2.5 mAP（Deformable-R50 12ep 50.2） | 0 | +NMS（破坏零推理开销偏好） | 小 | 高（推理策略改变） |
| 14 | DEIM (Dense O2O + MAL) | CVPR25 | 用数据增广成倍增加每图目标数（增密正样本）+ 质量感知损失 MAL 抑制低质量匹配 | RT-DETRv2 53.2 AP/单卡4090一天；训练时间 -50%；D-FINE-X 56.5 | 增广开销，损失零开销 | 0 | 小-中：MAL 是一行损失级改动 | 低；**与我们 copy-paste 天然同构** |
| 15 | Seesaw Loss | CVPR21 | 梯度再平衡：按类量比削减尾类负梯度（缓解）+ 尾类误分时加罚（补偿） | LVIS SOTA（显著超 CE） | ≈0 | 0 | 小：换损失 | 低；但为"类尾"设计，搬到"尺度尾"需重构 |
| 16 | EFL | CVPR22 | 每类独立 focal 调制因子，按训练状态动态调整——证明全局 γ 无法同时服务头尾 | LVIS v1 29.2 AP（一阶段 SOTA） | ≈0 | 0 | 小 | 低；同上需按尺度桶化 |
| 17 | Mask Scoring R-CNN | CVPR19 | MaskIoU 头回归预测掩码与 GT 的 IoU，校准 cls 与掩码质量错位 | 稳定超 Mask R-CNN（~+1 mask AP 量级） | +一个小头 | +一次前传乘法 | 中 | 中；本质是"学出来的乘法重排"，与已证伪路线同族 |
| 18 | Calibrated Teacher | CVPR23 | 把 EMA teacher 置信度校准到 precision，使固定阈值全程可用 + Focal IoU loss | 稀疏标注 COCO SOTA | ≈0 | 0 | 小（我们已有 EMA0.9999） | 低 |
| 19 | Consistent-Teacher | CVPR23 | ASA 动态分配 + GMM 动态伪标签阈值 + 特征对齐，稳住分数监督 | 10% COCO 40.0 mAP（+~3 over 伪标签系） | 小 | 0 | 中 | 中 |
| 20 | NorCal | NeurIPS21 | 事后按类样本量重加权分数（类别长尾专用） | LVIS 稀有/常见/频繁类均升 | 0 | 近 0 | — | **不适用**：依赖类先验可分；尺度尾同类 hard-FP 不可分 |
| 21 | 检测校准系列 (Küppers 等) | CVPRW20/22/WACV24 | 多变量（位置×尺度条件化）置信度校准；结论：检测/分割器系统性失准，且失准依赖尺度 | ECE 显著改善 | 0（事后） | 近 0 | — | 改 ECE 不改 AP；同我们已证伪路线 |
| 22 | TCD | NeurIPS22 | 训练期校准损失（域适应场景） | 域移位下校准+检测双升 | 小 | 0 | 中 | 中；为域移位设计 |
| 23 | FRACAL | CVPR25 | 免训练 logit 调整（分形维数平衡稀有/常见类） | 稀有类 +8.6% | 0 | 近 0 | — | 类别长尾事后法，同 NorCal 风险 |
| 24 | Copy-Paste (Ghiasi) | CVPR21 | 随机贴入实例掩码，无需上下文建模 | COCO 49.1 mask AP；LVIS 稀有 +3.6；小目标 +3-7pt（团队锚） | +数据管线 | 0 | 我们已建 | — |
| 25 | Kisantal 增密 | arXiv19 | 过采样含小目标图 + 小目标反复多次贴入（不限次数） | COCO 小目标 IS 相对 +9.7%、检测 +7.1% | 小 | 0 | 已具备 | 低；支持"高 paste 数" |
| 26 | InstaBoost | ICCV19 | 概率图引导贴入位置（局部外观相似处） | R101 Mask R-CNN 35.7→37.9 | 小 | 0 | 小（贴入位置策略） | 低 |
| 27 | X-Paste | arXiv22 | 用 CLIP+SD 无限量合成实例源 + 零样本质量过滤 | LVIS 长尾 +6.8 box / +6.5 mask | 中（生成离线） | 0 | 中 | 中；**质量过滤是承重组件** |
| 28 | Occlusion/Location-aware C&P | arXiv22/23 | 面向遮挡/特定域的场景化贴入（同类遮挡、位置感知） | OCHuman SOTA / MMSports23 冠军方案组件 | 小 | 0 | 小 | 低 |
| 29 | SDI-Paste | arXiv24 | 视频 IS：合成动态形变实例贴入 | YT-VIS21 +2.9 AP (+6.5%) | 中 | 0 | 中 | 中 |
| 30 | AD-Det | arXiv25 | UAV：自适应小目标增强 + 动态类平衡 copy-paste（缺什么贴什么） | VisDrone 37.5 AP，超对手 ≥3.1 | 小 | 0 | 小（贴入策略） | 低；**按需/按桶贴入的范式锚** |
| 31 | SOC | arXiv25 | 3D 布局+相机增强+生成式和谐化的合成组合 | 超 Copy-Paste/X-Paste 管线 +24-36%，LVIS +10.9 | 大（离线合成） | 0 | 大 | 指出朴素贴入的和谐化缺口（我们深度同步正是在补这个） |
| 32 | InstaDA | arXiv25 | LLM+扩散双智能体数据增广 | LVIS +4.0 box / +3.3 mask | 大 | 0 | 大 | 超预算 |
| 33 | Poisson C&P | arXiv23 | 泊松图像编辑无缝贴入 | 超声分割训练更稳 | 小 | 0 | 小 | 低；深度弱对比下的备选平滑 |
| 34 | Stitcher (DST) | arXiv20 | 用优化反馈收集小目标欠拟合样本，拼图回灌 | COCO +2 AP | 小 | 0 | 中 | 低 |
| 35 | SAHI | ICIP22 | 切片超推理+切片微调 | Visdrone 推理 +5.1~6.8 AP；加切片微调累计 +12.7~14.5 | 微调需重训 | **推理 ×N 切片** | 中 | 推理开销违反零开销约束；切片边缘标签截断 |
| 36 | NWD (+NWD-RKA) | arXiv21/ISPRS22 | 框=2D 高斯，Wasserstein 距离替代 IoU 做分配/NMS/损失 | AI-TOD +6.7 AP（vs 标准微调）；NWD-RKA +4.3（AI-TOD-v2） | ≈0 | 0 | 小-中 | 低；只动分配度量，不动分数 |
| 37 | CP-SSOD (Simple Baseline) | arXiv23 | 仅靠 copy-paste（EMA teacher + 强弱增广）做到 SSOD SOTA，无需复杂一致性设计 | COCO 部分标注下超复杂 SSOD 管线（小目标 AP 显著） | 小 | 0 | 小（我们已有 EMA） | 低；**copy-paste 增密与 EMA 组合的直接先例** |
| 38 | SNIP/SNIPER | CVPR/ECCV18 | 尺度归一化：只在合适尺度采样/推理 | 经典基线 | 中 | 0/中 | 中 | 需多尺度训练；与我们的分辨率方案部分重叠 |

> 未单列但相关：FeatAug-DETR（arXiv23，特征级 one-to-many 增密，Swin-L 24ep 58.3 AP）、MDS-DETR（arXiv26，单 decoder 混合 o2o/o2m +2.8 mAP）、LoRA-DETR（arXiv26，结论"分配多样性>数量"）、Mask Transfiner（掩码边界质量）。Ghiasi 系综述性证据链：贴入数量可持续加码（Kisantal/X-Paste）、位置与和谐化决定上限（InstaBoost/SOC）、按需贴入优于均匀贴入（AD-Det）。

---

## 3. Top-5 可移植候选

两件事分别对应：**(甲) 训练期让小目标分数成型**（候选 1/2/3/5）、**(乙) copy-paste 启用参数最优取值**（候选 4）。所有候选推理开销为零或近零，训练开销均在 +10% 内。

### 候选 1：DEIM-MAL 质量感知分类损失（首选，最便宜）
- **是什么**：DEIM (CVPR25) 的 Matchability-Aware Loss：matched query 的分类目标从 one-hot 1 改为质量标量 q（DETR 版=匹配 IoU），损失按 q 凸加权——高质量匹配学高分、低质量匹配不强推。DEIM 的另一半（Dense O2O 用增广成倍加目标）与我们的 copy-paste 通道**天然同构**：我们贴入 12-20 个小目标就等于 DEIM 的目标增密，而 MAL 正是 DEIM 论文里"增密后必须配质量感知损失，否则低质量匹配拖垮性能"的配套件。
- **移植**：Mask2Former 的分类是 masked-pooling 后的 BCE。把 matched query 的 target 由 1 改为 `q = maskness_t`（我们已实时计算 maskness，直接复用，建议 detach/EMA 平滑避免早期噪声），负样本项保持 0.25 权重；凸指数 α∈[1.5, 2] 起步。改动 ≈ 损失函数 10 行。
- **收益锚**：DEIM：RT-DETRv2 53.2 AP 且训练时间 -50%；其消融明确"无 MAL 的 dense 匹配退化"。
- **最便宜验证**：当前数据 10-15k iter、seed 固定、单因子开关。看四个数：`<64px 桶 TP 分数中位（0.29→?）`、`recall@0.3（小桶）`、`AP_small`、`固定池容量下的 FP 率`。预期信号：TP 中位上移而 FP 率不升（区别于事后抬分的搭车效应）。

### 候选 2：尺度分桶的 Equalized Focal（VFL 目标 + EFL 调制）
- **是什么**：把 EFL (CVPR22) 的"每类独立 focal 调制因子"从类尾搬到尺度尾：`<64px` 桶用独立 γ/调制因子，按训练状态（损失点份额 0.1-0.4% 实时统计）自适应。EFL 的核心结论正是"全局 γ 无法同时服务头尾分布"，直接对应我们"focal 平衡被中大目标主导"。
- **移植**：在候选 1 的质量目标之上（正样本 target=maskness），负样本项按 query 匹配目标的尺度桶给不同 γ；单类场景下"类别频率"替换为"尺度桶频率"。改动 ≈ 损失内 20 行。
- **收益锚**：EFL：LVIS 29.2 AP 一阶段 SOTA；Seesaw 的"补偿因子"（对尾类 FP 加罚）对我们 80.5% hard-FP 池是对症组件。
- **最便宜验证**：与候选 1 同一脚本，2×2 网格（质量目标 on/off × 尺度桶 γ on/off），每格 10k iter；主看小桶 TP 分数分布（P25/P50/P75）与 hard-FP 分数分布是否拉开间距。

### 候选 3：RS Loss / Rank-DETR 式排序目标（直接对齐 AP 语义）
- **是什么**：RS Loss (ICCV21 oral)：让分类器把每个正样本排到所有负样本之上、正样本之间按（掩码）IoU 排序——损失直接逼近 AP 的排序语义，且论文证明其对不平衡稳健、免采样启发式。Rank-DETR (NeurIPS23) 是 DETR 专用版：促正抑负 + 高 IoU 阈值 AP 增强。
- **移植**：一步到位移植 RS Loss 较重；**先做轻量探针**——每图内 matched queries vs 未匹配 queries 的成对 margin 损失（hinge，margin 按尺度桶缩放），~30 行。
- **收益锚**：RS Loss：Faster R-CNN +3 box AP；LVIS+RFS +3.5 mask AP（稀有 +7），并**免掉辅助质量头**。
- **最便宜验证**：轻量 margin 探针 10k iter，同候选 1 的四指标。若小桶 TP 中位抬升但 hard-FP 同抬，说明排序边界仍在，转完整 RS Loss（其"正样本之间按质量排序"正是分离 hard-FP 的机制）。

### 候选 4：copy-paste 参数网格（尺度分层贴入 + 质量过滤入库）
- **文献参数锚**：
  - **数量**：Kisantal（反复多次贴入，相对 +9.7%）、Ghiasi（贴入 donor 全部实例，通常每图 5-30）→ max_paste 12-20 在文献支持区间内、偏保守侧，**可以再加码**；
  - **策略**：AD-Det（VisDrone 37.5 AP）证明"动态按需贴入"（缺什么贴什么）优于均匀贴入 → 建议每图目标"<64px 可见实例 ≥4"的**尺度分层预算**而非全局 max_paste；
  - **质量**：X-Paste 的承重组件是**入库质量过滤**（CLIP 零样过滤）；我们的对应物 = 入库时过滤深度不可分实例（内外深度差 <1.5σ 的不进 bank），SOC 证明和谐化缺口决定上限（+10.9 LVIS over CP 系）——我们的深度同步贴入正是在补这个缺口；
  - **EMA 组合**：CP-SSOD (arXiv23) 证明 copy-paste + EMA teacher 即可到 SSOD SOTA，小目标 AP 显著——与我们 EMA 0.9999 现状直接对齐。
- **最便宜验证**：3 点网格 max_paste ∈ {4, 12, 20}（固定 seed、20k iter、其余不动），外加 1 点"分层预算版"（小目标优先填充至每图 4 个）。指标：`AP_small`、`小桶 recall@0.3`、`每图 FP 数 @ 等池容量`、`贴入区边缘 FP 计数`（监控上下文失配 FP）、`TP 分数中位漂移`。判据：AP_small 增益/paste 数曲线的拐点即最优值；若 20 仍线性升，考虑加码（文献支持）。

### 候选 5：H-DETR 式 one-to-many 辅助支路（增密正样本的第二通道，候选 1 的放大器）
- **是什么**：训练时给 Mask2Former decoder 加一条 one-to-many 辅助匹配支路（K 个 query 匹配同一 GT），推理丢弃（H-DETR CVPR23：推理同效率；Group DETR/MS-DETR/MDS-DETR 同族）。对 <64px 每步 0.85 个 GT 的饥饿状态，这是把"每 GT 正样本数 ×K"的直接手段。
- **前置条件（关键教训）**：DEIM/MS-DETR/NMS Strikes Back 一致指出：o2m 增密若无质量感知损失/解耦处理，低质量匹配与重复预测会污染分数——**必须与候选 1 (MAL) 打包上线，不可单飞**。
- **收益锚**：H-DETR：Deformable/PETR 系一致提升（~+2 AP 量级）；MS-DETR：无需额外分支版本；Co-DETR（66.0 AP）证明上限但开销超预算。
- **最便宜验证**：候选 1 验证通过后，加 K=4 的 o2m 辅助支路（只挂小目标 GT 或全 GT 均挂、loss 权重 0.5 起步），20k iter 对比候选 1 单独跑。看 `小桶 TP 分数中位` 是否进一步上移且 `AP_small` 增益 > 训练时间增幅（预期 +10-15% 步时，在预算内）。

**推荐执行序**：候选 1 → 候选 4 网格（可并行）→ 候选 2（在 1 之上单因子）→ 候选 5（与 1 打包）→ 候选 3（若排序间距仍不足）。

---

## 4. 反面教训

1. **事后单调重标定不可能救 AP——排序不变性**。AP 对分数的任意单调变换不变（Rank-DETR 的立论出发点；我们分位重标定 -13pt 的机理正在于此：AP 没变好，变的是 operating point 处的池组成，80.5% hard-FP 搭车涌入）。检测校准文献（Küppers CVPRW20、2202.12785、WACV24 2312.06645）的系统结论：temperature scaling/多变量校准改善 ECE 与可靠性，**不改善 AP/排序**；且检测置信度失准是尺度条件的——即便做尺度条件化校准，也仍是事后补丁。
2. **类别长尾的事后法（NorCal、FRACAL）迁移到尺度长尾会复现我们的失败**。它们有效的前提是"稀有类 vs 头类"的分数先验可分（稀有类分数整体被压、FP 多来自头类）；我们的尺度尾里 hard-FP 与 TP **同类同分布**，无可借力的先验分离面。FRACAL 的 +8.6% 稀有类收益不可类比。
3. **增密不配质量感知损失 = 分数通胀**。DEIM 消融：Dense O2O 引入大量低质量匹配，无 MAL 时性能退化；NMS Strikes Back：o2m 强到需要 NMS 兜底；MS-DETR/MDS-DETR 专门设计解耦/掩码处理 o2o-o2m 分数冲突。映射到我们：copy-paste 把小目标正样本 ×10-20 之后，若 cls 损失仍是 one-hot BCE，学到的只是"小目标一律给分"，hard-FP 同步抬升——**贴入通道上线必须与 MAL/质量目标同 commit**。
4. **copy-paste 的 FP 侧信道：上下文失配**。InstaBoost（位置引导才拿到 +2.2）、X-Paste（合成实例必须 CLIP 过滤）、SOC（和谐化缺口价值 +10.9 LVIS）共同说明朴素随机贴入会制造 out-of-context 假阳性监督。我们的专项风险：深度差上限 1.5σ 意味着贴入实例深度对比度弱，**bank 入库时应过滤深度不可分实例**，并监控贴入边缘 FP 计数作为消融判据。
5. **全局 focal γ 调参救不了稀有桶**。EFL 的核心实验：单一 γ 无法同时优化头尾类 AP——对应我们损失点份额 0.1-0.4% 的 <64px 桶。γ 全局加大只会让中大目标欠拟合来换小目标边际改善，净收益为负或噪声级。必须桶级调制（候选 2）。
6. **SAHI 的账**：切片推理 +5.1-6.8 AP 诱人，但推理 ×N 切片违反零开销约束；切片微调 +5-7 AP 的额外收益需要重训，且切片边缘标签截断引入新的 FP 源。若走训练态，Stitcher（+2 AP、零推理开销、反馈式收集小目标样本）是更合规的同族替代。
7. **Mask Scoring 式乘法重排慎投**。MaskIoU 头（CVPR19，~+1 mask AP）本质是"训练期学出来的 score×quality 乘法重排"——与我们已证伪的 score=IoU（-11.9pt）同族，只是质量来自学习而非当前帧估计。在 hard-FP 主导池下先验风险偏高，排位低于改变损失目标的候选 1/2。
8. **MS R-CNN→Mask2Former 的语义差**：Mask2Former 的 cls logit 由 masked-pooling 特征产生，小目标池化像素少导致 logit 系统性收缩——这不是校准问题而是**表征池化问题**，任何纯损失侧方法只能部分补偿；若候选 1/2 后小桶 cls 因子仍贴地，需检查 pooling 分辨率（P3 通道）而非继续调损失。

---

## 5. 引用列表

**分数成型 / 质量感知分类**
- VarifocalNet (CVPR21) — https://arxiv.org/abs/2008.13367
- GFL/QFL (NeurIPS20) — https://arxiv.org/abs/2006.04388
- GFLv2/DGQP — https://arxiv.org/abs/2011.12885
- TOOD (ICCV21) — https://arxiv.org/abs/2108.07755
- OTA (CVPR21) — https://arxiv.org/abs/2103.14259
- RS Loss (ICCV21) — https://arxiv.org/abs/2107.11669 ｜ code: https://github.com/kemaloksuz/RankSortLoss
- Rank-DETR (NeurIPS23) — https://arxiv.org/abs/2310.08854 ｜ code: https://github.com/LeapLabTHU/Rank-DETR
- Align-DETR (BMVC24) — https://arxiv.org/abs/2304.07527
- DEIM (CVPR25) — https://arxiv.org/abs/2412.04234 ｜ code: https://github.com/ShihuaHuang95/DEIM
- Mask Scoring R-CNN (CVPR19) — https://arxiv.org/abs/1903.00241

**DETR 增密监督（o2m 家族）**
- H-DETR (CVPR23) — https://arxiv.org/abs/2207.13080
- Group DETR (ICCV23) — https://arxiv.org/abs/2207.13085
- Co-DETR (ICCV23) — https://arxiv.org/abs/2211.12860
- MS-DETR — https://arxiv.org/abs/2401.03989
- NMS Strikes Back — https://arxiv.org/abs/2212.06137
- FeatAug-DETR — https://arxiv.org/abs/2303.01503
- MDS-DETR — https://arxiv.org/abs/2605.23507
- LoRA-DETR（分配多样性>数量） — https://arxiv.org/abs/2601.09247

**损失再平衡（类尾→尺度尾可迁）**
- Seesaw Loss (CVPR21) — https://arxiv.org/abs/2008.10032
- EFL (CVPR22) — https://arxiv.org/abs/2201.02593

**copy-paste 增密系**
- Simple Copy-Paste (CVPR21) — https://arxiv.org/abs/2012.07177
- Kisantal 小目标增密 — https://arxiv.org/abs/1902.07296
- InstaBoost (ICCV19) — https://arxiv.org/abs/1908.07801
- X-Paste — https://arxiv.org/abs/2212.03863
- Occlusion C&P — https://arxiv.org/abs/2210.03686 ｜ Location-aware C&P — https://arxiv.org/abs/2310.17949
- SDI-Paste (VIS) — https://arxiv.org/abs/2410.13565
- AD-Det (UAV 动态平衡贴入) — https://arxiv.org/abs/2504.05601
- SOC（和谐化合成） — https://arxiv.org/abs/2510.09110
- InstaDA — https://arxiv.org/abs/2509.02973
- Poisson C&P — https://arxiv.org/abs/2308.14772
- CP-SSOD（copy-paste 半监督 SOTA） — https://arxiv.org/abs/2312.06312

**采样/课程/切片**
- Stitcher/DST — https://arxiv.org/abs/2004.12432 ｜ code: https://github.com/yukang2017/Stitcher
- SAHI (ICIP22) — https://arxiv.org/abs/2202.06934 ｜ code: https://github.com/obss/sahi
- SNIP — https://arxiv.org/abs/1808.08786

**微小目标分配**
- NWD — https://arxiv.org/abs/2110.13389 ｜ NWD-RKA — https://arxiv.org/abs/2206.13996

**校准（多为反面教材/边界确认）**
- Multivariate Confidence Calibration (CVPRW20) — https://arxiv.org/abs/2004.13546
- Confidence Calibration for OD and Segmentation — https://arxiv.org/abs/2202.12785
- TCD (NeurIPS22) — https://arxiv.org/abs/2209.07601
- NorCal (NeurIPS21) — https://arxiv.org/abs/2107.02170
- FRACAL (CVPR25) — https://arxiv.org/abs/2410.11774
- 检测校准定义与估计 (WACV24) — https://arxiv.org/abs/2312.06645
- Calibrated Teacher (CVPR23) — https://arxiv.org/abs/2303.07582
- Consistent-Teacher (CVPR23) — https://arxiv.org/abs/2209.01589

---
*备注：标注 ~ 的数字（H-DETR +2 AP 量级、MS R-CNN +1 mask AP 量级）为论文正文量级记忆，摘要页未复核；其余数字均来自 arXiv 摘要页 2026-09-20 抓取。Rank-DETR 的 Swin-L 52.9 细节数字未能在摘要页复核，故表中只保留摘要可证表述。*
