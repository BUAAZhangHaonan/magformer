# 调研B：DETR 系小目标存在性与 query 覆盖

> 调研员：B（文献侧）｜完成日期：2026-09-20｜检索手段：arXiv API + 全文页抓取（WebSearch 配额耗尽，改用 arXiv/S2 API 与 ar5iv 全文）
> 覆盖时间窗：优先 2022-2026。共收录 26 篇，全部核对过 arXiv 编号与摘要/全文数字。

---

## 1. 问题画像（我们的数字 + 种子实验证据）

**我们的设定**：Mask2Former 系，200 可学习 query，训练期二分匹配（已换 AIM：尺度无关 soft-Dice，匹配边距 0.13σ→2.43σ），推理全量输出按分数 top-100 导出；单类工业零件，1024²。

**已量化的核心病灶**：
- **存在性缺口**：38.1% 的 COCO-small GT 在 200 query 输出中找不到任何 IoU≥0.5 掩码——任何分数都不行。这不是打分问题，是**覆盖问题**：没有任何 query "站到" 小目标上。
- **变现门控**：oracle 分数下该缺口值 +37.5pt AP_s；现实分数下仅 +6.7pt。即使覆盖解决，**分数流（cls score）单独门控了收益**——覆盖与分数必须捆绑解决。
- **种子实验（09-14，小模型）**：probe 头找小目标峰值 + 64 种子 query 出生在小目标位置（注意力先验对准）→ 覆盖率 92.1%（IoU≥0.5 拥有小 GT），但 cls 分数停在 0.22-0.29 导不出去。**覆盖可解，分数必须捆绑**。
- **注意力覆盖残余**：8 层解码器仅最后一层消费 stride-4 门控注意力（256² 网格 topk），前 7 层在 stride 32/16/8 粗网格 attend-everywhere；<32px 目标在 stride-4 上仅 2-3 cell。
- **预算约束**：训练开销 +10-20% 内；推理近零开销。

**文献画像对照（本调研的核心发现）**：我们的三个观察在文献中各有一条成熟战线——
1. "正样本供给不足/匹配不稳定"战线（DN-DETR→DINO→Group/H/Co-DETR→DEIM）；
2. "小目标 query 选点偏置（scale bias）"战线（Salience-DETR→DQ/Dome/D³R/HGSQ，AI-TOD-v2 系）；
3. "分类分数与定位质量脱节"战线（Stable-DINO→Align-DETR→DEIM MAL→CLSC/CQTR）。
   我们种子实验的"覆盖 92% 但分数 0.22-0.29"精确对应战线的**断裂点**：1+2 解决覆盖，3 解决变现——三者有现成组件，且大多训练期专用、推理零开销，与预算约束正交。

---

## 2. 方法总表

图例：**⚑** = 与种子实验结论一致；**⚠** = 有张力/教训；**◦** = 中性相关。

### 2.1 查询分配 / 去噪 / 匹配族（覆盖的"供给"侧）

| 方法 | 年份/venue | 机制一句话 | 小目标/收敛收益(数字) | 训练开销 | 推理开销 | 移植改动面 | 兼容性风险 | 种子实验关系 |
|---|---|---|---|---|---|---|---|---|
| DN-DETR | CVPR 2022 | 加噪 GT 作为额外"去噪查询"直接重构 GT，attn mask 防泄漏，绕过二分匹配的不稳定 | R50 12ep 43.4 AP（+1.9 over baseline）；同性能只需 50% epoch | 解码器变长（去噪组数×2 噪声级），整体约 +10-15% | **零**（去噪查询仅训练期） | 几十行：构造噪声 GT→query，加 attn mask | Mask2Former 无现成 mask 版；需自定噪声方式（中心抖动+尺度抖动） | ⚑ 每迭代都有"站在 GT 上"的查询收梯度=覆盖的直接供给 |
| DINO (CDN) | ICLR 2023 | 对比对噪（正/负两组：负样本大扰动→学"此处无物"）+ 混合 query 选择 | R50 12ep 49.4 AP（比 DN-DETR +6.0）；Swin-L 63.3 | 同上，略增 | **零** | 同上+负样本分支 | 负样本的"无物"目标与单类小零件的假阳性抑制对路 | ⚑ 正样本管覆盖、负样本管分数纪律——天然捆绑 |
| Group DETR | ICCV 2023 | 11 组×300 query 组内 o2o、组间等效 o2m；组内独立自注意力 | Conditional-DETR 12ep 32.6→37.6（+5.0）；**Mask2Former R50 mask AP 12ep 38.5→39.7（+1.2）** | +5 min/epoch（23→28、28→33），内存 +1.2-1.7GB ≈ +18% | **零**（推理只留 1 组） | 加 K-1 组 query+分组自注意力+分组匹配 | **⚠ naive o2m（3300 query 不分组）直接崩到 8.4 AP**——正样本洪峰必须隔离 | ⚑ 供给增加惠及小目标；⚠ 分组隔离是硬前提 |
| H-DETR | CVPR 2023 | 主干 o2o 300 query + 辅助 o2m 分支 1500 query（K=6 重复 GT） | Deformable-DETR R50 12ep 47.0→48.7（+1.7）；SwinL 54.5→55.9 | ~+7% 训练时间（实测） | **零**（268 GFLOPs 不变） | 辅助 query 集+第二套匹配成本 | o2m 分支输出与 o2o 分支需严格分离 | ⚑ 同上，更省的 o2m 变体 |
| Co-DETR | ICCV 2023 | 多个辅助头（ATSS/FasterRCNN 式 o2m）+ 从辅助头正样本坐标提取"定制正查询"回灌解码器 | DINO-Deformable SwinL 58.5→59.5；ViT-L 66.0 test-dev | 辅助头+回灌查询 | **零**（辅助头全丢弃） | 需外挂 2-3 个异构辅助头——对 Mask2Former 是大手术 | 异构头的匹配语义与 mask 匹配(AIM)不同轨 | ◦ 证据最强但改动面超出 +20% 预算 |
| Semi-DETR | CVPR 2023 | 阶段式混合匹配：先 o2m 稠密化正监督→再 o2o 收尾 + 跨视角 query 一致性 | SSOD 各 setting SOTA（COCO/VOC） | 两阶段训练 | 零 | 训练日程重构 | 需两阶段日程，与现训练管线不合 | ⚑ "o2m 先、o2o 后"印证覆盖先行的次序 |
| DEIM | CVPR 2025 | 稠密 o2o（mosaic/mixup 增目标数）+ 匹配感知损失 MAL（按质量加权） | RT-DETRv2 53.2 AP/单卡 4090 一天；~50% 训练时间削减 | 负开销（加速） | 零 | 数据增广+损失替换 | MAL 的质量度量针对框 IoU，需换 soft-Dice | ⚑ MAL=分数与匹配质量绑定的最新形态 |
| Align-DETR | BMVC 2024 | 联合质量度量的对齐损失+指数降权（正→负平滑过渡）+ H-DETR 式 m2o | H-DETR 基线 +0.6；50.5 AP(1x)/51.7(2x) | ~0 | 零 | 损失函数级改动 | 指数降权超参需调 | ⚑ 直击"cls 分数≠定位质量" |
| Stable-DINO | ICCV 2023 | 位置监督损失（IoU 等位置度量监督正样本 cls）+ 位置调制匹配代价 | R50 12ep 50.4 AP / 24ep 51.5；SwinL 63.8 | ~0 | 零 | 损失+cost 两处替换 | 位置度量选 IoU；我们要换尺度无关 soft-Dice | ⚑⚑ **与 AIM 最同构的先例：位置度量进 cost+进 cls 目标** |

### 2.2 小目标专用 query/proposal 族（覆盖的"选点"侧）

| 方法 | 年份/venue | 机制一句话 | 小目标/收敛收益(数字) | 训练开销 | 推理开销 | 移植改动面 | 兼容性风险 | 种子实验关系 |
|---|---|---|---|---|---|---|---|---|
| DQ-DETR | ECCV 2024（v1=IJCAI 2022 DQDETR） | 类别化计数模块预测目标数+密度图→动态决定 query 数量与位置增强 | AI-TOD-v2 SOTA 30.2 mAP | 计数头+动态批次 | 推理含计数头+可变长 query（**非零**） | 计数头+query 数量调度 | 可变长 query 与固定 shape 批处理冲突 | ⚑ 固定 query 数对小目标密集场景不敷使用的直接证据 |
| Salience-DETR | CVPR 2024 | 层级显著性过滤（region 级→query 级）+**尺度无关显著性监督**+查询精化 | 三个任务专用集 +4.0/+0.2/+4.4 AP；COCO 49.2 AP 且 FLOPs 更低 | 编码器重排 | 编码器更省 | 两阶段选点改造 | 过滤阈值对小目标漏检敏感（作者用尺度无关监督专门对冲） | ⚑ "query 选点偏大目标"被点名为一级问题 |
| Dome-DETR | ACM MM 2025 | 密度焦点提取器出前景掩码→稀疏窗口注意力+渐进自适应 query 初始化（PAQI） | AI-TOD-v2 +3.3 AP；VisDrone +2.5 AP | 轻量密度模块 | 计算量更低（稀疏化） | 密度分支+注意力稀疏化 | 窗口稀疏化改动编码器本体 | ⚑ 密度引导 query 初始化=种子思想的密度版 |
| D³R-DETR | arXiv 2026-01 | 空间+频域双域精化低层特征→更准密度图引导 query 定位 | AI-TOD-v2 SOTA（摘要无具体数） | 双域分支 | 增分支 | 双域编码器 | 工程复杂 | ⚑ 密度引导仍是主线 |
| HGSQ | arXiv 2026-09 | 热图预算预测器（一次前向出前景预算图）→稀疏 query 选点+自适应 query/解码器预算（AQDB） | NWPU 95.10 mAP50；VisDrone 54.8 mAP50；96 FPS（4070/TensorRT FP16） | 热图头 | 热图头一次前向（轻） | 热图头+选点替换 | 单作者新作，数字未过同行评审 | ⚑⚑ **与我们 probe 头+种子 query 几乎同构**，差异在他们把热图当"预算"而非可视化 |
| HMPE | arXiv 2025-04 | 热图诱导的高质量解码器 query（HIDQ）+多尺度目标框热图融合编码 | NWDU VHR-10 +1.9 mAP；VOC +1.2；**解码器 8→3 层** | 热图监督 | 更省（层数减半） | 热图分支+query 生成替换 | 小数据集上的温和增益 | ⚑ 热图生 query 路线又一证；⚠ 8→3 层暗示前几层粗注意力价值有限（与我们"前 7 层 attend-everywhere"疑虑一致） |
| HELP（Learning Where to Embed） | ICMR 2026 (arXiv 2604.15065) | 只在前景显著处保留位置编码、梯度掩码过滤背景主导 query | 参数 163M→66.3M（-59.4%），解码器 8→3，精度反升 | 梯度热图监督（仅训练期） | 更省 | 位置编码注入策略 | 同上，会议级证据 | ⚑ 背景 query 噪声是负资产 |
| SO-DETR | arXiv 2025-04 | 双域混合编码器+**扩展 IoU 判据的增强 query 选择**+蒸馏 | VisDrone/UAVVaste 超同算力方法（摘要无数） | 双域编码器 | 双域编码器 | 编码器+选点 | 扩展 IoU 对极小目标的 IoU 病态（我们换 soft-Dice 的原因） | ⚑ 选点判据要尺度无关——与我们 AIM 同一结论 |
| NRQO（Revisiting DETR for SOD） | ICME 2025 | 抗噪 FPN + 成对相似度 RPN 产出足量高质量正查询 | 多 benchmark 超 SOTA（摘要无数） | RPN 分支 | RPN 仅训练期 | 外挂 RPN | 与 Mask2Former 管线融合需适配 | ⚑ "正查询数量×质量"双诉求的又一定名 |
| CLSC DETR | arXiv 2026-08 | 跨层局部支撑（末层 query 关联中间层候选聚合几何证据）+分类/定位一致性校准 | VisDrone +1.5 AP、+2.0 AP₇₅ | 跨层关联 | 轻微（关联模块） | 解码器内跨层连线+校准 | 单作者新作 | ⚑⚑ **直接命中我们分数卡 0.22-0.29 的病灶**："单 query 几何证据不足→质量估计不可靠→排序不稳" |
| CQTR（Reading Decoder Trajectories） | arXiv 2026-09 | 训练自由：反事实尺度干预激发潜响应+读解码轨迹（空间收敛/语义持续/跨尺度冲突）判候选可靠性，重打分导出 | 9 个冻结检测器×3 数据集全升：COCO APs +1.95~3.81；TinyPerson 最高 +3.85 AP；VisDrone 最高 +2.66 AP；仅用 100 张无标注图校准 | **零训练** | 重打分逻辑（离线/导出期） | 导出端后处理 | 轨迹信号阈值需按模型-数据对选择 | ⚑⚑⚑ **"小目标知识在冻结模型里存在但欠激活"=我们 oracle +37.5 vs +6.7 的独立复现与变现路径** |

### 2.3 锚定/位置先验 query 族（种子路线的近亲）

| 方法 | 年份/venue | 机制一句话 | 小目标/收敛收益(数字) | 训练开销 | 推理开销 | 移植改动面 | 兼容性风险 | 种子实验关系 |
|---|---|---|---|---|---|---|---|---|
| Anchor DETR | AAAI 2022 | query=锚点，每个 query 只管锚点附近目标；锚点空间解耦注意力 | R50-DC5 44.2 AP；比 DETR 少 ~10× epoch | 0 | 持平 | query 参数化方式 | 锚点网格密度限制小目标覆盖 | ⚑ 位置先验加速收敛的起点证据 |
| DAB-DETR | ICLR 2022 | 动态锚框作为 query，逐层更新；宽高调制位置注意力 | R50-DC5 50ep 45.7 AP | 0 | 持平 | query 参数化 | 框先验，对 mask 任务要改 | ⚑ 解码器=级联软 RoI 池化的解释支持"出生位置决定覆盖" |
| Sparse R-CNN | CVPR 2021 | 可学习 proposal 框+特征，迭代动态实例交互头（级联精化） | R50-FPN 3x 45.0 AP，22fps | 0 | 持平 | proposal 集即 query 集 | 稀疏 proposal 对密集小目标需扩容（500 级） | ⚑ 稀疏集合检测可行，但**预算**必须匹配目标密度（我们 200 vs 图内实例数） |
| Mask DINO | CVPR 2023 | 检测/分割统一：编码器 query 选择 + **mask 版 CDN** + 混合选择 | COCO IS 54.5 AP / 全景 59.4 PQ | 同 CDN | 零（去噪仅训练） | Mask2Former→Mask DINO 的差异点正是 query 选择+CDN | 全量迁移=换架构；**只摘 CDN-mask 组件则小改** | ⚑ Mask2Former 加 CDN-mask 的现成参考实现 |
| DDQ | CVPR 2023 | 先稠密 query 铺满覆盖，再"去重选择"出 distinct query 做 o2o | R50 12ep 52.1 AP；CrowdHuman 93.8 AP | 编码器加密 | 选点后与稀疏 DETR 同 | 稠密化+选点两段 | **⚠ 稠密 query 的重复项若不选走会破坏 o2o 训练** | ⚑ "覆盖要靠密、唯一性要靠选"与我们"覆盖率×唯一导出"同构 |
| Focus-DETR | ICCV 2023 | 双注意力编码器：定位+类别双分数给 token 打分，只编码前景 | COCO 50.4 AP（+2.2），复杂度对标稀疏 DETR | 打分器 | 编码器更省 | token 打分器 | 前景打分对小目标漏检敏感 | ⚑ 编码器算力确实浪费在背景——支持我们 stride-4 门控残余的分析 |
| QueryDet | CVPR 2022（CNN 但路线同） | 低分辨率热图定位→高分辨率层只算被查询位置（级联稀疏查询） | COCO +1.0 AP/**+2.0 AP-small**；高分辨率推理 3.0×；VisDrone 2.3× | 0 | 更快 | 热图→高分辨率稀疏计算 | CNN 检测头范式 | ⚑⚑ 低分辨率找、高分辨率稀疏算=我们"probe 找峰+stride-4 门控"的 CNN 先例 |
| PanSR | arXiv 2024-12 | 对象中心 query 生成+监督重设计，修复"query 提案偏大目标、良好 query 漂移到别物、近邻实例合并" | LaRS +3.4 PQ；Cityscapes SOTA | 重设计 query 生成 | 持平 | mask transformer 内部重构 | 面向全景驾驶/海事场景 | ⚑⚑ 把"query 提案偏大目标→小目标漏检"作为 mask-transformer 一级问题点名的分割侧文献 |

> 检索备注：用户方向里提到的 "DNTR" 与 "Infusion（实例注入式）" 在 arXiv 检索无对应 DETR 论文（DNTR 零命中；Infusion 命中全是扩散/风格类），疑为记忆偏差，未收录，其功能位由 Co-DETR 定制正查询、PanSR 对象中心 query、HGSQ 热图选点覆盖。

---

## 3. Top-5 可移植候选

针对两个靶点：**A. 38.1% 无检测（存在性）**；**B. 覆盖×分数必须捆绑（oracle +37.5 vs 现实 +6.7）**。全部满足训练 +10-20%、推理近零。

### 候选 1：Mask 版对比去噪（CDN-mask，DN-DETR→DINO→Mask DINO 谱系）
- **论证**：去噪查询每迭代把"加噪 GT"直接放进解码器并在 GT 位置收重构监督——这是对存在性缺口的**直接供给**（无需匹配决定生死）；CDN 负样本同时训练 cls 头在"似是而非处"输出低分，是分数纪律的免费赠品。正负两组天然把覆盖与分数捆在一次前向里。Mask DINO 已验证 mask 版可行（54.5 AP IS），组件可单独摘。
- **最便宜验证**：小模型上外挂 G=2 组×~100 个 mask 噪声查询（中心+尺度抖动，正/负两组），attn mask 隔离；跑 1-2 天。看两个仪表：存在率（任意分数 IoU≥0.5 命中率，基线 61.9%）与小 GT 上 cls 分布（基线 0.22-0.29）。解码器算力 +30-50% 但只占总步时 ~10-15%（pixel decoder 主导），总额在预算内。
- 风险：Mask2Former 无官方 mask-CDN，噪声在 mask 语义下的定义（中心/尺度/边界抖动）需自定。

### 候选 2：Group DETR 式分组 o2m 辅助查询（有 Mask2Former 直接数字）
- **论证**：唯一的**在 Mask2Former 上有公开数字**的 o2m 方案：mask AP 12ep 38.5→39.7（+1.2）、50ep +0.3；组内 o2o、组间等效 o2m——每个小 GT 每迭代被 K 组各领一次，正样本供给×K；推理只留 1 组，零开销；实测 +5min/epoch（≈18%），贴预算上限但合格。它与候选 1 正交可叠加（DINO=DN+CDN，Group DETR 可加在 DINO 上，文献明确兼容）。
- **最便宜验证**：K=3-5 组×100 query（文献 11×300 的缩小版），组间独立自注意力，分组匹配。重点盯**表 7 的反例**：不分组直接 3300 query o2m 会崩到 8.4 AP——隔离是硬前提。
- 风险：内存 +1.2-1.7GB（文献值）；我们 1024² 输入更大，需测。

### 候选 3：位置质量→分类分数绑定（Stable-DINO / Align-DETR / DEIM-MAL 谱系）——**改动最小、直击"捆绑"**
- **论证**：Stable-DINO 的处方="用且仅用位置度量（IoU）监督正样本的 cls 分数"+位置度量进匹配 cost。**我们的 AIM 已把尺度无关 soft-Dice 放进了 cost——这条路线的前一半我们已经走了**，缺的只是后一半：把 matched query 的 focal 目标从 1 换成 σ-Dice 值（或其变换），让分数流直接回归掩码质量。训练开销≈0，推理开销=0，改动是损失函数几行。种子实验的 92.1% 覆盖×0.22-0.29 分数正是"位置度量未进 cls"的症状；CLSC（2026）进一步指单 query 几何证据不足，可加跨层支撑。Align-DETR 的指数降权与 DEIM 的 MAL 提供了质量加权的成熟形态。
- **最便宜验证**：一天实验——只改 cls 目标：matched query 的目标 = soft-Dice(GT, pred) 的温度映射，重训短程，量 AP_s（现实分数）是否从 +6.7 向 oracle +37.5 方向移动，及种子 query 的分数分布是否脱离 0.22-0.29。
- 风险：focal 与质量目标的耦合需调温度；单类场景负样本稀少，CDN 负组（候选 1）是天然补充。

### 候选 4：热图/计数引导的种子 query + 自适应预算（DQ-DETR→Dome→HGSQ 家族，与 09-14 实验同血统）
- **论证**：这条线就是种子实验的文献化：HGSQ 的热图预算预测器+稀疏 query 选点+AQDB 自适应预算、Dome 的密度引导 query 初始化（AI-TOD-v2 +3.3）、DQ-DETR 的计数模块（AI-TOD-v2 30.2）。它们证明了**覆盖侧的顶格做法**（我们已达 92.1%），并且 HGSQ 把热图当"实时计算预算"的用法给我们 stride-4 门控的推广提供了样板（前 7 层粗网格 attend-everywhere 的残余可换成预算化稀疏消费）。与候选 3 捆绑即"种子出生×质量监督"完整闭环。
- **最便宜验证**：我们 probe 头已在——只需给它加两条：(a) 小目标计数/密度回归（DQ-DETR 式）用于把 200 预算按图内小目标密度倾斜；(b) 每种子质量回归分支（=候选 3 的 per-query 化）。训练 +5-10%。
- 风险：推理期热图头是**非零开销**（一次轻前向，HGSQ 报 96 FPS 可控）；可训练期专用规避。

### 候选 5：CQTR 式训练自由轨迹重打分（导出端变现）
- **论证**：CQTR 用反事实尺度干预+解码轨迹三信号（空间收敛/语义持续/跨尺度冲突）在**不训练**的前提下给冻结检测器重打分，9 检测器×3 数据集全部提升，APs +1.95~3.81、TinyPerson 最高 +3.85 AP。这独立复现了我们的核心判断——**小目标知识在模型里存在但被分数流淹没**（oracle +37.5 vs +6.7）——并给出零训练的变现路径：top-100 导出的排序键从 cls 换成轨迹可靠性复合分。
- **最便宜验证**：零训练、纯离线：在 09-14 种子实验的 checkpoint 上实现三信号（跨层 query 位移收敛度、类别一致性、尺度干预前后一致性），对小 GT 候选重排，量现实分数下的 AP_s 增量。若 +6.7pt 明显抬升，即证明"分数门控"可在不动权重下撬动。
- 风险：信号阈值需 100 张无标注图校准（文献做法）；与我们 top-100 导出协议的接口需自定义。

**叠加顺序建议**：候选 5（零成本，先变现存量）→ 候选 3（几行损失，捆绑分数）→ 候选 1/2（+15-18%，供覆盖）→ 候选 4（预算化，推广 stride-4 门控到多层）。

---

## 4. 反面教训

1. **naive o2m 崩溃**（Group DETR 表 7）：3300 query 不分组直接 o2m，AP 从 32.6 跌到 8.4。正样本洪峰会摧毁 o2o 分数语义——任何 o2m 增供必须分组/分分支隔离。与我们"覆盖×分数捆绑"同一枚硬币的反面。
2. **去噪无隔离=作弊**（DN-DETR）：attn mask 防泄漏是去噪成立的前提，漏了则任务平凡化、无监督价值。
3. **稠密 query 不去重的毒副作用**（DDQ）：稠密化覆盖的同时产生大量相似 query，破坏 o2o 训练——覆盖率指标（我们 92.1%）必须与唯一性/去重联合度量，否则假阳性换真覆盖。
4. **动态 query 数量不是银弹**（DQ-DETR 谱系）：计数模块+动态数量已 SOTA AI-TOD-v2，但后续 Dome/D³R 仍要叠密度引导初始化——数量解决"够不够"，选点质量与分数仍需单独机制。与种子实验"覆盖已达 92%、分数卡 0.22-0.29"同构。
5. **query 选点的尺度偏置**（Salience-DETR/PanSR）：两阶段 query 选择天然偏向大目标；Salience-DETR 专门发明"尺度无关显著性监督"对冲，PanSR 在 mask transformer 侧点名"query 提案偏大→小目标漏检"。我们的 AIM（尺度无关 soft-Dice）方向正确，但只进了匹配 cost，未进选点与 cls（候选 3/4 的动机）。
6. **编码器/前层注意力的价值存疑**（Focus-DETR/HMPE/HELP）：背景 token 白耗算力、解码器 8→3 层精度反升——我们"前 7 层粗网格 attend-everywhere"的残余可能是负资产而非资产，预算化（HGSQ 式）优于加层。
7. **推理期架构性开销要警惕**（DQ-DETR 的推理计数头、HGSQ 的热图头）：凡推理端引入模块的方案与"近零开销"约束冲突，训练期专用（CDN/Group/H-DETR/损失类）才是合规形态。
8. **单一会议/单作者新作的证据强度分层**：HGSQ、CLSC、CQTR、PanSR 均为新作（其中三个单作者/未评审），方向一致但数字引用需谨慎——验证实验应先在小模型复算。

---

## 5. 引用列表

**匹配/去噪/供给族**
1. DN-DETR (CVPR 2022) — https://arxiv.org/abs/2203.01305 ｜ code: github.com/FengLi-ust/DN-DETR
2. DINO (ICLR 2023) — https://arxiv.org/abs/2203.03605 ｜ code: github.com/IDEA-Research/DINO
3. Group DETR (ICCV 2023) — https://arxiv.org/abs/2207.13085 ｜ code: github.com/Atten4Vis/GroupDETR
4. H-DETR: DETRs with Hybrid Matching (CVPR 2023) — https://arxiv.org/abs/2207.13080 ｜ code: github.com/HDETR
5. Co-DETR (ICCV 2023) — https://arxiv.org/abs/2211.12860 ｜ code: github.com/Sense-X/Co-DETR
6. Semi-DETR (CVPR 2023) — https://arxiv.org/abs/2307.08095
7. Align-DETR (BMVC 2024) — https://arxiv.org/abs/2304.07527 ｜ code: github.com/FelixCaae/AlignDETR
8. DEIM (CVPR 2025) — https://arxiv.org/abs/2412.04234 ｜ code: github.com/ShihuaHuang95/DEIM
9. Stable-DINO (ICCV 2023) — https://arxiv.org/abs/2304.04742 ｜ code: github.com/IDEA-Research/Stable-DINO

**小目标 query 族**
10. DQ-DETR (ECCV 2024；IJCAI 2022 前身 DQDETR) — https://arxiv.org/abs/2404.03507 ｜ code: github.com/hoiliu-0801/DQ-DETR
11. Salience-DETR (CVPR 2024) — https://arxiv.org/abs/2403.16131 ｜ code: github.com/xiuqhou/Salience-DETR
12. Dome-DETR (ACM MM 2025) — https://arxiv.org/abs/2505.05741 ｜ code: github.com/RicePasteM/Dome-DETR
13. D³R-DETR (arXiv 2026) — https://arxiv.org/abs/2601.02747
14. HGSQ (arXiv 2026) — https://arxiv.org/abs/2609.13306
15. HMPE (arXiv 2025) — https://arxiv.org/abs/2504.13469
16. HELP: Learning Where to Embed (ICMR 2026) — https://arxiv.org/abs/2604.15065
17. SO-DETR (arXiv 2025) — https://arxiv.org/abs/2504.11470 ｜ code: github.com/ValiantDiligent/SO_DETR
18. NRQO (ICME 2025) — https://arxiv.org/abs/2507.19059
19. CLSC DETR (arXiv 2026) — https://arxiv.org/abs/2608.21457
20. CQTR: Reading Decoder Trajectories (arXiv 2026) — https://arxiv.org/abs/2609.06581

**锚定/稀疏/分割侧**
21. Anchor DETR (AAAI 2022) — https://arxiv.org/abs/2109.07107 ｜ code: github.com/megvii-research/AnchorDETR
22. DAB-DETR (ICLR 2022) — https://arxiv.org/abs/2201.12329 ｜ code: github.com/SlongLiu/DAB-DETR
23. Sparse R-CNN (CVPR 2021) — https://arxiv.org/abs/2011.12450 ｜ code: github.com/PeizeSun/SparseR-CNN
24. Mask DINO (CVPR 2023) — https://arxiv.org/abs/2206.02777 ｜ code: github.com/IDEA-Research/MaskDINO
25. DDQ (CVPR 2023) — https://arxiv.org/abs/2303.12776 ｜ code: github.com/jshilong/DDQ
26. Focus-DETR: Less is More (ICCV 2023) — https://arxiv.org/abs/2307.12612
27. QueryDet (CVPR 2022) — https://arxiv.org/abs/2103.09136 ｜ code: github.com/ChenhongyiYang/QueryDet-PyTorch
28. PanSR (arXiv 2024) — https://arxiv.org/abs/2412.10589 ｜ code: github.com/lojzezust/PanSR

> 数据出处说明：各条数字均取自 arXiv 摘要页或 ar5iv 全文页（Group DETR / H-DETR 的表格数字来自 ar5iv 全文抓取）；标注"摘要无数"者未取得精确数字，仅收录机制与定性结论。WebSearch 配额于 2026-09-25 重置后可补充 S2 引用数与 leaderboard 交叉核验。
