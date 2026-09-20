# P4-c 提案 #5：「配对成型」——分层贴入增密（DEIM Dense-O2O 同构）× AIM 度量质量感知分类目标（MAL-Q）

提案人 #5 / 5 ｜ 2026-09-21 ｜ 场：曝光与分数成型
一句话主张：**把 DEIM (CVPR25) 的"增密必配质量感知损失"配对整体移植——贴入通道承担 Dense-O2O（供 给），AIM 的尺度无关 soft-Dice 从匹配代价延伸进分类目标（MAL-Q，语义）——同一个 commit 上线，攻击"每步 0.85 个可见 <64 GT"与"one-hot focal 均衡点卡在 0.29"两个根因。**

---

## 0. 证据定位（本提案攻击的确切病灶）

| 病灶证据 | 数字 | 本提案的对应机制 |
|---|---|---|
| 小桶 TP 分数中位 0.29 vs 大目标 0.80，24K 起持平 88K 步（r1_score §2：斜率 −0.003/8K） | 均衡点问题，非"未成熟" | MAL-Q 改变分类目标的均衡点定义 |
| batch=4 每步可见 <64 GT 0.85 个；小目标像素份额 0.124%、损失点份额 0.1-0.4%（r5_data §1） | 正样本饥饿 | 分层贴入预算：可见 <64/步 ×3~5 |
| <64 det 池 hard-FP 80.5%、AUC 0.753；64-256 桶 AUC 0.490（r1_score §3） | 抬分即 FP 搭车 | 质量目标只给 matched keeper，FP query 只见负监督 |
| 事后抬分全灭：分位重标定 −13.1、score=IoU −11.9、matched→1 −14.85（r1_score §1） | 排序不变性 + 重复搭车 | 训练期改变学的排序，非事后单调变换（见 §1.3） |
| 每命中 <64 GT 平均 14.05 个 IoU≥0.5 det（喷洒）（r1_score §3） | query 竞争无纪律 | o2o 下仅 1 个 keeper 得正目标，其余 13 个被压向 eos |
| copy-paste 已修复可启用，默认参数 bank 200 / max_paste 5 = COCO 量级（r5_data §4-5） | E[贴入小目标/图]≈1.2，密度仅 ×1.13 | 参数重设 + 分层预算 + 入库质量过滤 |
| 深度对比度 1.5σ 上限（r5_data §3：p50 \|Δ\|≈0.038 norm） | 弱模态信号 | 入库过滤：只收深度可分（≥1.5σ_raw）实例进 bank |

---

## 1. (a) 机制因果链

### 1.1 根因一：正样本饥饿（供给侧）

小目标分类分数是"晚成型、密度驱动"的量（r5_data §9 内部旁证：小模型臂密度×2-3 时 TP 分数成熟从最小桶扩散；c0 全量 300K 步 AP_s 仍 0.236 证明图数≠密度）。当前协议下每步只有 0.85 个可见 <64 GT 进入二分匹配，正分类梯度质量份额 ~5%（<64 实例占比 5.25% × 更晚饱和），focal 的均衡点因此停在"小目标 cls 不饱和"处——24K→112K 斜率 −0.003/8K 是**均衡点证据**而非优化衰减证据（r1_score §2 明确排除 lr 解释）。**不增加每步正样本供给，任何损失侧改动都在与均衡点拔河。**

机制：`CopyPasteTransform`（transforms.py:1007）重设为**尺度分层预算**——每图先数可见小目标（<1024px²，均值 2.37、p50 1），不足目标数则从 bank 按需补齐（AD-Det arXiv25 "缺什么贴什么"），bank 扩容 200→2000（合格源池 56,123，扩容 10 倍仍有 28 倍余量）、prefill 预填防冷启动。深度通道已同步贴入（transforms.py:1131-1141 已实现 write_mask 保护），使贴入实例对 DCCG 融合是模态一致的。

### 1.2 根因二：分类目标不携带可达质量（语义侧）

现状 `_loss_labels`（criterion.py:390-421）：matched query 的 one-hot 目标 = 1，`sigmoid_focal_loss(alpha 0.25, gamma 2)`。one-hot 目标下 cls 分数的均衡位置由梯度质量竞争决定（正 13.9 vs 负 10.9，1.27:1 偏正但小目标只占正质量的 ~5%），且**分数语义与掩码可达质量完全脱钩**——query 画出一个 soft-Dice 0.55 的模糊小掩码与画出 0.9 的好掩码，分类监督一字不差。

机制（MAL-Q）：matched query 的分类目标从 1 改为 **q̃ = clamp(q, min=q_floor).pow(τ)**，其中 q = 匹配对 (pred_mask, GT) 的 soft-Dice，用与 AIM 完全相同的**确定性 R×R 网格**（GT bbox 外扩 30%、R∈[8,64]、无 RNG——matcher.py:276-354 的构造）逐层 no_grad 计算，detach 不回传。损失形式取 GFL/QFL（NeurIPS20）的软目标推广：

```
L_cls = Σ_matched   |σ(z) − q̃|^β · BCE(z, q̃)        （QFL 正支，β=2 起步）
      + Σ_unmatched eos_coef · sigmoid_focal_loss     （负支原样保留，含 eos 通道）
```

这正是 DEIM-MAL（CVPR25）的 DETR 版配对件（其 q=框 IoU，我们换成已在匹配代价里被验证的尺度无关 soft-Dice——调研B 候选 3 点名 Stable-DINO 谱系"位置度量进 cost + 进 cls 目标"只走了前一半，AIM 已进 cost，本提案补后一半）。**两个根因的修法必须同 commit**：DEIM 消融明确 Dense O2O 无 MAL 时性能退化（调研C 反面教训 3）——贴入把小目标正样本 ×3-5 之后若 cls 仍是 one-hot，学到的只是"小目标一律给分"，hard-FP 同步抬升。

### 1.3 为什么这不同于已证伪的事后路线（关键论证）

- **分位重标定 −13.1 的机理**是 AP 对单调变换不变，只移动 operating point，80.5% hard-FP 搭车涌入。MAL-Q 是**训练期**改动：它改变模型学到的排序本身——FP query 从未收到正目标，其分数在训练中被持续下压；keeper 分数被拉向可达质量。排序改变 + 存在性改变，恰是 P4-d 调研指出的"仅允许改变排序或存在性的机制"。
- **score=IoU −11.9 的机理**是按质量排序把"高 IoU 重复 det"排到"低 IoU 唯一 TP"前面。事后变换**无法知道哪个 det 是 keeper**；训练期的二分匹配天然知道——MAL-Q 只给 matched keeper 质量目标，14.05 个喷洒重复全部是 unmatched、被压向 eos。重复- keeper 的不对称性正是事后变换无法表达、o2o 训练期可以表达的信息。
- **Mal-maskness 双重计入风险**（matched p50 0.613/0.723/0.821 随桶单调）：最终 score = cls × maskness，若 cls 目标 q 与 maskness 高相关会双重压小目标。对策：(i) τ<1 的温度映射抬升目标（q^0.75）；(ii) q_floor 防塌；(iii) 冻结探针实测 q 分布后定 τ（§5 探针），并把"q̃ 中位 ≈ 期望 cls 中位"作为标定准则，而非简单取 τ=1。

### 1.4 因果链小结

贴入增密（每步 <64 正样本 0.85→3-5，损失点份额 0.1-0.4%→0.6-1.5%）→ AIM 全覆盖匹配把增量供给稳定分配（已修复，matcher.py:322 无数量帽）→ MAL-Q 让每个正样本的分类监督携带"这个匹配实际可达的质量"，keeper 上分、喷洒下沉、贴入带来的低质量匹配不强推 → 小桶 TP 分数中位脱离 0.29 均衡点，且 FP 不同步上抬 → 存在性（topk 切割前 cls 提高）与分数变现（E3 缺口 16.5pt 的训练侧份额）同时受益。

---

## 2. (b) 与二分匹配 / set-prediction 的兼容性

1. **匹配器零改动**。MAL-Q 只作用于匹配完成之后的损失目标；`HungarianMatcher.forward`、AIM 列构造、代价权重全部不动。o2o 语义保留：每个 GT 仍恰好 1 个正样本，"软"在目标数值不在分配数量——GFL/VFL/Stable-DINO/DEIM 全系都在 Hungarian 匹配下运行，范式无冲突。
2. **q 与匹配代价同源**。q 复用 AIM 的确定性网格 soft-Dice（同一统计量同一实现），Stable-DINO 处方（"用且仅用位置度量监督正样本 cls"）的完整落地；不存在"代价用一个度量、目标用另一个度量"的语义分裂。
3. **贴入只是扩大 GT 集合**。每图 GT 50→~55-58，代价矩阵 200×N 加宽几列，匈牙利原生处理；set-prediction、双塔、解码器结构零改动。
4. **强化而非稀释唯一性**：与 naive o2m（Group DETR 表 7：3300 query 不分组 32.6→8.4）相反，本方案下喷洒重复收到的是更明确的负监督——o2o 的唯一性语义被 MAL-Q 锐化。
5. **o2m 打包条款满足**：场规则"o2m 增密类必须与质量感知损失打包、不得单飞"。本提案的增密走数据侧（DEIM 自己的 Dense-O2O 就是增广增目标，非加 query），且 MAL-Q 在同一 commit——按构造合规；对 P4-a 的 query 侧 o2m，本提案预装了其必备配对件（§7）。

---

## 3. (c) 理论收益上界（引调研数字注出处）

**收益构成**（对 AP_s，trainer 口径基线 0.2638 / 标准口径 0.2925）：

| 成分 | 依据 | 量级 |
|---|---|---|
| 贴入增密（存在性侧） | Ghiasi CVPR21：COCO +1.2 AP / LVIS +2.4-3.7 / 小目标专用线 +3-7pt（调研C 表 #24-25）；R5 内部估算修复深度后 max_paste 12-20 → AP_s +2~5pt；密度 ×2.5（2.37→~6 小目标/图）落在文献支持区间偏保守侧（Kisantal 支持更高） | +2~5pt |
| MAL-Q（分数变现侧） | VFL CVPR21 +2.0 AP / GFL ~+1（调研C 表 #1-2）；DEIM 消融：MAL 是增密收益的**兑现条件**而非独立加成（无 MAL 的 dense 匹配退化） | 0~+2pt（含防搭车保底） |
| 组合包上界 | E3 检测存在性+分数缺口 16.5pt（rescore 口径，SYNTHESIS §一）是整个复数的硬天花板，P4-a/b/d 各认领一部分 | **+3~6pt 现实预期，+8pt 理论上界** |

**针对我们数字结构的独立论证**：(i) <64 det 池 AUC 0.753 说明 cls 信号并非不存在而是未成型——MAL-Q 直接优化该排序；(ii) recall@0.5_small = 0.619，贴入攻击的是"query 从不学会站上小目标"的 62% 缺检成分（R1 §5 分解）；(iii) 0.29→0.40 的 TP 中位位移本身即解除 topk-100 切割的挤压（keeper 图内排名中位 85/100 的直接成因是 cls 低）。诚实的边界：若 MAL-Q 后小桶 cls 因子仍贴地（<0.35）而 q 已上升，则是 masked-pooling 表征上限（调研C 反面教训 8），移交 P3-b/P4-b，本提案的探针可提前识别。

---

## 4. (d) 开销预算（算式）

**训练 ms/it（基线 1.44-1.60 s/it，红线 +20% = +288-320ms）**：

- MAL-Q 的 q 计算（GPU、no_grad、仅前向）：每步 matched 对 ≈ B×GT/图 = 4×54 ≈ 216；每对 R×R ≤ 64² = 4096 点（混合尺度均值 ~2500）→ 216×2500 ≈ **0.54M 点采样/层 × 9 层（深监督逐层独立匹配）= 4.9M/步**。对照：现行 mask 损失 12544 点 × 216 对 × 9 层 ≈ 24M 点采样（含反向）。即 q ≈ 现行点采样前向 FLOPs 的 ~20%。显存读 ≈ 4.9M×4B×2 ≈ 40MB/步 → 亚毫秒；瓶颈是 kernel 启动（9 层 × ~4 kernel ≈ 36 次）≈ **+5-15ms/步（+0.3-1%）**。向量化 bbox 用 any_row/any_col 技巧（matcher.py:299-304 同款），无 Python 逐 GT 循环。备用零成本方案：从 `_loss_masks` 已算的逐对 dice 缓存复用（损失次序 labels→masks 需对调，作为预算紧张时的降级）。
- 贴入：CPU dataloader worker 侧（transforms 在 worker 内执行，预取掩盖）。每图 ≤16 次贴入 ×（≤64² crop 的 resize+IoU 检查+写入 ≈ 0.1-0.3ms）≈ **≤5ms/图 CPU**；GPU 关键路径 **+0ms**。prefill 256 图一次性 ~1s/worker。
- 匹配器连带：tiny GT ×3 → AIM 列计算点数 +~4M/步（200 query × ~256 点 × +8 GT × 9 层），为现行 AIM 成本的 +17%，匹配器占步时 ~3-5% → **+0.5-1%**。
- mask 损失连带：matched 对 +4-16/216 → 点采样 +2-7% × 其 ~5-10% 份额 → **+0.2-0.7%**。
- **合计 ≈ +1~3% ms/it（+15-48ms），远低于 +20% 红线。**

**推理**：**+0ms**。MAL-Q 是损失侧；贴入是训练期数据增广。打分链路（gpu_postprocess.py / arch.py:1509-1570）零改动。

**显存**：GPU 激活 +≤10MB（q 张量 216×~2500×4B 瞬时）；CPU bank 2000 条 × ~8KB（中位 crop 22²×3 uint8 + 22²float32 深度）≈ 16MB/worker × 4 worker ≈ 64MB RAM。24GB/卡红线无压力。

**GPU 侧合规**：q 在 criterion 内对 GPU 张量计算，无 CPU 往返；贴入留在 worker 侧是标准例外（crop 为 numpy，搬 GPU 的传输时延超过计算收益，符合场规则允许条款）。

---

## 5. (e) 正确性验证方案

### 5.1 单测（不占卡）

1. `test_mal_q_matches_aim_dice`：合成 pred/GT 对，新 helper `_match_quality` 的 q 与 matcher `_aim_small_gt_costs`（matcher.py:341-342）对同一 (query, GT) 对的 dice 列值逐位一致（同一网格构造 ⇒ 公共 util 重构后容差 0）。
2. `test_mal_target_detached_offgrad`：`q.requires_grad == False`；backward 后 logits.grad 非空、GT mask grad 为 None。
3. `test_mal_disabled_bitwise_baseline`：`mal_enabled=False` 时 `loss_ce` 与现行实现逐位相等（开关回归门）。
4. `test_paste_topup_budget`：可见小目标 < 目标数时补齐 draw 数 = clamp(target−n_small, 0, max_paste)；已 ≥ 目标数时只做 base draw；masks/boxes/labels 计数一致。
5. `test_paste_depth_consistency`：贴入后区域深度来自 bank crop（write_mask 语义），无效源深度保留目标值（transforms.py:1134-1141 行为锁定）。
6. `test_bank_depth_filter`：对比度 < 1.5σ_raw（raw |Δ| < 0.015，σ_raw=0.01）的实例不入库；预期保留 ~50% 的 56K 候选池 ≈ 28K ≫ 2000 容量。

### 5.2 冻结基座探针（封存 F1 权重，仅前向，~30min GPU 或 CPU 抽样）

对 val 全部 <64 GT 的 matched keeper 计算 q：报告 **q 分布 p25/p50/p75**、**Spearman(q, matched IoU)**、Spearman(q, 最终分)。
- **门：Spearman(q, IoU) ≥ 0.5** → q 是有效质量信号，MAL-Q 有靶可学；< 0.4 则 q 换成网格 soft-Dice 的 EMA 平滑版再测一次，仍败则本提案降级（MAL 因子否决，仅保留贴入臂）。
- **τ 标定**：取 q̃ 中位 ≈ 0.55-0.65（对应期望 cls 中位较现状 ~0.48 上移但不越过可达质量），反解 τ ∈ {1.0, 0.75}。

### 5.3 短训门（10-15K 单因子开关，封存 128K 基座续训，固定 seed，四臂）

A0 对照（什么都不开）/ A1 仅贴入 / A2 仅 MAL-Q / A3 贴入+MAL-Q（主张臂）。判读数字：

| 指标（口径） | 基线 | 10K 门 | 15K 确认门 |
|---|---|---|---|
| <64 TP 分数中位（best-IoU det，全 val，非前 1200 图子集） | 0.29 | A3 ≥ 0.36 | **A3 ≥ 0.40**；A2 方向为正；A1 若 >0.35 必须同查 FP 门（DEIM 警告） |
| 固定池 FP 率（<64 det 池 hard-FP 份额；及 iso-recall 下 FP/图） | 80.5% | 不升 | **A3 ≤ 78% 且 iso-recall（recall@0.5_small ±0.005）FP/图 ≤ A0 × 1.10** |
| AP_s（trainer 口径） | 0.2638 | A3−A0 ≥ +1.0pt | 对 Phase B 退火对照线净增 ≥ +1.5pt（SYNTHESIS Phase D 判据） |
| recall@0.3（<64） | 0.113 | A1/A3 ≥ 0.13 | ≥ 0.15 |
| 贴入区边缘 FP（探针批，贴入 bbox 2px 环内 FP 计数） | — | ≤ A0 + 0.5/图 | 同左（上下文失配监控，InstaBoost/SOC 教训） |
| q 健康（训练遥测） | — | q 中位不塌（≥ 探针值 −0.1） | keeper/unmatched 分数分布间距拉大 |

**消融判读**：A3 > max(A1, A2) + 0.5pt（配对性成立）；A1 出现 FP 通胀 → max_paste 16→12、iou_threshold 0.7→0.8 重跑；A2 TP 中位反降或 AUC 降 → τ/β 网格 {0.75,1.0}×{1.5,2} 一轮，仍败否决 MAL 因子保留贴入臂（此时贴入臂必须补 MAL 的替代纪律——收紧为"贴入实例只进 mask/dice 损失、cls 目标仍 one-hot"的降级模式并重验）。

---

## 6. (f) 与其它场胜出方案的叠加兼容性

| 场 | 交互 | 兼容性 |
|---|---|---|
| **P3-a 边界监督** | q=soft-Dice 对边界敏感；P3-a 改善掩码 → q 上升 → MAL-Q 训出的分数自动跟上 | **协同**：损失项不相交（P3-a 动 mask/dice，本提案动 cls 目标+数据），且 P3-a 的收益被 MAL-Q 自动"变现"成分数 |
| **P3-b 小掩码表示/解码** | 导出/表示侧，训练目标正交 | 兼容；P3-b 抬高小掩码可达质量上限 → q 上限同步抬高 |
| **P3-c 边界稳定性/退火** | 日程类改动 | 正交；本提案短训门已内置 Phase B 退火对照线作归因基准 |
| **P4-a 匹配稳定性/o2m** | Group-DETR/H-DETR 的 o2m 需要质量感知损失配套——**本提案预装该配对件**，`_match_quality` helper 可直接复用于 o2m 支路正样本的 q；贴入增多的 GT 也喂给 o2m 组 | **按构造兼容**（场规则的打包要求由本提案替 P4-a 预满足） |
| **P4-b query 预算/种子** | R2b 已证"覆盖 92% 但分数 0.22-0.29"——缺的正是分数绑定；MAL-Q 即该捆绑的损失侧交付物；种子 query 出生后由同一 cls 头按 q 成型 | 兼容且互补；执行序上 P4-b 依赖本场的分数绑定结论，本提案即其输入 |
| **P4-d 导出/分数融合** | 若 P4-d 改融合公式（去 maskness 因子/尺度归一）：MAL-Q 训出的 cls 已自带质量信息，cls-only 导出变得可行；若保留 cls×maskness，τ 按探针标定避免双重计入（§1.3） | 兼容；建议 P4-d 胜出方案在 MAL-Q checkpoint 上重扫融合超参一次 |

---

## 7. 实施清单（文件与函数级接入点）

| 改动 | 位置 | 量级 |
|---|---|---|
| `_match_quality(outputs, targets, indices)` helper：AIM 网格确定性 soft-Dice，no_grad、向量化 | `models/common/criterion.py` 新增（网格构造与 `matcher.py:298-320` 重构为公共 util `models/common/aim_grid.py`） | ~60 行 |
| `_loss_labels` 软目标分支：matched 位置 scatter q̃（替换 1），QFL 正支 `|σ(z)−q̃|^β·BCE`，负支原样；warmup 线性混合 1→q̃（前 2K 步） | `models/common/criterion.py:390-421` | ~30 行 |
| `SetCriterion.__init__` 新参：`mal_enabled/mal_beta/mal_tau/mal_min_q/mal_warmup_steps` | `criterion.py:132`；构造点 `arch.py:139` 与 `_sync_criterion_from_config`（`arch.py:723`，在 no_object_weight 之后传参） | ~10 行 |
| 配置键 `mask_former.mal_*` | `config/schema.py` MaskFormerConfig（no_object_weight 行 277 附近） | 5 键 |
| 分层贴入预算：`__call__` 内数可见小目标→决定 draw 数；`tiny_bias` 抽样偏置 | `data/transforms.py:1007-1040`（CopyPasteTransform.__call__） | ~15 行 |
| 入库质量过滤：deposit 时算 in-mask vs 环带深度对比度，< 1.5σ_raw 拒收 | `data/dataset.py:207`（_InstanceBank.deposit，depth 形参已在） | ~20 行 |
| 配置键 `data.copy_paste.*`：enabled/prob 1.0/max_paste_instances 16/bank_capacity 2000/prefill_images 256/small_topup_target 6/tiny_bias 0.5/bank_min_depth_contrast 0.015 | `config/schema.py:71`（CopyPasteConfig）+ `transforms.py:1298`（Compose 装配） | 4 新键 |
| 单测 6 项（§5.1） | `tests/` | ~150 行 |

**执行序**：单测 + 冻结探针（τ 标定，零训练风险）→ A2/A3 短训门（10-15K，四臂）→ 过门后 32-48K 激活臂（与 Phase B 对照线比净增），期间把 MAL-Q 的 q helper 交付 P4-a 复用、分数绑定结论交付 P4-b。

## 8. 风险与自限

1. **q 与 maskness 相关性过高**（双重计入）：τ<1 + 探针标定 + P4-d 联合重扫兜底（§1.3、§6）。
2. **贴入上下文失配 FP**：边缘 FP 探针监控 + iou_threshold 收紧档位；文献锚 InstaBoost/SOC（调研C 反面教训 4）。
3. **深度入库过滤偏置训练分布**：只影响贴入增量（bank 是额外供给），真实例全量保留；过滤阈值取分布 p50（1.5σ_raw）而非更狠，保留 ~28K 池。
4. **MAL 早期塌缩**（q 小时目标近 0）：q_floor 0.25 + 2K 步 warmup 线性混合 + EMA 选项。
5. **均衡点未移动**（A2 无效）：探针可提前识别；降级路径明确（§5.3 消融判读），不会浪费已建的贴入臂。

**判分自评**：收益 +2~5pt（文献+内部双锚）× 置信度中高（DEIM 同构配对 + 排除法论证充分）÷ 开销 +1-3% 训练 / 0ms 推理——本场的"收益÷开销"最优形态之一。
