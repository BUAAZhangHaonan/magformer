# P4-c 提案 #3：「MAL-CP」——σ-Dice 质量感知分类目标 × 尺度分层 copy-paste 增密（单 commit 打包）

提案人 #3/5 ｜ 2026-09-20 ｜ 场：P4-c 曝光与分数成型（训练期让小目标分数成型的最短路径）

---

## 0. 一句话机制与方案总览

**机制一句话**：把 AIM 已经放进匹配 cost 的尺度无关 soft-Dice **延伸进分类损失目标**（matched query 的 focal 目标从 one-hot 1 改为它自己的 σ-Dice 质量标量 q，即 DEIM-MAL / Stable-DINO 位置质量绑定的 mask 版），同时用**已修复的深度同步 copy-paste 通道 + tiny 桶分层抽样**把 <64px² GT 的每步可见曝光从 0.85 个提到 ~10 个（×10）——损失目标解决"分数学什么"，贴入增密解决"分数拿什么学"，二者互为必要条件（DEIM 教训：增密不配质量感知损失=分数通胀），故**单 commit 打包、不可拆分上线**。

三个组件（一个核心损失改动 + 一组数据参数 + 一张 Phase-2 扩展卡）：

| 组件 | 内容 | 改动面 | 出处谱系 |
|---|---|---|---|
| C1 质量感知分类目标（核心） | matched query 的 cls 目标 = 该层自身预测的 soft-Dice q（小 GT 直接复用 AIM 匹配时已算出的 dice cost，零额外计算）；正通道损失换 QFL/MAL 形式；负样本项（focal α=0.25、eos 0.1）不动 | `SetCriterion._loss_labels` 约 25 行 + matcher 返回值扩展 | DEIM-MAL (CVPR25)、VarifocalNet/QFL、Stable-DINO（调研C 候选 1；调研B 候选 3 明言"AIM 已走前一半，缺的正是后一半"） |
| C2 copy-paste 分层增密（供给） | bank 2000 + tiny(<64px²) 独立抽样桶（draw 权重 tiny 0.60 / small 0.25 / any 0.15）+ max_paste 12 + 入库深度可分性过滤（SNR>1.0σ）+ 大 crop 入库上限 + prefill 断点修复 | `CopyPasteTransform` / `_InstanceBank` / `CopyPasteConfig` 参数与小改 | Ghiasi CVPR21、Kisantal、AD-Det 按需贴入、X-Paste 质量过滤（调研C 候选 4） |
| C3 尺度分桶 focal 调制（Phase-2 扩展卡，不进首包） | EFL 式按尺度桶独立 γ/调制，针对 80.5% hard-FP 池的 Seesaw 补偿项 | 损失内 ~20 行 | EFL (CVPR22)、Seesaw（调研C 候选 2） |

---

## 1. (a) 机制因果链——为什么攻击根因

### 1.1 小目标分数低的根因不是一个，是一条三环链，本提案同时断其中两环

**环 1（根因，供给饥饿）**：batch=4 每步仅 0.85 个可见 <64 GT（R5 §1：仅 16.2% 图含 <64 目标，<64 像素份额 0.0007%，损失点份额 0.1-0.4%）。分类正梯度里 <64 桶的质量份额 ~0.5%。88K 步分数持平（24K→112K 斜率 −0.003/8K，R1 §2）不是"没成熟"，是**均衡点**：one-hot 目标 1 与弱特征证据（masked-pooling 像素少，R1 §4）在 0.48 附近僵住。**喂不饱的 focal 学不出区分性分数**——这是数据侧根因，任何损失形式都替代不了样本量。

**环 2（根因，目标语义错误）**：当前 matched query 的分类目标恒为 1（`criterion.py:400-411` one-hot），与匹配质量无关。于是：
- 同一个小 GT 上抢位的 ~14 个 query（R1 §3：每命中 tiny GT 平均 14.05 个 IoU≥0.5 det，70% ≥2 个），o2o 下只有 1 个拿到目标 1，其余 13 个拿 eos 目标 0——但它们与 keeper 的 masked-pooling 特征几乎相同，损失在"几乎相同的特征"上要求 1 vs 0 的输出差，**分类头只能折中，折中就是 AUC 0.49-0.75 的纠缠**（<64 det 池 hard-FP 80.5%）。
- one-hot 目标还迫使分类头对"只画得出 0.65 Dice 的小掩码"的 query 输出 1.0——一个特征证据永远到不了的目标，正是 88K 步僵持的第二个来源。

**环 3（症状放大器，非本场职责）**：score = cls × maskness 对小掩码结构性压分（matched p50 0.613/0.723/0.821 随桶单调，小掩码 maskness 上限 0.85-0.9）——归 P4-d 处理；本提案使其中的 cls 因子可分、可信。

### 1.2 C1 如何断环 2：把 R1 证明的"AP 最优排序"直接训练进 cls

R1 §1 的反事实给出了关键结论：AP 最优排序不是"按质量排"（score=IoU −11.9pt，高 IoU 重复 det 挤到唯一 TP 前面），而是 **"每 GT 一个代表、代表按质量排、冗余沉底"**（`a_plus_dedup_ranked` +1.33pt，AR_s 锚 +1.61pt）。这个排序**无法事后构造**（已两度证伪），但**可以训练期塑造**：

- **代表按质量排**：matched query 的目标从 1 改为 q = 它自己的 soft-Dice（AIM 匹配时对每列算的就是这个量，`matcher.py:_aim_small_gt_costs` 的 `dice_cols`，匹配选中格的值即 q——匹配选的就是它，分数目标就是它）。质量高的代表学到高分，0.65 Dice 的代表学到 0.65——**目标变得可达**，僵持的均衡点被移动。对画得好的小检测（q≈0.8），cls 目标 0.8 **高于**当前均衡的 ~0.48（=0.29/0.61 反推）；对画不好的（q≈0.4），目标低于现状。MAL 不是整体抬分，是**按质量重排分数**——这正是 FP 不搭车的唯一形态：FP 侧（未匹配 query，含 13 个抢位者）的 eos/focal 压低路径原封不动，正样本侧携带质量信息。
- **冗余沉底**：抢位者目标仍为 0，但现在正样本侧有了可学的分离面（keeper 目标 q>0 vs 抢位者 0），而不是 1 vs 0 的不可实现分离面。喷洒重复（14.05→目标 <9）与 hard-FP 纪律（AUC 0.75→目标 ≥0.82）由此而来。
- **与事后证伪路线的本质区别**：分位重标定/score=IoU 是对**冻结分数**的单调变换，排序不变性决定其必败且 FP 搭车；MAL 改变的是**训练信号本身**，FP 与重复的分数在训练中被同步压低——被搭车的对象在源头就被处理。

### 1.3 C2 如何断环 1：×10 的 <64 曝光，且是 DEIM Dense O2O 的合法同构

- DEIM 的 Dense O2O 用增广把每图目标数成倍增加以增密正样本；我们的 copy-paste 深度通道修复后（`transforms.py:1131-1141` 深度同步贴入已落地，`dataset.py:_InstanceBank.deposit` 已存 `crop_depth`，`prefill_images` 已实现）正是同一机制的数据侧实现：**贴入增加的是 GT 数（每 GT 仍只 1 个正样本），不是每 GT 的正样本数**——o2o 语义零污染，这与被反面教训封禁的 naive o2m（3300 query 不分组 32.6→8.4，正样本洪峰摧毁分数语义）在机制上不同类。但 DEIM 消融同时警告：**增密必须配质量感知损失，否则低质量匹配拖垮分数**——贴入实例画不好时，one-hot 目标仍教"给 1"，hard-FP 池膨胀；配 C1 后贴入实例自动按其可画性得分。这就是"必须打包"的机制依据，不是合规套话。
- 量化供给（用 R5 §5 的核算式，E[贴入/图] = prob × N × P(抽中桶) × 通过率 ~0.65）：max_paste 12、draw 权重 tiny 0.60 → E[<64 贴入/图] ≈ 0.5×12×0.60×0.65 ≈ **2.3/图** → 每步 <64 可见 GT 0.85 → **(0.85 + 4×2.3×0.163… 按均值计) ≈ 10/步（×10）**；小目标(<1024px²)整体密度 ×2-3（R5：像素份额 0.124%→0.3-0.45%）。注意 COCO AP_s 覆盖全部 <1024px² GT，<64/64-256/256-1024 三桶全都计入 AP_s，small 与 any 抽样同样有效，tiny 加权只把火力对准最差的桶。
- 贴入质量守门（反面教训 4 的对症件）：入库时深度可分性过滤（掩码内 vs 外环 |Δdepth| > 1.0σ_raw，σ_raw=0.01；R5 §3：p50 SNR 1.5 → 保留 ~60-65%，源池 56,123 → ~3.5 万，仍是 bank 2000 的 17 倍）；深度弱对比 1.5σ 的上限约束由此从"整体风险"降为"入库门槛"。

### 1.4 附带的一个代码级发现（C2 的前置修复）

`dataset.py:503-523` 的 prefill 断点条件 `if len(self._instance_bank) >= self._instance_bank.capacity: break` 中 `__len__` 返回的是 **all_bank** 长度（`dataset.py:327-329`）。all_bank 以每图 ~55 实例的速度增长，capacity 2000 时 prefill 在 ~36 张图后即断出，small_bank 仅积 ~97 条——**prefill 现状对 small/tiny 桶近乎无效**。修复：断点改为检查 small（或 tiny）桶长度达容量，上限仍受 prefill_images 约束。这是启用 C2 参数网格前必须修的一行。

---

## 2. (b) 与二分匹配 / set-prediction 的兼容性论证

1. **匹配器零改动**：AIM（尺度无关 soft-Dice cost、无 [:16] 帽、bbox+30% 确定性网格）完全不动。C1 只消费匹配结果（indices + 选中格的 dice 值），不参与代价计算、不改变匹配的确定性。匈牙利 o2o 约束原样：每个 GT 仍恰好 1 个 matched query，每个 query 至多 1 个 GT。
2. **query 集合与输出契约零改动**：200 queries、无辅助支路、无 query 复制、无 NMS。推理链路（`arch.py:_inference_raw_gpu` → `gpu_postprocess.py`：cls sigmoid → topk → mask 双线性 → 0.5 二值化 → final = cls × maskness）**逐字节不变**——C1/C2 是纯训练期改动，推理 +0ms。分数语义从"P(是物体)"迁移为"P(是物体)×(能画多好)"，仍是逐 query 标量、仍是全局排序键，set-prediction 的输出合同（集合 + 分数）不变；VFL/QFL/Stable-DINO 已在 o2o/单阶段框架内验证过该语义迁移。
3. **增密走数据不走匹配**：贴入实例以一等 GT 身份进入 masks/boxes/labels（`transforms.py:1169-1185`），匹配与损失对它们与原生 GT 一视同仁——不存在"同一 GT 多个正样本"的 o2m 污染路径；Group DETR 表 7 的崩溃模式（正样本洪峰）在结构上不可达。这正是本提案选择 copy-paste 而非 o2m 支路作为增密通道的原因；若 P4-a 的 o2m 支路后续胜出，C1 的 q 目标天然延展到支路的每个 matched query（见 §7）。
4. **深监督一致性**：criterion forward 对每层 aux 输出重新匹配（`criterion.py:250-259`），C1 的 q 逐层取该层自己的 dice cost——第 k 层的分类目标反映第 k 层的可达质量，与深监督逐层重匹配的语义自洽。
5. **与 AIM 的度量同源**：q 与匹配 cost 是同一个量（同一确定性网格上的 soft-Dice），"匹配选中的依据"与"分数回归的目标"完全一致——不会出现"匹配按 A 标准选、分数按 B 标准学"的两套语义打架。

---

## 3. (c) 理论收益上界与依据

按通道分解（互不重复计费），全部注明出处：

| 通道 | 上界 (pt AP_s) | 依据 |
|---|---|---|
| T1 排序通道（冻结 det 集上重排） | **+1.3 ~ +1.6** | R1 §1：`a_plus_dedup_ranked` 0.2925→0.3058（+1.33），AR_s 锚 +1.61；MAL 训练的正是这个排序（keeper 按 q、冗余沉底），且是在 det 集本身也在变好的同时逼近 |
| T2 存在性/topk 通道 | **+0.5 ~ +1.5** | SYNTHESIS P5：keeper 图内排名中位 85/100、36% ≥90；c0 实测 topk200 +0.55。cls 因子抬升（0.48→0.6-0.7 量级）直接把 keeper 推进 100 名内——这是训练期版的"topk 解锁" |
| T3 det-set 通道（copy-paste 增密） | **+2 ~ +5** | R5 §5（Ghiasi CVPR21：COCO +1.2 / LVIS +2.4-3.7 / tiny 线 +3-7；Kisantal 相对 +9.7%；单类无类别语境噪声、深度已同步）；我们的 tiny 分层版取区间中上沿。DEIM CVPR25（dense 增密+MAL，RT-DETRv2 53.2 AP）证明打包形态成立 |
| T4 变现率通道（为 P4-b 供给的分数流） | 不计入本提案独立收益；为后续场解除门控 | R3：oracle 补全 @oracle 分数 +37.5 vs @现实分数 +6.7——变现率被分数流门控在 18-78%；R2b 种子实验：覆盖 92.1% 而 cls 卡 0.22-0.29。分数流成熟是 P4-b 收益的乘法系数 |

**本提案独立收益主张：+3 ~ +6 pt AP_s**（trainer 口径 0.2675 → 0.30-0.33，32-48K 组合臂）＝ T1+T2+T3 的交互和（T3 的收益依赖 T1 的纪律性才能兑现为 AP 而非 FP 通胀——DEIM 消融方向；反之 T1 的质量目标依赖 T3 的曝光才能在小桶形成统计）。上限诚实声明：
- 任何 det 集上排序通道 ≤ AR_s − base（当前 +1.6 封顶），T3 抬 AR_s 本身；
- E3 检测存在性+分数山（rescore 口径 +16.5）中，纯分数可区分性份额 ≤2pt（R1），其余需覆盖（P4-b）——本提案是那座山的"分数流段"最短路径，不是整座山；
- 残余风险定价：R1 §4.8 指出小目标 cls 收缩部分来自 masked-pooling 表征（损失侧只能部分补偿）——若 C1 后小桶 cls 因子仍贴地，是转 P3 的信号，不是继续调损失的信号（已写进验证门的判读规则）。

置信度：C2 高（机制+文献+代码审计三重锚，R5 全链核算）；C1 中高（DEIM 消融方向 + Stable-DINO 同构先例 + R1 的排序上界证明目标可达；扣除 pooling 上限与 q 噪声两项不确定）。

---

## 4. (d) 开销预算（估算算式）

### 训练 ms/it（基线 1440-1600ms，红线 +20% = +288ms）

| 项 | 算式 | 估算 |
|---|---|---|
| C1 小 GT 的 q | matcher 每层已算 `dice_cols`（Q×n_small），取选中格 = 索引操作，9 层 × 4 图 | **≈0 ms**（免费搭车） |
| C1 大 GT 的 q（area≥4096px²） | N_large≈50 matched × P_q=2048 均匀点 × 9 层 = 0.92M grid_sample | +1~2 ms |
| C1 正通道 QFL 损失 | 逐元素 (4×200×2) 张量运算 | <0.5 ms |
| C2 AIM 匹配列增长 | n_small/图 均值 2.37→~5.7（tiny +2.3、small +1.0）；AIM 点数 ∝ Q(200)×n_small×R²(R∈[8,64]，均值 ~2500)：现 ~42M 点/步 → ~100M 点/步，增量 ~60M 次 bilinear 查值（~1-2 GB/s 显存带宽级），grid_sample 吞吐 ~1G 点/s | +40~80 ms（+3~5%）|
| C2 掩码损失行数增长 | matched N 54→~62（+15% 仅在 50% 贴入图上），点采样行线性 | +3~8 ms |
| **合计** | | **+45~90 ms/it ≈ +3~6%**，红线内余量 >3 倍 |

峰值注记：小目标密集图（small p90=11/图 + 12 贴入）单图 AIM 列可到 ~23，单步尖峰 +120ms——need-based 抽样权重已把均值压住；上线时在 matcher 计时打点，若 P95 超 +150ms，回退 max_paste 12→8（E[tiny/图] 仍有 1.5，×7 曝光）。

### 推理 ms

**+0 ms**。损失与数据管线改动，推理链路（含 topk、maskness 融合公式）逐字节不变。全场"推理近零开销同分优先"条款以最满分满足。

### 显存 / 内存

- GPU：+0（q 向量 (N,) float32 ≈ 250 B；AIM 增量激活受 `aim_gt_chunk=16` 分块约束，激活上界不变）；远低于 24GB/卡。
- CPU RAM（bank，每 worker 副本）：2000 条 × 平均 crop（tiny/small 桶 ~40×40、any 桶 ≤192 边）≈ 30 KB/条（RGB 3B + depth 4B + mask 1B/px）≈ **60 MB/worker × 8 workers ≈ 0.5 GB 主机内存**（新增 `bank_max_crop_edge=192` 入库上限砍掉大目标 crop，防止 any 桶被 100×100+ crop 撑爆）。
- prefill 一次性成本：填满 small 桶 2000 条需 ~740 图（2.7 small/图）× ~50 ms 原始读取 ≈ **~40 s**（数据集构造期一次，修复断点后有效）。
- 贴入计算在 DataLoader worker（CPU，8 worker 并行流水）——每图 12 次贴入 ~3-5 ms，供给能力 8×1.44s vs 需求 4×5ms，利用率 <0.2%，**不进训练关键路径、零 GPU 争用**（GPU 侧仅 q 与 matcher，均为已有 kernel 的增量调用）。

---

## 5. (e) 正确性验证方案

### 5.1 单元测试（pytest，CPU）

1. `test_mal_q_matches_aim_dice`：随机掩码对，`_match_quality` 的 q 与稠密网格 soft-Dice 差 <1e-4；q `detach` 无梯度回传到 pred_masks。
2. `test_mal_off_bitcompat`：`mal_enabled=False` 时 `loss_ce` 与现行实现逐位一致（回归锁，保证开关干净）。
3. `test_mal_loss_minimum_at_q`：固定 q=0.7，扫描 logit，损失最小点 = σ(x)=0.7；q=1 时退化为现行 focal。
4. `test_bank_three_tier_split`：混合面积 deposit 后 tiny(<64)/small(<1024)/all 三桶成员正确；`state_dict`/`load_state_dict` 含 `crop_depth` 往返一致。
5. `test_prefill_fills_small_bank`：prefill 后 **small 桶**长度达 capacity（覆盖 §1.4 断点修复）。
6. `test_paste_depth_consistency`：贴入后每个 True 掩码像素处 depth>0 或保留目标值（无 RGB-深度矛盾像素）；深度 SNR 过滤阈值行为正确。
7. `test_paste_budget_and_repro`：draw 权重统计（1000 次采样 tiny 占比 0.6±0.05）；worker state 往返后贴入结果可复现。
8. `test_matcher_untouched`：C1 开/关时匹配 indices 逐位一致（MAL 不泄漏进匹配）。

### 5.2 冻结基座探针（封存 F1 EMA 权重 + dets dump，纯 CPU，~1h）

- **P1（q 信息量门，训练前必过）**：对全 val 的 matched <64/64-256 dets 计算 soft-Dice(det, GT) 分布。判读：<64 桶 q 的 **IQR ≥ 0.15 且 p50 ∈ [0.5, 0.8]**——若 q 几乎无方差，MAL 目标退化为常数（≈one-hot），方案在源头失效，须在开训前否决。
- **P2（喷洒可分性门）**：每个被命中 tiny GT 的 keeper-q vs 重复-q 中位差 **≥0.10**——MAL 要训练的分离面必须在 q 上预先存在。
- **P3（复现锚）**：在当前 dump 上复算 R1 的 `a_plus_dedup_ranked`（+1.33）确认读数链（脚本已存在于 r1_score/）。

### 5.3 短训门（10-15K 单因子开关，封存 128K 基座 finetune，固定 seed，2×2 = {C1 on/off} × {C2 on/off}，12K/臂）

| 门 | 指标（口径固定） | 判读数字 |
|---|---|---|
| G1 分数成型 | <64 **GT 桶** TP 分数中位（best-IoU det，全 val，subbucket 口径） | 0.29 → **≥0.40 过 / ≥0.50 强**；64-256：0.62 → ≥0.68 |
| G2 FP 纪律 | <64 det 池 hard-FP 分数中位（现 0.063）与 hard-FP>0.5 计数（现 91） | 中位 ≤0.10、计数 ≤150（**不得**随 TP 上移同步上移——这是区别于"事后抬分搭车"的判据） |
| G3 可分性 | <64 det 池 AUC(TP vs hard-FP)（现 0.75；64-256 det 桶现 0.49） | ≥0.82 |
| G4 固定池 FP 率 | <64 recall@0.5 ±0.01 内锁定，每图 det 预算 100 下 <64 FP/img | 不升 |
| G5 净收益 | AP_s（trainer 口径）vs 同预算对照臂 | ≥+1.0 pt @12K |
| G6 去喷洒 | 每命中 tiny GT 的 ≥0.2 分 det 数（现 14.05） | ≤9 |
| G7 稳定性 | warmup 后 loss_ce ≤1.3× 对照；q 直方图不塌缩到 0/1 | 过 |

判读规则（预先定义，防事后合理化）：
- **G1∧G2∧G5 全过** → 进 32-48K 组合臂（Phase D 协议，对照线用 R4 退火臂），预期 +3~6。
- **G1 过、G5 不过、AUC 平** → FP 搭车签名（C2 贴入毒化或 q 目标失效）→ 检查贴入边缘 FP 计数与入库 SNR 过滤；C2 降档 max_paste 8 复测。
- **G1 不过、大桶正常** → cls 因子贴地 = pooling 表征上限（R1 §4.8）→ 移交 P3，本场止损。
- **单因子归因**：2×2 设计让 C1/C2 各自的边际可直接读出；若仅 C1+C2 联合臂过而单臂均不过，说明 DEIM 式耦合成立（增密与质量目标互为必要），仍按组合推进。

---

## 6. (f) 与其它场胜出方案的叠加兼容性

| 场 | 交互 | 兼容性 |
|---|---|---|
| P3-a 边界监督 | P3-a 改 `_loss_masks` 的点采样/边界损失 → 掩码变好 → q 抬升 → C1 的 cls 目标水涨船高。**正向耦合**：P3-a 抬 q 上限，C1 负责把 q 变成分数。代码上分属不同函数（`_loss_masks` vs `_loss_labels`），无编辑冲突；单因子归因要求两场不同臂上线 |
| P3-b 小掩码表示/解码 | 表示变好 → maskness 与 q 同时抬升 → 排序与分数双受益。若 P3-b 改渲染链（阈值/软输出），需在 P4-c 之后的 checkpoint 上重校准——无结构性冲突 |
| P3-c 退火/稳定性 | MAL 改变损失目标，**不得与退火臂混在同一 A/B**（混淆变量）；胜出顺序：P4-c 12K 开关臂先行，赢家搭载 Phase-B 收尾退火（退火同时降低 q 与 cls 的噪声，正向） |
| P4-a o2m 匹配增密 | **最强协同**：简报明令 o2m 必须与质量感知损失打包——C1 就是那个配套件（q 目标对 o2m 支路每个 matched query 天然适用，每 query 用自己的 q，重复预测自动低 q 低分，正是 MS-DETR 需要专门设计的解耦的损失侧等价物）。若 P4-a 胜出，两场共用一个 commit 语义；**共享预算风险**：o2m 支路（+10-18% 步时）与本提案（+3-6%）叠加需重审计 +20% 红线——缓解：C2 max_paste 降 8、o2m K 降 3 |
| P4-b 覆盖/query 预算 | 互补依赖：P4-b 造出小目标检测（R2b：覆盖 92.1% 而分数卡 0.22-0.29），P4-c 的成熟分数流是变现通道（oracle +37.5 vs 现实 +6.7）。P4-b 的种子 query 若胜出，其 cls 分支需要的正是可达的质量目标（C1）——R2b 协议复用时 MAL 分数绑定探针已在设计内 |
| P4-d 导出/分数融合 | 完全兼容：P4-d 在封存权重上做（零训练），本提案训练后需重跑 P4-d 校准；P4-d 的 maskness 尺度归一/解耦与本提案的 cls 质量化相乘（cls 管"能画多好"的先验、maskness 管"这次画成没"的证据），方向一致不冲突 |

冲突总账：唯一需要联合调度的是与 P4-a 的训练预算叠加（已给降档方案）；其余六场全部正交或正向耦合。

---

## 7. 接入点清单（对照真实实现，文件:函数级）

**C1（损失）**：
- `models/common/criterion.py` `SetCriterion._loss_labels`（L390-421）：matched 格的正通道目标 scatter q；正通道损失换 `|σ(x)−q'|^β · BCE(x, q')`（β=2.0）；未匹配与 eos 路径（L416-420 focal α=0.25 + eos_coef）不动。
- `models/common/criterion.py` `SetCriterion.forward`（L195-341）：`self.matcher(...)` 调用（L218、L259）改为同时取 matched-quality；新增 `_match_quality(outputs, targets, indices)` helper（小 GT：直接用 matcher 返回的选中格 dice；大 GT：P_q=2048 均匀点 soft-Dice；全程 detach）。
- `models/common/matcher.py` `HungarianMatcher.forward`（L357-）：加 `return_quality=False` 开关，把 `_aim_small_gt_costs` 的 `dice_cols`（L337-342）选中格值随 indices 返回（默认关闭，现行为不变）。
- `models/magformer/arch.py` `_sync_criterion_from_config`（L722-800）：透传新 kwargs 到 `SetCriterion.__init__`（L132-193）。
- `config/schema.py` MaskFormer 配置节：`mal_enabled=False, mal_warmup_steps=1000, mal_beta=2.0, mal_q_temperature=1.0`（q^τ，τ=0.5 为降敏感消融档）。

**C2（数据）**：
- `data/dataset.py` `_InstanceBank`：deposit（L207-278）加 tiny 桶分裂（area<64）+ 入库深度 SNR 过滤（raw σ=0.01）+ `bank_max_crop_edge` 大 crop 跳过；sample（L287-325）加 tiny 抽样权重；**prefill 断点修复（L520）**。
- `data/transforms.py` `CopyPasteTransform.__call__`（L1007-1185）：贴入循环改用三桶加权抽样结果（几何/深度路径 L1062-1164 已就绪，不动）。
- `config/schema.py` `CopyPasteConfig`（L71-86）：`tiny_threshold=64, tiny_weight(在 draw 权重中占 0.60), depth_snr_min=1.0, bank_max_crop_edge=192`；运行配置：`enabled=true, prob=0.5, max_paste_instances=12, bank_capacity=2000, prefill_images=800, scale_jitter=(0.8,1.2)`。

**C3（Phase-2 卡，仅登记）**：`_loss_labels` 负样本项按 matched GT 尺度桶给分桶 γ（EFL 形态）+ Seesaw 补偿项对 <64 桶 FP 加罚；在 C1/C2 落定后单因子上线，复用 §5.3 全部门。

---

## 8. 风险与反面教训对照表

| 反面教训（出处） | 本提案的对症 |
|---|---|
| 事后单调重标定必败、FP 搭车（R1：−13.1/−11.9） | C1 改训练信号不改推理分数；G2/G4 门显式监测 FP 不搭车 |
| naive o2m 崩溃（Group DETR 表 7：32.6→8.4） | 增密走 GT 数（copy-paste）不走正样本数；q 目标正是简报要求的打包件 |
| 增密不配质量感知损失=分数通胀（DEIM 消融） | C1+C2 单 commit，2×2 设计可归因耦合 |
| 朴素贴入的上下文失配 FP（InstaBoost/X-Paste/SOC） | 深度同步已修 + 入库 SNR 过滤 + 贴入边缘 FP 计数作为消融判据 |
| 全局 focal γ 救不了稀有桶（EFL） | 不动全局 γ；主攻目标语义；分桶 γ 留作 C3 卡 |
| Mask Scoring 式乘法重排慎投（调研C 反面 7） | C1 非"学一个乘法重排头"：q 进监督目标，推理公式不变、零头开销 |
| NorCal/FRACAL 类先验不可用（同类同分布 hard-FP） | 不用任何类/桶先验重加权推理分数；分离面来自质量信号 |
| cls 收缩部分是 pooling 表征问题（R1 §4.8） | G1 失败分支显式移交 P3，不无限调损失 |

残余风险三件：q 早期噪声（warmup 1000 步目标从 1 线性过渡到 q + detach 缓解）；贴入分布偏移（scale jitter + IoU 守门 + max_paste 上限，val AP_s 门兜底）；maskness×q 跨桶双重计数（桶内排序改善、跨桶归 P4-d，监控桶间分差不恶化）。

---

## 9. 结论

P4-c 的问题是"0.85 个样本/步 × one-hot 目标"的乘积——供给与语义双缺。本提案用两个各 ~25 行/参数级的改动同时补两端：**把 AIM 的 σ-Dice 从 cost 延伸进分类目标（匹配选的就是它，分数学的就是它），把已修复的 copy-paste 从 COCO 默认参数重设为 tiny 分层增密（×10 曝光）**，推理零开销、训练 +3-6% ms/it、GPU 显存 +0，判分公式（理论收益 × 置信度 ÷ 开销）在本场候选中处于有利位置；且它同时是 P4-a o2m 与 P4-b 覆盖两场胜出方案的指定配套件，是分数流这条战线的地基提案。
