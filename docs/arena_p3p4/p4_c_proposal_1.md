# P4-c 提案 #1：QSP —— 质量目标分类损失（MAL）× 尺度分层贴入曝光

> 提案人 #1 ｜ 2026-09-21 ｜ 场：P4-c 曝光与分数成型（训练期让小目标分数成型的最短路径）
> 一句话机制：把分类目标的"1"换成 AIM 同款尺度无关 soft-Dice 质量标量（DEIM-MAL 形态），与按 <64 桶缺额分层填充的 copy-paste 曝光同 commit 打包——前者改变小目标 query 分数收敛的**目标均衡点**，后者改变其**正样本迭代频率**，二者共同决定 0.29 这个 88K 步不动摇的均衡值。

---

## 0. 速览

| 项 | 值 |
|---|---|
| 改动核心 | `SetCriterion._loss_labels` 质量目标（~60 行）+ `CopyPasteTransform` 分层填充/深度合意贴位（~50 行）+ `_InstanceBank` tiny 子库（~40 行）+ config 键 |
| 训练开销 | +1~5 ms/it（占 1.44-1.60 s/it 的 **+0.1~0.3%**，红线 +20%） |
| 推理开销 | **0 ms，逐 bit 不变**（无新模块、无新头、打分公式不动） |
| 显存 | 峰值 +60~120 MB 瞬态（质量网格采样），远低于 24GB |
| 理论收益 | 12K 短训门 +0.8~2.0 AP_s；32-48K 激活臂 +2.5~5.0 AP_s（本场 16.5pt 缺口的 15-30%） |
| 主判据 | <64 GT 桶 TP 分数中位 0.29→≥0.38 且固定池 hard-FP 份额不升（80.5% 封顶） |

---

## 1. (a) 机制因果链：为什么这打在根因上

### 1.1 病灶的因果重构（从我们的数字推出，不是从文献搬来）

小目标 TP 分数 0.29 可精确分解：`score = cls × maskness`，matched maskness p50 按桶 0.613/0.723/0.821（SYNTHESIS P9），故 **cls 因子 p50 ≈ 0.29/0.613 = 0.47（小）vs 0.80/0.821 = 0.97（大）**（与 R1 §4 推算 0.45-0.55 一致）。所以小目标分数病灶一半在 maskness（P3 场），一半在 cls——本场打 cls。

cls 为什么停在 0.47 且 24K→112K 斜率 −0.003/8K 持平（R1 §2）？这不是"还没成熟"，是**均衡点**。因果链三层：

1. **目标不可达**：`_loss_labels`（criterion.py:390-421）对 matched query 的目标是 one-hot 1。但 query 的 cls 由 masked-pooling 特征产生（M2F 语义），小目标池化像素少 + 掩码糊（matched IoU p50 0.635-0.737，R2），特征歧义度决定它**到不了 1**；focal 的 (1−p)^γ 因子随 p 升高梯度衰减，p 在歧义上限处失速。one-hot 目标下的失速位 ≈ 该 query 特征可支撑的水平 ≈ 0.47。
2. **正样本迭代频率饥饿**：单 query 的 cls 均衡 = 它跨越图像的"被匹配（目标 1）/未匹配（eos 0.1 压 0）"频率比。锚定在小目标空间位置的 query，因 <64 GT 仅 0.273 个/图（16.2% 图含），匹配频率极低 → 负迭代淹没正迭代 → 均衡压低。全局看正负梯度质量 1.27:1 并不缺正（R1 §4），缺的是**小目标锚定 query 的正迭代份额**（<64 实例份额 0.5%，像素份额 0.0007%，损失点份额 0.1-0.4%，R5）。
3. **无判别轴可借**：事后抬分全灭（分位重标定 −13.1、score=IoU −11.9、matched→1 −14.85，R1 §1）——因为 <64 det 池 80.5% 是 hard-FP、AUC 仅 0.49-0.75，任何不改变"每个 query 学到什么"的单调变换都让 FP 搭车。唯一可用的判别轴是**该 query 的掩码质量**：TP 与 hard-FP 同类同尺度，但质量不同。把质量绑进 cls 的训练目标，是让分数携带判别信息的唯一训练期通道。

### 1.2 两个组件分别打均衡点的两个输入

- **Q-target（MAL 形态）**：matched query 的分类目标从 1 换为 q（该 query 本步掩码与 GT 的尺度无关 soft-Dice，detach）。目标变为**可达**（小桶 matched E[q]≈0.66-0.78 > 现失速位 0.47），cls 收敛到 E[q|特征]——分数从"挣扎向 1 而失速"变为"回归自身质量"。这同时是判别轴的注入：糊掩码 query 学低分、精掩码 query 学高分， pooled PR 排序按质量重排，这正是 `a_plus_dedup_ranked`（+1.33pt，R1）那种"代表按质量排"的构造，但由模型自己学出来而非事后构造。
- **分层贴入曝光**：copy-paste 不改任何损失，直接把 <64 GT 的每步可见数从 0.85 抬到 ~10（§5 算式），把小目标锚定 query 的正迭代频率 ×9——打均衡点的分母。文献同构：DEIM 的 Dense O2O 用增广成倍加目标数，其配套结论"增密必须配 MAL，否则低质量匹配与重复预测污染分数"（调研C #14 + 反面3）——本提案把这条铁律作为打包承诺写进设计。

### 1.3 为什么是"最短路径"

Q-target 是损失函数内 ~60 行、零参数、零推理改动；贴入是已修复通道（深度已同步写、prefill 键已建，dataset.py:258/1131-1141）的参数与策略重设 + ~90 行。两者都不动塔、不动 matcher、不动解码器——是所有能改变"分数学什么"的干预里改动面最小的组合。

---

## 2. 设计与接入点（对照真实实现）

### 2.1 组件 A：Q-target 质量分类损失

**损失形式**（单前景通道，混合式，负侧逐 bit 保持现状）：

```
matched query 的前景 cell：  L = |p − q|^β · BCE(p, q)          （QFL/MAL 凸调制，β=2.0，q detached）
其余所有 cell（含 eos 通道）： L = sigmoid_focal_loss(α=0.25, γ=2)  （与现在完全一致）
query_weights[eos 行] = 0.1 不变；归一化 mean×Q 不变
warmup：前 500 步 one-hot，500-2500 步线性混合 target = (1−t)·1 + t·q（防早期 q≈0.2 的悲观目标）
```

**质量标量 q 的取法（对调研C 候选1 的关键改进）**：调研C 建议 `q = maskness_t`（本帧掩码内均值概率）。**不采用**——maskness 对小掩码结构性压分（p50 0.613 vs 0.821，正是本场的病灶因子之一），拿它当目标等于把压分烤进 cls、双重惩罚小目标。改用 **AIM 同款尺度无关 soft-Dice**：GT bbox 外扩 30%（min 3px）窗内的确定性 R×R 网格（R≤64），`q = 2·Σ(p·g)/(Σp+Σg)`，分母按物体大小归一——跨桶目标可比，且与匹配代价用的是**同一个统计量**。这补上 Stable-DINO 处方的后一半："位置度量既进 cost 又监督正样本 cls"（调研B 候选3：我们的 AIM 已走前一半，缺的正是后一半）。

**接入点**（文件均在 `experiments_archive/v317_next_stage_2026-07/source/magformer/`）：

| 位置 | 改动 |
|---|---|
| `models/common/criterion.py:390-421` `_loss_labels` | 新参数 `matched_quality`；`target_classes_onehot` 构建后，对 idx 索引处的前景 cell 以 `|p−q|^β·BCE` 覆盖 focal 值（约 25 行） |
| `models/common/criterion.py` 新增 `@torch.no_grad() _matched_quality(outputs, targets, indices)` | 逐图逐 matched 对做窗口网格 soft-Dice（复用 `matcher.py:276-354 _aim_small_gt_costs` 的网格构造模式：bbox+30% margin、`torch.linspace` 确定性网格、`point_sample` 双线性、einsum 求 Dice）。GT 侧网格与标签每步只算一次缓存（GT 不随层变），pred 侧逐层算。约 40 行 |
| `models/common/criterion.py:195-270` `forward` | 每次 `self.matcher(...)` 后（主层 218 行、aux 层 259 行）调用 `_matched_quality` 得 (N,) 张量，经 `_get_loss`（758 行）随 `scale_weights` 同路径传入 `_loss_labels` |
| `models/magformer/arch.py:723-802` `_sync_criterion_from_config` | 透传新键（`getattr` 带默认，同现有风格，约 5 行） |
| `config/schema.py:271-281` `MaskFormerConfig` | 新键：`cls_quality_target: bool=False`、`cls_quality_beta: float=2.0`、`cls_quality_warmup_iters: int=2500`、`cls_quality_min: float=0.05`、`bucketed_focal_gamma_tiny: float=0.0`（§2.3 开关，默认关） |
| warmup 步数来源 | criterion 持有 `global_step` 由 trainer 注入（或用现有 iteration 计数透传，接法同 `use_uncertainty_weighting` 的参数注入模式） |

**日志（免引擎改动）**：criterion 返回 dict 中新增 `loss_ce_qpos`（matched 项均值）与 `q_med_tiny`（<64 桶 matched q 中位，按 targets 面积分桶）——trainer 对 losses dict 的键透传是既有行为，门判读直接读曲线。

### 2.2 组件 B：尺度分层贴入曝光（已修复通道的参数与策略重设）

现状数字：默认 `prob 0.5 × max_paste 5 × P(小库抽中 0.75) × 通过率 0.65` → E[贴入小目标/图] ≈ 1.2，密度仅 ×1.13（R5 堵点3）。且 `bank._small_bank` 阈值 1024px²，抽中者以 256-1024 桶为主——**均匀 max_paste 根本喂不到 <64 桶**。

**设计（AD-Det"缺什么贴什么"的桶级化）**：

1. **tiny 子库**：`_InstanceBank`（dataset.py:101）增第三列表 `_tiny_bank`（area < 64）与配额；`sample()`（dataset.py:287-325）增 `bucket: str` 参数；`state_dict` version 2→3（旧 v2 反序列化时 tiny 库为空，向后兼容，同 v1→v2 先例）。容量 200→6000（源池 56,123，瓶颈从来不是数据，R5 §4）。
2. **分层填充**：`CopyPasteTransform.__call__`（transforms.py:1007）把 `bank.sample(max_paste)`（1034 行）换成两段：先数本图变换后可见 <64 实例 n64，从 tiny 库抽 `min(attempts, ceil((6−n64)/0.65))` 个；再从 small 库抽 4 个喂中桶；IoU 0.7 守卫、面积守卫、`synchronize_instances_after_geometry`（1187 行）全部不动。
3. **深度合意贴位（InstaBoost 的深度版，针对 1.5σ 上限约束）**：贴入位置接受准则 `|median(目标区深度) − median(crop 深度)| ≤ 2σ_norm(0.05)`，最多试 3 个候选位否则退回纯随机——只把实例贴到深度面相近处，压制上下文失配 FP（调研C 反面4：贴入 FP 侧信道）。深度贴入写本身已是修复后的实现（transforms.py:1131-1141），零改动。
4. **prefill 小目标优先**：现有 prefill（dataset.py:495-520）顺序走图。改为先用 COCO 标注的 `area` 索引筛出含 <64 实例的图（16.2%，无需解码图像），优先走这些图直至 tiny 配额满——一次性 ~1,500 图 × ~80ms ≈ 2-3 分钟启动成本，tiny 库开局即满（否则 fork 后各 worker 冷启动，prefill 键白建）。
5. **入库深度可分性过滤（X-Paste 承重组件的极简版）**：`deposit`（dataset.py:207）对 area<64 的候选算 in-mask vs 2px 环的深度差，<1.0σ（0.025 norm）不入 tiny 库——只丢弃明确与深度矛盾的 ~25%，保住供给（p50 本就 1.5σ，门槛不能高）。

**config（`data.copy_paste.*`，schema.py:71-86 已有类）**：`enabled: true, prob: 0.6, max_paste_instances: 16, tiny_target_per_image: 6(新), tiny_area_threshold: 64(新), bank_capacity: 6000, tiny_bank_quota: 2500(新), scale_jitter: (0.85, 1.25), depth_agree_sigma: 2.0(新), prefill_images: 4000`。

### 2.3 组件 C（默认关的开关）：桶级 focal γ（EFL 的尺度尾移植）

`_loss_labels` 内对 matched 且 GT area<1024 的前景项 γ 从 2 降到 `bucketed_focal_gamma_tiny`（建议 1.5）——EFL 核心结论"全局 γ 无法同时服务头尾"（调研C 反面5）的直接移植，单类下"类尾"换"尺度尾"。**默认 0=关**：MAL 的凸调制已改变正项加权，同开有双重通胀风险（调研C 候选2 自己也要求"在候选 1 之上单因子"）；仅在 G1 门 TP 中位位移 < +0.06 时作为 G4 臂启用。

### 2.4 明确不做的事

- **不引入 query 侧 o2m/辅助支路**（P4-a 的场）。本提案的增密在**数据侧**（贴入 = DEIM Dense O2O 的实例库转置版），匹配始终 o2o——Group DETR 表 7 的 32.6→8.4 崩溃模式（正样本洪峰污染 o2o 分数语义）**按构造不可能发生**：贴入 GT 是互异真值，不存在同 GT 多正样本。若 P4-a 日后胜出加 o2m 支路，本提案的 MAL 就是它法定的前置配套（DEIM 打包铁律），`_compute_dn_loss`（criterion.py:584）的 `loss_dn_ce` 同样换质量目标只需 5 行——预留接口，不预付成本。
- 不动 matcher、不动推理图、不动打分公式、不加任何推理期头。

---

## 3. (b) 与二分匹配 / set-prediction 的兼容性

1. **指派不变**：MAL 作用于匹配完成之后，只改 matched query 的回归目标。匹配代价、AIM 小 GT 列（matcher.py:276-354）、o2o Hungarian、`scale_balanced` 选项全部原样。set-prediction 的置换不变性保持：损失是给定指派下对 query 置换对称的集合函数，目标只依赖被配对双方的掩码。
2. **无代价-目标环**：q 全程 detach（cls 梯度不回流掩码头），本步内不反馈进匹配；跨步看，匹配的质量敏感性本来就来自 AIM 的 dice 列（margin 已 2.43σ 稳定），目标侧加质量不会制造 Align-DETR 式 cost/target 双敏感的振荡——我们把"质量进 cost"（AIM 已做）与"质量进目标"（本提案）保持为同一统计量的两个角色，这是 Stable-DINO 验证过的组合形态（调研B 表 2.1）。
3. **贴入不改集合语义**：贴入实例作为普通 GT 进入 targets（dataset.py:1170-1181 已是追加语义），Hungarian 在 60 vs 200 query 上照常全量运行（每图 GT max 26+16≪200，"查询容量不是堵点"SYNTHESIS 已证）。无重复 GT 监督、无分数语义污染——这正是与 o2m 支路的本质区别。
4. **eos 通道语义保持**：负样本项逐 bit 不动（α0.25/γ2/eos 0.1），单类 sigmoid 校准语义不破坏；matched 目标 q∈[0.05,1] 仍是合法 sigmoid 目标。
5. **深度监督一致**：aux 层各自重匹配（criterion.py:259），各自算各自层的 q——与现有 per-layer matching 语义对齐。
6. **单类无类别噪声**：Ghiasi 主要风险（贴入类别-语境错）在 num_classes=1 下不存在，任何贴入零件都是正确正样本（R5 §5）。

---

## 4. (c) 理论收益上界与依据

**本场切片**：E3 rescore 口径"检测存在性+分数" 16.5pt（SYNTHESIS §一阶梯表）。

**上界分解**（保守→乐观）：

| 通道 | 上界 | 依据 |
|---|---|---|
| 冻结检测集合上的纯排序收益 | ≤1.6pt | R1 §5：`a_plus_dedup_ranked` +1.33 / AR_s 锚 +1.61——**这是任何不新增检测的打分改动的物理上限**；MAL 直接效应受此封顶 |
| 贴入供给（新检测+质量） | +2~5pt | R5 §5 收益边界：Ghiasi CVPR21 +1.2 AP COCO / +2.4-3.7 LVIS / 小目标专用线 +3-7pt（调研C #24-25）；R5 密度公式 N=12-20 → ×2.2-3.1 |
| 质量绑定家族的移植系数 | 相对 +1~3.5pt 量级 | VFL +2.0 AP over FCOS+ATSS（调研C #1）、QFL 单类 ~+1（#2）、RS Loss LVIS+RFS +3.5 mask AP/稀有类 +7（#6）、Stable-DINO R50 12ep 50.4（调研B 候选3）；DEIM：MAL 是增密兑现的前提件而非加分项（#14） |
| **打包兑现估计** | **12K 门 +0.8~2.0；32-48K 臂 +2.5~5.0** | 贴入供给 × MAL 兑现率（DEIM 消融逻辑：无 MAL 的增密 = 分数通胀，有 MAL 才转 AP）；对 16.5pt 切片回收 15-30%，其余需 P4-b 覆盖（38.1% 无检测）与 P3 掩码内容 |

**为什么 cls 水平面会动（量化的水平预测）**：小桶 matched soft-Dice E[q]≈0.66-0.78（由 matched IoU p50 0.635-0.737 + soft-Dice≥IoU 的 +0.02-0.08 推出）> 现失速位 0.47 → cls 因子 0.47→~0.70 上界，TP 分数 0.29→**0.38-0.45**（maskness 不动，属 P3 场）。大桶 cls 0.97→E[q]≈0.85、分数 0.80→~0.70 是**水平下降但桶内排序改善**——COCOeval 面积档内，配到档外 GT 的检测被 ignore 而非计 FP，故大桶水平下降不伤 AP_s；反而未匹配的大 FP 分数同步下沉，让小 TP 在 pooled PR 里更靠前。这正是"训练期质量绑定"与已证伪的"事后单调抬分"的分水岭：后者 FP 搭车，前者 FP 学到的是自己的低质量分。

**曝光面（供给算式）**：E[贴入 <64/图] = 0.6 × 6 × 0.65 ≈ 2.3 → 每图 <64 可见 0.27→2.6，**每步 0.85→~10.4（×12）**；<64 正实例占 cls 正梯度份额 0.5%→4.4%（×9）；<64 像素份额 0.0007%→~0.010%（×14）。全部 small 桶密度 ×2.65（文献区间 ×2.2-3.1 内，R5 §5）。

**跨桶判别（AP 视角的关键）**：AUC(TP vs hard-FP) 现值 <64 桶 0.753 / 64-256 桶 0.490（R1 §3）。MAL 后预期 0.753→≥0.78、0.490→≥0.55——64-256 桶现在近乎随机，是最大的可改善排序面。

---

## 5. (d) 开销预算（给算式）

**训练 ms/it**（现 1.44-1.60 s/it，红线 +20% = +288-320ms）：

- Q-target 质量网格：matched 对/步 ≈ 4×(55.4+3.9) ≈ 237；每对 ≤64×64=4096 点。GT 侧每步一次（网格与标签跨层缓存）：237×4096 ≈ 1.0M 双线性采样；pred 侧 ×9 深监督层 ≈ 9.0M；合计 ~10M grid_sample 点。512² float32 画布 1MB 驻 L2，采样吞吐 ~10^9/s 量级 → 计算耗时 **<2ms**；kernel 启动 9 层×2 侧×4 图 ≈ 72 次 × ~10µs ≈ 0.7ms。**合计 +1~5ms/it（+0.1-0.3%）**。
- 桶级 γ（若启用）：逐元素 where，**≈0**。
- 贴入：DataLoader worker CPU 侧。每 session ≤12 次尝试 ×（36×36 resize ~20µs + 画布 IoU 检查 ~10µs + 区域写 ~20µs）≈ 0.6ms/图，×0.6 图占比摊薄 ~0.4ms/图——worker 预算 ~360ms/图（1.44s/it ÷ 4 worker），**完全被 GPU 步时掩蔽，ms/it 增量 0**。
- prefill：一次性启动成本 ~2-3 分钟（§2.2.4 算式），不进 ms/it。

**推理 ms**：**+0**。前向图、头、`score = cls × maskness`、topk、二值化逐 bit 不变——学到的 cls 语义变了，计算没变。

**显存**：质量函数瞬态（逐图 55×1×512×512 float32 gather ≈ 58MB，即用即释，逐图循环同 matcher 风格）→ 峰值 **+60~120MB**；无持久新参数（0 个新 Parameter）。红线 24GB，余量两个数量级。

**CPU RAM**：bank 6000 条 × ~10KB（36²×3 uint8 图 + bool 掩码 + float32 深度）≈ 62MB/worker × 8 worker ≈ **500MB**。

**GPU 侧原则**：质量计算、损失全部 GPU；贴入在 worker（CPU↔GPU 零传输；CPU 侧合规因它是数据管线既有位置且时延被掩蔽）。

---

## 6. (e) 正确性验证方案

### 6.1 单测（CPU，无需 GPU）

1. `test_q_mal_targets`：合成 outputs/targets → 断言 matched 前景 cell 的目标=注入的 q、未匹配行与 eos 通道逐 bit 等于基线实现；q.requires_grad==False（detach 生效，cls 梯度不进掩码头）；loss 有限。
2. `test_matched_quality_metric`：同形掩码 q≈1、不相交 q≈0；**尺度不变性**（同形状 2× 缩放 Δq<ε）；确定性（两次调用 bitwise 相等——无 RNG）。
3. `test_warmup_blend`：step<500 目标为 1；2500 后为 q；中间线性。
4. `test_bank_stratified`：0 个 <64 的图触发 tiny 填充至目标或尝试耗尽；IoU 守卫仍生效；贴入区深度==crop 深度（valid 处）；v2 旧 state_dict 载入 tiny 库为空不炸。
5. `test_deposit_depth_filter`：注入合成 σ，深度差 <1.0σ 的实例不入 tiny 库。
6. `test_depth_agree_placement`：构造深度平面分层场景，断言贴入落在合意区或 3 次后退回。
7. config 单测：新键 schema 校验 + `_sync_criterion_from_config` 透传断言。
8. **离线干跑（CPU，200 图）**：实测 E[贴入 <64/图]、贴入后每步可见 <64 数、贴入边缘 IoU 分布——验证 §4 供给算式（预期 ≥2.0/图），防"bank 饥饿静默 no-op"。

### 6.2 冻结基座探针（裁决后首个实验，1 次前向，判前提）

封存 F1 权重在 val 上算 matched 对的 q 分布（本提案的核心前提检查）：**判读：E[q]@<64 ≥ 0.55（预期 0.66-0.78）**。若 <0.55，说明掩码质量低到 MAL 只能改排序不能抬水平 → 提案降级为排序臂并知会 P3 场优先。这一步把"水平抬升"承诺与实测前提绑定，失败早于任何训练。

### 6.3 短训门（10-15K，单因子，固定 seed，自封存 128K 起，topk200 作底座）

| 门 | 配置 | 主判据 | 通过线 | 守门（任一破坏即停） |
|---|---|---|---|---|
| **G1** | 仅 MAL | <64 GT 桶 TP 分数中位（R1 口径，best-IoU det） | **0.29 → ≥0.38**（中枢预测 0.42） | 固定池 hard-FP 份额 ≤80.5% 且 hard-FP 分中位 ≤0.063+0.01（**FP 不搭车**——区别于已证伪事后路线的仪器）；AP@256-1k ≥ −0.3pt；val mAP ≥ −0.3pt |
| **G2** | 仅贴入（分层） | 每步可见 <64（供给仪表） | 0.85 → ≥4（预期 ~10） | 预期 TP 中位微升且 FP 同升（把 DEIM 警告做成可见数据，为打包立法）；贴入边缘 FP 计数不爆炸 |
| **G3** | MAL+贴入（打包） | TP 中位 + AP_s | TP ≥0.40；recall@0.3@<64 0.113→≥0.14；**AP_s(trainer 口径 0.2675) +≥0.8pt** | 同 G1 守门 + AUC<64 ≥0.78、AUC64-256 ≥0.55 |
| G4（条件） | G3 + 桶级 γ | 仅当 G1 TP 位移 <+0.06 时跑 | TP ≥0.40 | 同上 |

**EMA 注意**：0.9999（τ=10K）在 12K 门内刚收敛，G1/G3 判读用 raw-shadow 对照 eval（R1 堵点5 先例，trainer 已有 raw-shadow 机制）。**通过 G3 → 32-48K 激活臂（+2.5~5pt 目标），并对照 Phase B 退火线分离因果（SYNTHESIS Phase B）**。

**失败分诊**：TP 不动但 E[q] 探针通过 → 查 masked-pooling 表征（R1 §6 反面8：池化问题损失救不了，移交 P3-b）；TP 动但 FP 搭车 → β 2.0→2.5 或启用桶级 γ；AP_s 不动但 TP/AUC 动 → 排序已改善而检测集合未变，确认贴入供给仪表是否真实生效。

---

## 7. (f) 与其它场胜出方案的叠加兼容性

| 场 | 交互 | 结论 |
|---|---|---|
| **P3-a 边界监督** | P3-a 抬掩码质量 → q 抬升 → MAL 的 cls 目标**机械性跟着抬**（同一统计量的上下游）；代码零重叠（_loss_masks 采样 vs _loss_labels/质量函数） | 强协同，任意先后；若同时进实验室可同臂 |
| **P3-b 小掩码表示** | 抬 maskness 上限 → final=cls×maskness 双因子受益；q 在训练画布计算，与推理解码/阈值无关 | 正交兼容 |
| **P3-c 退火/分组** | 训练动力学层，无代码交叠；可共享激活臂 | 正交兼容 |
| **P4-a o2m/CDN** | **本提案的 MAL 即 P4-a 的法定打包件**（DEIM 铁律）；o2m 支路/DN 的 ce 换质量目标预留 5 行接口（`_compute_dn_loss` 622-647 行处）；贴入与 o2m 是两条叠加的供给通道，都汇入 MAL | 预兼容（前置件已就位） |
| **P4-b 种子/预算** | R2b 已证"覆盖 92.1% × cls 0.22-0.29"断裂——MAL 正是缺失的分数绑定半边；种子 query 出生后在同一目标下成熟；无 query 侧改动冲突 | 互为缺环，强互补 |
| **P4-d 导出/重打分** | 本提案零推理改动，P4-d 的一切离线实验对新 checkpoint 重跑即可；若 P4-d 胜出去 maskness 因子，cls=E[q] 本身仍是正确排序键（甚至更纯） | 双向兼容 |
| 已落地件 | topk200（P5 修复）作所有门的底座；AIM 去帽是贴入小 GT 被 AIM 列覆盖的前提（已修）；`small_object_sample_threshold` 点重采样自动覆盖贴入 tiny GT 的掩码监督（criterion.py:455-462 既有） | 依赖已满足 |

**唯一真实交互风险**：P4-d 若改融合公式，MAL 的水平面预测（0.38-0.45）需按新公式重标——但判据（TP 中位位移、FP 率、AUC）都是公式无关的相对量，门的定义不受影响。

---

## 8. 风险与回退

| 风险 | 概率 | 缓解/回退 |
|---|---|---|
| 大桶 cls 水平下降引发 overall mAP 回退 | 中 | 面积档 ignore 语义下不伤 AP_s（§4 论证）；G1 守门 −0.3pt；回退旋钮：大桶目标混合 `t = 0.3 + 0.7q`（λ 混合，保持桶间水平差不扩大） |
| 早期 q 低导致悲观目标 | 中 | 500-2500 步 warmup 混合 + q_min 0.05 + β 凸调制天然压低质量对权重 |
| 贴入上下文失配 FP | 中 | 深度合意贴位 + 单类无类别噪声 + 边缘 FP 仪表；最坏回退关贴入，MAL 单独成臂（G1 已单独验证） |
| bank 饥饿 / worker 分叉冷启动 | 低 | tiny-first prefill + 干跑单测 8 + 供给仪表 |
| E[q] 前提不成立（探针 <0.55） | 低-中 | §6.2 早停分诊，降级为排序臂并移交 P3 |

---

## 9. 与调研候选的关系（引用 + 改进声明）

- **调研C 候选1（DEIM-MAL）**：采纳为骨架；改进点——q 弃 maskness 改用 AIM 同款尺度无关 soft-Dice（避免把 maskness 压分烤进目标，并与 cost 侧同统计量）；补 warmup/detach/per-layer/GT 网格缓存四个工程决策；新增冻结探针前置前提检查。
- **调研C 候选4（贴入网格）**：采纳 AD-Det"缺什么贴什么"，但从"全局 max_paste 网格"改为"<64 桶缺额分层填充"（网格版喂不到 tiny 桶，§2.2 论证）；新增深度合意贴位（InstaBoost 位置思想 × 我们深度模态）与 tiny-first prefill。
- **调研C 候选2（EFL 桶级）**：降级为条件开关（G4），理由是其与 MAL 凸调制叠加的双重通胀风险。
- **调研C 候选3（RS Loss/Rank-DETR）**：不进本提案——排序语义与 MAL 目标部分重叠，且 RS 的完整移植改动面大；MAL 门若出现"TP 抬 FP 同抬"的排序边界证据，它是既定的下一棒。
- **调研B 候选3（Stable-DINO 位置质量→cls）**：本提案即其处方后一半的落地（前一半=AIM 已进 cost），引用其"用且仅用位置度量"原则作为 q 选型依据。
- **原创部分**：0.29 均衡点的因果分解（§1.1 三层链）、固定池 FP 率作为反搭车判据的门设计（§6.3）、冻结 E[q] 探针（§6.2）。

---

## 10. 工作量

代码 ~1.5 天（criterion 60 行 / transforms 50 行 / dataset 40 行 / schema+arch 15 行 / 单测 200 行）；干跑+探针 0.5 天；G1-G3 各 12K 步（4 卡，~5h/臂 @1.5s/it）。全部在预算红线内，推理零开销。
