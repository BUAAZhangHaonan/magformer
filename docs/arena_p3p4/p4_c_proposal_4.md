# P4-c 提案 #4｜QST 三件套：质量绑定分类目标 × 尺度流量再平衡 × 分层贴入

提案人：#4（独立，互不可见）｜2026-09-20｜场：P4-c 曝光与分数成型——训练期让小目标分数成型的最短路径

**路径勘误（对照真实实现）**：分类损失实际位于 `experiments_archive/v317_next_stage_2026-07/source/magformer/models/common/criterion.py`（非 models/magformer/），`_loss_labels` 在 390-421 行，与简报的行号带吻合；copy-paste 与配置键位置与简报一致。

---

## 0. 一页摘要

**机制一句话**：小目标 TP 分数 0.29 是 one-hot 分类目标 + 流量饥饿 + FP 同分布三力夹出的**均衡点**（24K 起持平 88K 步，R1 §2），本提案用一个 commit 同时换掉三个力：① 分类目标从 one-hot 1 换成 **matched-pair 尺度无关 soft-Dice 质量标量 q**（DEIM-MAL / Stable-DINO 谱系的 mask 版，统计量与 AIM 完全同源）；② 分类正样本梯度按 **GT 尺度桶反比加权**（把代码里已存在但未接线进 `_loss_labels` 的 `_compute_scale_weights` 接通，EFL 思想的单类尺度尾改造）；③ copy-paste 以**小目标分层配额**启用（bank 2000 / max_paste 16 / tiny 配额 3 / prefill 900，全为已存在 config 键 + 一个新键）。贴入增密与质量目标同 commit 是结构必然（DEIM 反面：增密无质量损失 = 分数通胀）。

**收益上界**：AP_s（标准口径 0.2925）**+3.5~7pt → 0.33-0.37**。分解：分数侧 ≤1.6pt 硬上界（R1 完美打分 +1.33）+ topk 存在性耦合 +0.5-1.5（c0 topk200 +0.55 实测锚）+ 贴入检出率 +1.5-3.5（R5 估计 +2~5 的贴入子集，与 P4-b 覆盖收益不重复计入）。

**开销**：GPU 侧 **+6-10 ms/it（+0.4-0.7%）**，推理 **+0 ms**，显存 **<10 MB**；copy-paste 全在 DataLoader worker CPU 侧（prefetch 隐藏）。红线富余约 30 倍。

**验证门**（10-15K 四臂短训，vs A0 对照）：`<64 TP 分数中位 0.29 → ≥0.42`（A3 臂）、`<64 det 池 TP/hard-FP AUC 0.753 → ≥0.80`、`hard-FP 分数中位 0.063 → ≤0.10`（搭车哨兵）、`AP_s +≥1.2pt`（A3）且 64-256 AP 不降 >0.3pt。

---

## (a) 机制因果链

### a.1 病灶定性：均衡点，不是未成熟

- `<64` 桶 TP 分数中位 **0.29**（全 val，best-IoU det 口径），24K→112K 斜率 **-0.003/8K**——88K 步持平，且 lr 仍在峰值 85%（cosine 44% 处），排除"优化衰减"解释（R1 §2）。这是当前损失目标下的**收敛值**：继续原配方训练不会动它，必须换目标。
- 事后抬分全线证伪（分位重标定 -13.1 / score=IoU -11.9 / matched→1 -14.85，R1 §1）：det<64 池 **80.5% hard-FP 且与 TP 同类同分布**，任何桶级/单调提分变换让 FP 搭车。**唯一未被证伪的抬分面是 per-query 的"可达质量"维度**——TP 的匹配质量（matched soft-Dice p50 0.613/0.723/0.821 随桶单调，P9/SYNTHESIS）与 hard-FP（无匹配或低质量匹配）在这个维度上真实可分。

### a.2 均衡点的三条腿（每个组件对一条）

**腿 1：目标错位（one-hot 要求不可达的 1.0）**。`criterion.py:412-415` 的 `sigmoid_focal_loss` 对所有 matched query 一律以 1.0 为目标，但小目标 query 的可达掩码质量 p50 仅 0.613（masked-pooling 像素少 + 渲染链结构性上限）。后果：(i) 所有 matched query 共享同一目标值 → 分类头**没有任何排序信息**（AP 是全局排序量）；(ii) 梯度持续砸在"够不着"的目标上，学到的均衡解就是小目标输出低分。把目标换成 q（该 query 本次匹配的真实 soft-Dice）后：目标可达 → 拟合方差收缩、TP 分数向 0.6 一带收敛（相对 0.29 是**上移**，因为 0.29 远低于其可达质量 0.61）；不同质量 query 得到不同目标 → 分数流直接携带排序语义。

**腿 2：流量饥饿（每步 0.85 个可见 <64 GT）**。小目标像素份额 0.124%、损失点份额 0.1-0.4%、batch=4 每步期望可见 <64 GT 0.85 个（R5 §1）。分类流上 R1 §4 已证：本数据 ~28% query 为正（55 实例/图 × 200 query），eos/alpha 的整体质量论证不成立，问题在**每对象粒度**——小目标正样本仅占 ~5% 的正梯度质量。代码里已埋好而未接线的钩子：`_compute_scale_weights`（criterion.py:356-388，`w=(mean_area/(area+1))^α` clamp [0.5,5]）已按匹配结果逐对象计算，但 `_loss_labels`（390-421 行）**接收 scale_weights 参数后从未使用**——只进了 mask/dice。接通它即把小目标分类梯度份额 ×2-4，零新基础设施。

**腿 3：曝光稀缺（数据本体）**。仅 16.2% 图含 <64 目标；copy-paste 通道深度 bug 已修复（`dataset.py:735-741` deposit 已传 depth、`transforms.py:1131-1141` 贴入同步写深度、prefill 495-523 已实现），但默认参数是 COCO 量级：bank 200 / max_paste 5 → E[贴入小目标]≈1.2/图，密度仅 ×1.13（R5 §4）。参数重设 + 小目标分层配额（AD-Det 式"缺什么贴什么"）后每图可见 small 2.37→5-7，喂给腿 1/2 的修复以原料。

### a.3 为什么三件必须同 commit（单因子的已知反面）

- **只贴入不换目标** = DEIM 反面教训 1:1 复刻（score_maturation §4.3）：正样本 ×3-4 后 one-hot BCE 学到的只是"小目标一律给分"，hard-FP 同步抬升——贴入必须与质量目标同 commit，本提案不设"贴入单飞"臂。
- **只换目标不贴入**：质量目标仍只有 0.85 个可见 <64 GT/步可拟合，q 的估计噪声不收敛，分数成型无原料。
- **只加权不换目标**（腿 2 单飞）：逐对象放大的是"把 0.61 质量的预测推向 1.0"的梯度——低质量匹配拖垮性能，正是 DEIM 消融中"无 MAL 的 dense 匹配退化"。q 目标对冲之：q 低 → 目标低 → 梯度小；加权只放大"方向正确"的梯度。
- **事后一切**：已证伪（a.1）。

### a.4 机制设计（对着我们的数字结构）

**Q1：质量标量 q 用什么**。与 AIM 完全同源的**确定性窗口 soft-Dice**：GT bbox+30% margin（min 3px）窗内 R×R 网格（R∈[8,64]），`point_sample` 双方后 `q = 2·inter/(psum+A)`，detach。选它而非 IoU/maskness 的理由：(i) 尺度无关（分母归一对象大小），与我们已裁决胜出的匹配度量同一统计（AIM margin 0.13σ→2.43σ 的验证直接继承）；(ii) 大小 GT 统一处理（大 GT 窗口 R 顶 64 上限，是 GT 的确定性下采样估计）；(iii) matcher 里 299-342 行的实现直接 factor 复用，q 与匹配 cost 的数值一致性可单测。

**Q2：损失形式**。GFL/QFL 推广（score_maturation 表 #2，单类 ~+1 AP）：`QFL(z;q) = -|q-p|^γ·BCE_soft(z,q)`，γ=2 与现 focal 一致；q=0 退化为 focal 负项、q=1 退化正项——即 torchvision `sigmoid_focal_loss` 的严格推广，`_loss_labels` 412-421 行替换为 ~12 行。eos_coef 0.1 的 unmatched 降权机制原样保留（负样本纪律不动，防 FP 抬升）。

**Q3：早期目标塌缩防护**。训练初期 q≈0.2-0.4，纯 q 目标会自我实现低分。设 `q_eff = 1 - s·(1-clamp(q, q_floor=0.15, 1))`，s 由 warmup 线性 0→1（从头训 2K it；从 128K 封存权重 resume 则 500 it——模型已成熟，q≈当前 matched Dice 分布）。

**Q4：q 会把"小目标分数必然低"写进损失吗**。会且应当——q 携带的是 per-query 排序信息而非桶间抬升；小目标 cls≈0.6、final=cls×maskness≈0.37-0.40，仍低于大目标 0.80（maskness 结构上限 0.95-0.96 vs 0.99 是 P4-d 的场）。收益机制是：(i) TP 从 0.29 向可达质量 0.61 收敛（+0.3 的"可达未达"区间）；(ii) TP 内部按质量排序 + hard-FP（质量低/无匹配）沉底；(iii) 与 P3 场正循环——掩码质量改善 → q 上移 → 分数随之上浮，而非固定压低。可选温度 `q' = q^τ (τ=0.7)` 作为 A2 臂消融，不强推。

**Q5：贴入分层配额**。`CopyPasteTransform` 新增 `tiny_area=64` 与 `min_tiny_paste=3`：采样循环先从 bank 的 area∈[16,64) 条目补足配额（<64px² 桶即边长<8px，`min_instance_area=16` 天然滤掉 16px² 以下不可贴的碎目标），再走现有 `prefer_small` 采样至 max_paste 16；`bank.sample()` 已有 `small_weight` 参数（dataset.py:287-291，默认 3.0）但 `transforms.py:1034-1037` 调用处未传——暴露为 config 键并提到 6.0。深度质量过滤（X-Paste 承重组件，SNR<1.5σ 不入库）做成开关，默认关，由贴入区环带 FP 哨兵决定是否打开——1.5σ 是 p50，一刀切会砍掉一半源池。

---

## (b) 与二分匹配 / set-prediction 的兼容性

1. **匹配器零改动**。AIM cost 不动：q 只进损失目标，不进匹配代价。Stable-DINO（ICCV23, 2304.04742）的完整处方是"位置度量进 cost + 进 cls 目标"两半——我们经由 AIM 已走了前半（soft-Dice 已在小 GT 列的 cost 里，matcher.py:426-438），本提案只补后半，恰好闭合该谱系。
2. **o2o 拓扑不变**。仍每 GT 恰一个 matched query；q 不增加正样本数（正样本供给增加走数据侧贴入，不触碰匹配结构）。无第二匹配支路、无去噪组、无 query 扩容——与"二分匹配/set-prediction 范式不得改动"的字面与精神都兼容。
3. **无反馈环**。Hungarian 在当前预测上由 cost 决定；q 从匹配**结果**导出且在 `no_grad` 下计算（detach）——不存在"压低 q 以降损失"或"匹配偏向低分 query"的回路。每层深监督各自重匹配、各自算 q（匹配不同则 q 不同），语义正确。
4. **focal→QFL 的 DETR 先例**。QFL/VFL 在 dense 检测器验证（GFL NeurIPS20 / VarifocalNet CVPR21）；DETR 系 o2o 框架内同族工作：Stable-DINO（R50 50.4/51.5）、Align-DETR（BMVC24）、DEIM-MAL（CVPR25）——全部保持二分匹配不变、只改正样本分类目标。单类 + eos 通道结构原样。
5. **与 AIM 增密类提案的打包义务**（场规重申）：本提案不含 o2m 支路；若 P4-a 胜出（Group/H-DETR 式），其支路的 cls 损失必须同样换 QFL（每组匹配各自产生 q），本提案 A1 臂的短训结果即为其前置证据。naive o2m 的 32.6→8.4 崩溃（Group DETR 表 7）在本提案框架下的解释：无质量目标的 o2m 洪峰正是"腿 1 目标错位"的极端形态。

---

## (c) 理论收益上界与依据

**上界分解（标准 COCOeval 口径，基线 AP_s 0.2925 @112K）**：

| 收益路径 | 上界 (pt) | 依据 |
|---|---|---|
| P1 纯排序（当前 det 集上） | **≤1.6** | R1 §1：`a_plus_dedup_ranked` +1.33，AR_s 锚 +1.61——任何分数训练的硬上界 |
| P2 topk 切割耦合（存在性变现） | **+0.5~1.5** | R1 堵点 2：keeper 图内排名中位 85/100、36% ≥90；c0 topk200 实测 +0.55；TP 分数上移与 topk200 部分等效 |
| P3 贴入增密的检出率 | **+1.5~3.5** | R5 §5 估 +2~5（Ghiasi CVPR21 小目标线 +3-7、LVIS +2.4-3.7；Kisantal 反复贴入相对 +9.7%）；取贴入子集并扣与 P4-b 覆盖收益的重叠 |
| **合计** | **+3.5~7 → 0.33-0.37** | P1+P2 分数侧 ≤3（保守取 1.5-2.5）+ P3 |

**逐路径的机制可达性论证**：

- P1：当前分数-质量错位是"目标错位"的直接投影（所有 matched query 同目标 → 无排序压力）。QFL 后 cls≈q=soft-Dice → 跨图全局排序按可达质量对齐，逼近 R1 构造的真上界（"每 GT 一个代表、代表按质量排、冗余沉底"）中"代表按质量排"的成分。VarifocalNet +2.0 AP / GFL ~+1 AP（单类）为量级锚。
- P2：TP cls 中位 0.29→0.5 意味着小目标 det 图内排名前移 30-40 位，把 36% 边缘 keeper 从 topk-100 切割中救回——与 c0 的 topk200 +0.55pt 同机理但零推理开销。
- P3：贴入喂的是"检出率"本体（每图可见 small 2.37→5-7、像素份额 0.124%→~0.3%），配合 q 目标不产生 FP 搭车；CP-SSOD（2312.06312）证明 copy-paste+EMA（我们 0.9999 在用）即可显著抬小目标 AP。
- 尺度再平衡（腿 2）不单列收益：它是 P1/P3 的放大器（把成型原料导向小桶），EFL 的核心结论（全局 γ 无法服务头尾，LVIS 29.2 SOTA）支持分桶调制的存在价值，但单类尺度尾无文献直接数字。

**诚实声明**：(i) 分数侧硬上界 ~1.6pt + topk 耦合，超出部分必须是检出率贡献（P3/P4-b 场）；(ii) 本提案不触及掩码内容 27.1pt（P3 场）与 query 覆盖 38.1%（P4-b 场）；0.33-0.37 距 0.768 天花板仍远，两座山结构不变；(iii) <64 桶的贴入增密受 `min_instance_area=16` 约束只覆盖 16-64px² 子集，<16px² 部分依赖 64-256 桶的流入——收益估计已按此打折。

---

## (d) 开销预算

**训练 ms/it（现 1.44-1.60 s/it，4 卡 DDP，1 img/GPU）**：

| 项 | 估算算式 | ms/it |
|---|---|---|
| q 计算（窗口网格） | 每层 matched pairs N≈55（p50 GT 50/图）：grid_sample N×R²≤55×4096≈225K 点 + bbox argmax 55×512²≈14.4M bool 扫描 ≈0.3-0.5ms；×9 深监督层 | **+3-5** |
| QFL 替换 focal | 同形逐元素（B,Q,2），差一次 pow+abs | **+<0.5** |
| scale_weights 接线 | 已有 `(N,)` tensor 的 clamp+pow，均值归一 | **~0** |
| 贴入（CPU worker 侧） | 16 次贴入 × (resize+np.where) ≈ 10-20ms/图；4-8 worker 并行 + prefetch → 主线程增量 ≈0（护栏：监控 worker busy 率，饱和则 max_paste 降档至 12） | **GPU +0** |
| **合计 GPU 侧** | | **+6-10（+0.4-0.7%）** |

红线 +20% = +288-320ms：富余约 30-50 倍。所有新增计算在 GPU（grid_sample/einsum），无 CPU↔GPU 往返。

**推理**：**+0 ms**。q 与 QFL 只存在于 loss 路径；推理走 `arch.py:_inference_raw_gpu` 不经过 criterion。目标 ~100ms 不受影响。

**显存**：q 张量 (B,N_matched)≈4×220×4B≈3.5KB；软目标与现 onehot 同形 (B,Q,2) 不变；贴入全 CPU。合计 **<10 MB**（红线 24GB）。CPU RAM：bank 2000 × 每条 (30×30×3 uint8 RGB + 30×30 bool mask + 30×30 float32 depth ≈7KB) ≈14MB/worker × 8 worker ≈ **112 MB**。

---

## (e) 正确性验证方案

### e.1 单测（CPU 可跑，无 GPU 依赖）

- **T1 QFL 退化性**：q=1 时 QFL 正项与 `torchvision.sigmoid_focal_loss` 正项逐元素一致、q=0 时负项一致（容差 1e-6）。
- **T2 q-AIM 一致性**：同一 (pred, GT) 对上，criterion 侧 `matched_quality` == 1 − `_aim_small_gt_costs` 的 dice 矩阵对应元素（同窗口 helper、同 R）。这保证"进 cost 的度量"与"进目标的度量"数值同源。
- **T3 q detach**：backward 后 `pred_masks` 不因 q 路径收梯度（q 在 no_grad 下计算）。
- **T4 加权归一化**：scale_weights 以 `sum(w·x)/sum(w)` 形式接入，均值归一——A0 与 A2 的 loss_ce 尺度可比，无隐式 LR 变化。
- **T5 贴入三通道一致性**（扩展 P2 修复测试）：贴入后 RGB/mask/depth 在贴入区逐像素断言；tiny 配额计数 = min_tiny_paste（bank 有货时）。
- **T6 bank 深度过滤开关**：SNR<1.5σ 条目不入库（若启用）。

### e.2 零训练诊断探针（门前，CPU，~30min）

对封存 F1 dets.json 的每 GT keeper det 计算 Spearman(IoU, final score)（按 GT 桶分层）——当前分数-质量错位度的基线读数，作为 MAL 后该指标上升幅度的参照（预期基线 ρ≈0.3-0.5，A1 臂后显著上升）。

### e.3 短训门（10-15K，从 128K 封存权重 resume，LR 5e-5 cosine→0，seed 固定）

| 臂 | 配置 |
|---|---|
| A0 | 原损失原 config（对照线；可与 R4 的 160K 收尾臂合并共享） |
| A1 | +QFL 质量目标（q_floor 0.15，warmup 500） |
| A2 | A1 + cls 尺度再平衡（α=0.5，w_max=4）|
| A3 | A2 + copy-paste（bank 2000 / max_paste 16 / tiny 配额 3 / small_weight 6 / prefill 900）——贴入与 QFL 同 commit，无"贴入单飞"臂 |

每 5K eval（标准 COCOeval + GT 面积分桶扩展，R1 协议复用）。**判读数字（15K 时点 vs A0 差分）**：

| 指标 | 基线 | A1/A2 门 | A3 门 |
|---|---|---|---|
| `<64` TP 分数中位（best-IoU det，**全 val**） | 0.29 | ≥0.38 | **≥0.42** |
| `<64` det 池 TP/hard-FP AUC | 0.753 | ≥0.78 | ≥0.80 |
| hard-FP 分数中位（搭车哨兵） | 0.063 | ≤0.08 | ≤0.10 |
| AP_s（标准口径） | 0.2925 | +≥0.5pt | **+≥1.2pt** |
| 64-256 AP（大目标护栏） | 0.1461 | 不降>0.3pt | 不降>0.3pt |
| 贴入区 20px 环带 FP 占比（上下文失配哨兵） | A0 值 | — | ≤A0+2pt |

**通过/回滚判据**：主门 = TP 分数中位上移**且** hard-FP 分数中位涨幅 < TP 涨幅（区别于事后抬分的搭车签名——这是本提案与已证伪路线的可裁决分界）。若 hard-FP 搭车（FP 中位涨幅 ≥ TP），回滚顺序：q 温度 τ→0.7 → tiny 配额 3→1 → 贴入档位 16→12。若 64-256 AP 塌 >0.3pt，降 w_max 4→2。15K 判向通过后进 32-48K 组合臂，长程目标 AP_s +3~6。

---

## (f) 与其它场胜出方案的叠加兼容性

| 场 | 交互 | 兼容性 |
|---|---|---|
| **P3-a/b**（边界监督/掩码表示） | 正交且**互放大**：q 的数值=掩码质量的函数，P3 改善掩码 → q 上移 → cls 目标与分数随之上浮（分数-掩码正循环，非固定压低）。P3-b 若换掩码表示（DCT/软掩码），q 的窗口 Dice 接口不变（吃 pred_probs+GT mask）。P3-a 改点采样坐标不碰窗口网格 q。 | **兼容，建议叠加** |
| **P3-c**（lr 退火） | 无冲突；本提案 A0 对照臂与 R4 的 160K 收尾臂可合并共享（省一臂成本）。 | **兼容** |
| **P4-a**（o2m 增密） | 本提案的 QFL 正是其文前提的必配质量感知损失（场规：o2m 不得单飞）。若 Group/H-DETR 式胜出：每组各自匹配各自产生 q，QFL 直接适用；A1 臂结果即 P4-a 上线前置。硬互斥面：两者都改 `_loss_labels`，合并时以本提案的 QFL 为底座。 | **打包伙伴** |
| **P4-b**（query 覆盖/种子/预算） | R2b 已证覆盖 92.1% 但 cls 0.22-0.29 变现失败——本提案正是其"分数绑定"半件，P4-b 的种子/预算改动与本提案正交（出生位置 vs 损失目标）；其 MAL 探针协议的训练期落地形态即本提案。 | **互补前置** |
| **P4-d**（导出端重打分/融合） | 零训练/近零开销档，正交。两点交互：(i) MAL 后 cls 已携带质量 → P4-d 的 maskness 解耦/尺度归一更有依据；(ii) CQTR 轨迹重打分与 cls 分布无关，但其阈值需在 MAL 后重新扫（cls 分布整体移动，排序不变性保证 AP 结论可迁移）。 | **兼容** |
| **本场其它提案** | 互不可见，按机制声明：任何同样改 `_loss_labels` 目标形式的方案（RS Loss 排序项、EFL 分桶 focal 等）与本提案在代码同一位置互斥，只能短训门择优或以 QFL+分桶加权的合并形态消融；贴入参数网格类方案与本提案 a.4-Q5 完全重叠，可直接取 A3 臂参数为网格中心点。 | 择优/合并 |

---

## 附A 接入点清单（文件/函数级）

| 改动 | 位置 | 量级 |
|---|---|---|
| factor 窗口 soft-Dice 为共享 helper | `models/common/matcher.py:276-354`（`_aim_small_gt_costs` 的 299-342 段抽出为模块级 `window_soft_dice`），matcher 与 criterion 共用 | ~40 行 refactor，行为不变（T2 锚定） |
| matched-pair q 计算 + QFL 替换 | `models/common/criterion.py:_loss_labels`（390-421）：入口算 q（matched pairs only，no_grad），412-415 的 `sigmoid_focal_loss` 换 QFL 逐元素式，416-417 的 query_weights matched 位乘 scale_weights（均值归一） | ~25 行 |
| 配置键 | `config/schema.py`：`model.mask_former.cls_quality{enabled, warmup_iters, q_floor, tau}`、`cls_scale_reweight{alpha, w_max}`；接线在 `arch.py:_sync_criterion_from_config`（723-775，照 dn_enabled 的 getattr 模式） | ~20 行 |
| 贴入分层配额 | `data/transforms.py:CopyPasteTransform.__call__`（1034-1037 传 small_weight；1062 起采样循环加 tiny 配额段）；`data/dataset.py` 透传新键 | ~20 行 |
| 参数重设（纯 config） | `config/schema.py:71-86` 现有键：bank_capacity 200→2000、max_paste_instances 5→16、prefill_images 0→900（每图 ~2.7 small GT，900 图填满 2000 容量）、small_weight 3→6（新暴露）、tiny_area/min_tiny_paste（新键） | 0 行 |

## 附B 风险与回滚

1. **FP 搭车**（最大风险，已有哨兵+回滚序，见 e.3）。
2. **q 早期噪声**：warmup + q_floor 防护；resume 场景 q≈成熟分布，风险低。
3. **贴入上下文失配**（InstaBoost/SOC 反面）：单类无类别混淆 + 深度已同步；环带 FP 哨兵监控；深度过滤开关备用。
4. **DataLoader 饱和**：worker busy 率监控，max_paste 降档预案（12/8）。
5. **大目标欠拟合**：64-256 AP 护栏 + w_max 上限。

## 引用

内部：BRIEFS §6；SYNTHESIS P2/P4/P5/P10、§二/§四；R1 §1-§5（堵点 2/3/4）；R5 §1-§5；score_maturation.md §3 候选 1/2/4、§4 反面 1/3/4/5；query_existence.md §3 候选 3、§4 反面 1。
外部：DEIM CVPR25 (2412.04234)；Stable-DINO ICCV23 (2304.04742)；GFL NeurIPS20 (2006.04388)；VarifocalNet CVPR21 (2008.13367)；EFL CVPR22 (2201.02593)；Group DETR ICCV23 (2207.13085)；Ghiasi CVPR21 (2012.07177)；Kisantal (1902.07296)；AD-Det (2504.05601)；X-Paste (2212.03863)；CP-SSOD (2312.06312)；CQTR (2609.06581)；NorCal NeurIPS21 (2107.02134，反面)。
