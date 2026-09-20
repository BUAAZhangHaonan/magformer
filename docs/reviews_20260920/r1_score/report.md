# R1 得分成型与排序 审阅报告

审阅人: R1（独立审阅，得分成型与排序质量/"可得分"条件）
日期: 2026-09-20 晚；对象: F1 全量模型 112K 步 EMA 权重的最新 eval（快照 `2001_0920/dets.json` = `coco_instances_results.json`，211MB，247,798 dets / 3276 图）
脚本: 本目录 `analyze.py`（主）、`analyze2.py`（修正 dedup 语义 + 干净上界）、`analyze3.py`（排序最优上界）、`analyze4_subset_check.py`（子集偏差核验）。全部 CPU（`CUDA_VISIBLE_DEVICES=`），pycocotools 标准 COCOeval，dets 只整读一次、单进程。
产物: `results.json` / `results2.json` / `results3.json`。

## 结论摘要（≤10 行）

1. **打分/排序不是当前 AP_s 的主要堵点**：对现有检测集合做"完美打分"（keeper 按 IoU 排序 + oracle 去重，`a_plus_dedup_ranked`），AP_s 仅 0.2925→0.3058（**+1.3pt**）；召回上限锚点 AR_s=0.3086，即模型已兑现自身检测集合上限的 **95.5%**。
2. **分数成熟度已停滞（关键疑点证伪）**：<64 与 64-256 桶 TP 中位分在 24K→112K 共 88K 步内**持平**（全历史斜率 −0.003 与 +0.001/8K，无增长）；仅 256-1k 桶仍 +0.39pt/8K（外推 256K ≈0.87）。全 val 的 <64 TP 中位分实为 **0.29**（best-IoU det）/0.61（best-score det）——主进程引用的 0.41 是前 1200 图子集读数，系统性偏高 ~0.10（趋势有效、水平偏乐观）。
3. **一切"把小目标分数抬上去"的后处理都被证伪**：(c) 尺度分位重标定 AP_s **−13.1pt**；只对 matched 小 det 做的 oracle 版校准也 **−2.9pt**；score=IoU 的"完美排序" **−11.9pt**。小目标 det 池被 FP 主导（det 面积 <64 桶 80.5% 是 hard-FP，matched 仅 9.5%），低分是诚实校准。
4. **AP_s 缺口归因（标准评估，总缺口 0.768−0.2925=47.6pt）**：打分/排序/去重合计 **≤1.6pt（~3%）**；缺检（IoU≥0.5 无 det）≈ **29pt（~62%）**；高 IoU mask 衰减 ≈ **17pt（~36%）**——与 c0 时代 EVIDENCE §4 的 59/39/≤1.7% 结构一致，F1 没有改变缺口结构，只是整体抬高了水平。
5. **打分与召回的耦合点是 topk-100 切割**：小 GT 的代表 det 在其图内排名中位 85/100（64-256 桶，36% 排在 ≥90），切割发生在 cls 分数上（200 query 取前 100，先于 mask-score 融合）——低 cls 直接吞掉小目标检测的存在性；E4a 在 c0 上 topk200 实测 +0.55pt。
6. **训练侧"可得分"条件的代码事实**：cls focal 完全是 M2F 默认（alpha 0.25/gamma 2/eos 0.1，从未按"1 类 + ~28% query 正率"重标定）；`balanced_ce:true` 只作用于 mask 点采样 BCE，不碰分类；score=cls×mean_prob_in_mask 对 tiny mask 有 ~0.85-0.9 的结构性上限（大目标 0.99）。EMA(0.9999) 评估滞后当前 ≈0.27pt AP_s，随 cosine LR→0 在 256K 消失。
7. **协议注记**：trainer 内部评估器 AP_s 比标准 COCOeval 系统性低 2.87pt（c0 时 2.82pt，稳定偏移；本文所有反事实用标准评估器，内部自洽）；AP_s 趋势（+0.42pt/8K 全程 → +0.21pt/8K 近 5 窗，104K→112K 仅 +0.03pt）线性外推 256K 落在 **0.33-0.38**（标准评估口径），距 stride-2 天花板 0.768 仍远，且增量由召回而非分数贡献。

## 堵点表（按严重度=预计 AP_s 影响排序）

| 排名 | 堵点 | 证据(数字) | 根因 | 预计AP_s影响(pt) | 修复方向 | 验证成本 |
|---|---|---|---|---|---|---|
| 1 | （跨维度，移交 R-recall/R-mask）缺检与高 IoU mask 质量占缺口 96% | 完美打分仅 +1.3pt；R@0.5_small=0.619 vs oracle≈0.999；R@0.95_small=0.003；AP 缺口分解 缺检≈29pt+mask≈17pt | 检测集合本身缺小目标 + 小 mask 高阈值崩 | 46（相对 0.768 天花板） | 见各自维度（query 容量/stride-2 mask 表征训练充分性） | — |
| 2 | topk-100 输出切割 × 小目标低 cls 的耦合（分数压制召回） | 64-256 keeper 图内排名中位 85/100、36% ≥90；<64 桶 23% ≥90；dets/img p90 已顶到 100 上限；E4a(c0) topk200=+0.55pt | `inference_topk=100 < num_object_queries=200`，按未成熟 cls 切割在 mask-score 融合之前 | +0.5~1.5 | 推理 topk 提到 200/或按尺度配额；训练侧 focal alpha 0.25→0.5 与小目标正样本加权促 cls 早期成型 | 低：topk200 一次重推理 eval（best.pt 上 ~1h GPU，需等 05:00 停训后）；focal 臂需 8-16K finetune（1-2 天） |
| 3 | 小桶 det 池分数对 TP/FP 不可分 + 喷洒式重复 | det<64 桶 hard-FP 占 80.5%、matched 9.5%，分数 AUC 0.75；det64-256 桶 AUC 0.49（随机）；每个被命中小 GT 平均 **14.05 个** IoU≥0.5 的不同 det（70% ≥2 个）；全局 oracle 去重仅 +0.26pt | 多 query 抢占同一小目标且互不自信；FP 主导池中"低分=诚实"；重复本身不伤 AP（排在 keeper 之下）但使任何整体提分方案把 FP 一起抬上去 | 直接 ≤0.3（去重）；间接为堵点 2 的成因 | 查询竞争/去重正则（如 IoU-aware 排序损失）；**不要做后处理提分** | 中：需训练臂；后处理路线已被本报告证伪（零成本负结果） |
| 4 | mask-prod 分数融合的尺度结构性压分 | score=cls×mean(mask_prob∈mask)；EVIDENCE §5：12-20px 理想 mask 上限 0.95-0.96 vs 大目标 0.99（stride-4）；stride-2 下 tiny 仍 ~0.85-0.9 | tiny mask 几乎全是边界，均值概率天然低一档 | ≤0.5（直接；`a_separate` 显示 TP/FP 完全分离也只 +0.5） | 分数融合去掉 mask 因子或尺度归一化（需重推理验证，勿离线改 dump 分数） | 低-中：一次重推理 eval |
| 5 | EMA 0.9999 评估滞后 | τ=10K 步≈1.25 个 eval 窗口；APs 斜率 +0.21pt/8K（近 5 窗）→ 滞后 ≈0.27pt；AUD-3 先例：欠训练模型上 EMA 滞后可达十几 pt | 评估/best.pt 都用 EMA shadow；仍在上升的量被平均到过去 | ~0.3（当前）→0（256K，LR→0） | 若 128K 提前封盘：补一次 raw-weights 对照 eval（trainer 已有 raw-shadow 先例） | 低：单次 eval |
| 6 | best.pt 按 val/mAP 选择（非 AP_s） | trainer `_finalize_eval_result`，early_stop monitor=val/mAP | mAP 被中大目标主导，AP_s 更优但 mAP 略低的 checkpoint 会被跳过 | ~0-0.5（存档选择噪声） | 封盘时并列存 AP_s-best checkpoint | 零（改存档逻辑） |

## 详细分析

### 0. 基线复算与协议锚定（方法→数据→数字）

- 用 `pycocotools` 标准 COCOeval（segm，maxDets=[1,10,100]）+ 自定义面积档（GT 面积 <64 / 64-256 / 256-1024，与 `subbucket_scores.py` 的 GT 面积分桶一致）在 112K EMA dets 上复算：
  - **AP=0.8692，AP_s=0.2925**（与 panel.json 完全一致）；分桶 AP：**<64 = 0.0298，64-256 = 0.1461，256-1k = 0.3697**；AR_s=0.3086（分桶 AR@100：0.0565 / 0.1750 / 0.3831）。
  - trainer metrics_log 同一 eval 报 AP_s=0.2638：内部评估器合同与标准 COCOeval 有 **+2.87pt 的稳定系统性偏移**（c0 时代同向 2.82pt，EVIDENCE §2 已记录"合同差异"）。本文所有反事实用标准评估器；跨步趋势用 trainer 口径做差分（偏移在差分中消去）。
- 子集偏差核验（`analyze4_subset_check.py`）：我独立复现了 subbucket.json 的全部三个中位数（0.3938/0.6202/0.7957，逐位一致），证明读数链一致；但同一统计量在**全 val** 上是 <64: **0.289**（best-IoU det）/0.613（best-score det）。差异全部来自"前 1200 图"子集的小目标更容易。结论：跨步趋势有效，<64 的绝对水平比主进程引用的 0.41 低 ~0.10。

### 1. 反事实重打分实验（必做 1）

匹配定义：单类别，det 与任一 GT 的最大 segm IoU；"matched" = maxIoU≥0.5。所有变体只改分数、不动 mask、不删 det（去重变体把被杀 det 置 −1 排到全序列尾部）。

| 变体 | 构造 | AP_s | Δ vs base | AP<64 | AP64-256 | AP256-1k |
|---|---|---|---|---|---|---|
| base | 原始分数 | 0.2925 | — | 0.0298 | 0.1461 | 0.3697 |
| (a) `a_matched_to_1` | matched→1.0 | 0.1440 | **−14.85** | 0.0118 | 0.0519 | 0.2253 |
| (b) `b_score_iou` | matched→IoU（未匹配保留原分） | 0.1709 | −12.16 | 0.0165 | 0.0694 | 0.2480 |
| (b′) `b_perfect` | matched→IoU，未匹配→0 | 0.1731 | −11.94 | 0.0167 | 0.0709 | 0.2526 |
| (c) `c_recalib_lt64` | det<64 桶分位映射到 256-1k 分布 | 0.1613 | **−13.12** | 0.0229 | 0.0436 | 0.3514 |
| (c′) `c_recalib_all` | 三个小桶都映射 | 0.1625 | −13.00 | 0.0241 | 0.0962 | 0.3082 |
| (c″) `c_matched_only` | 只对 matched 的 det<64 做 oracle 校准 | 0.2639 | −2.86 | 0.0368 | 0.1056 | 0.3693 |
| `dedup_fixed` | 每 GT 留最高分 det | 0.2951 | +0.26 | 0.0437 | 0.1544 | 0.3720 |
| `a_separate` | matched 整体抬到 FPs 之上（保序） | 0.2977 | +0.52 | 0.0433 | 0.1550 | 0.3731 |
| `a_plus_dedup_ranked` | keeper→1+IoU·1e-4，冗余 matched→−1 | **0.3058** | **+1.33** | 0.0486 | 0.1725 | 0.3830 |
| （锚点）AR_s | 重打分的召回上限 | 0.3086 | +1.61 | 0.0565 | 0.1750 | 0.3831 |

（注：`analyze.py` 里的 `dedup`/`a_plus_dedup` 有一处语义 bug——某 det 是 GT_A 的 keeper 却因不是 GT_B 的 keeper 被杀；`analyze2.py` 已修正为"至少是一个 GT 的 keeper 即存活"，上表用修正版。）

**判读**：
- **(a) 崩塌的机理**：把所有 matched 置 1.0 后，重复 det 与唯一 TP 同分顶置。每个被命中的 <64 GT 平均挂 **14.05 个** IoU≥0.5 的不同 det，小桶顶部精度≈1/14；(a) 的 AP_s 暴跌 14.9pt 恰好量化了"重复质量 + 完美分数"的自毁程度——**也反证当前模型的分数排序已经把重复压在 keeper 之下**（否则 base 不可能比 (a) 高 14.9pt）。
- **(b)/(b′) 崩塌的机理**：按 IoU（质量）排序会把"高 IoU 的重复 det"排到"低 IoU 的唯一 TP"前面；单类 COCO 的 PR 曲线是全 det 池化排序，高 IoU 重复在前会在每个高阈值处制造 FP-before-TP。**AP 最优排序不是"按质量排"，而是"每 GT 一个代表、代表按质量排、冗余沉底"**——这正是 `a_plus_dedup_ranked` 的构造，也是它（而非天真的 (a)/(b)）才是真上界。
- **(c) 全线负收益**：det<64 桶 80.5% 是 hard-FP（分数中位 0.063），把整桶分位映射到 256-1k 分布等于把 ~13.7K 个 FP 抬到 0.9+；即便 oracle 只校准 matched 的（c″），抬上去的低 IoU TP 也会在高 IoU 阈值处变成排在真高 IoU TP 前面的 FP，AP64-256 从 0.146 掉到 0.106。**尺度条件校准（无论盲版还是 oracle 版）拿不回任何 AP_s，反而倒贴。**低分是诚实的。
- **真正的重打分上界 = +1.3~1.6pt**（`a_plus_dedup_ranked` 0.3058，AR_s 锚 0.3086）。base 已达上限的 95.5%。

### 2. 分数成熟度曲线（必做 2）

数据：12 个 eval_snapshots 的 subbucket.json（快照↔步数经 dets mtime 与 metrics_log val wall_time 精确对齐，每快照滞后 val 行 5-9 分钟）。TP 中位分（前 1200 图子集，best-IoU det 口径）：

| step | <64 | 64-256 | 256-1k | | recall@0.5: <64 | 64-256 | 256-1k |
|---|---|---|---|---|---|---|---|
| 24K | 0.410 | 0.601 | 0.753 | | 0.090 | 0.325 | 0.646 |
| 48K | 0.403 | 0.621 | 0.762 | | 0.094 | 0.329 | 0.672 |
| 72K | 0.419 | 0.606 | 0.779 | | 0.110 | 0.353 | 0.692 |
| 96K | 0.424 | 0.616 | 0.787 | | 0.119 | 0.367 | 0.699 |
| 112K | 0.394 | 0.620 | 0.796 | | 0.113 | 0.376 | 0.694 |

- **<64 桶：不涨**。24K→112K 全历史线性斜率 **−0.003/8K**（近 5 窗 −0.017/8K，含 64K 处一个 n≈35 的噪声尖峰 0.554）；线性外推 256K ≈ 0.37-0.42，即**持平**。
- **64-256 桶：持平**（+0.001/8K，外推 256K ≈0.62-0.68）。
- **256-1k 桶：仍在涨**，+0.0039/8K（近 5 窗 +0.0046），外推 256K ≈ **0.865-0.879**。
- 对照：同期 recall@0.5 持续上升（<64: 0.090→0.113，64-256: 0.325→0.376）——**AP_s 的增长（+0.42pt/8K 全程，+0.21pt/8K 近 5 窗）由"检测到"贡献，不由"打得高"贡献**。
- 主进程疑点"0.41 是否仍在涨"的答案：**否**。最小两桶的分数成型已经停滞 88K 步；lr 仍在峰值的 ~85%（cosine 44% 处），不存在"lr 已塌所以分数不再涨"的解释——是均衡点问题不是优化衰减问题。
- AP_s 外推（trainer 口径 +0.21~+0.42pt/8K → 256K = 0.303-0.347；换算标准口径 ≈ **0.33-0.38**）。104K→112K 只 +0.03pt，减速明显，取下沿更稳妥。

### 3. FP 结构与重复度（必做 3）

按 **det 面积**分桶（注意与 GT 面积桶的错位：tiny GT 的 det 因 mask 出血通常落在一档更大的 det 桶）：

| det 桶 | n | matched | hard-FP(IoU<0.3) | TP分中位 | hard-FP分中位 | hard-FP>0.5 | AUC(TP vs hard-FP) |
|---|---|---|---|---|---|---|---|
| <64 | 16971 | 9.5% | **80.5%** | 0.100 | 0.063 | 91/13656 (0.7%) | 0.753 |
| 64-256 | 10882 | 30.1% | 59.0% | 0.080 | 0.093 | 365 (5.7%) | **0.490** |
| 256-1024 | 16408 | 58.2% | 29.1% | 0.522 | 0.105 | 741 (15.5%) | 0.654 |

- **高分 FP 在小桶几乎不存在**（<64 桶 0.7% 的 hard-FP 过 0.5）：分数已经把小桶 FP 压在底部，删光它们也只 +0.07pt（`fp_sanitize_lt64`）——与 c0 时代 E2/E4 的"FP 挤占无罪"结论一致。
- **按 GT 面积桶的重复度**（distinct det 数，无共享重复计数）：每个被命中的 GT 平均有几个 IoU≥0.5 det：<64 **14.05**（70% 的命中 GT ≥2 个）、64-256 **4.24**、256-1k **1.96**；全局 1.135。命中小 GT 时模型呈"喷洒"状——~14 个 query 同时锁上同一个 8px 目标。
- **oracle 去重**（每 GT 留最高分 det）：AP_s 0.2925→**0.2951（+0.26pt）**；<64 桶 0.0298→0.0437。重复 det 本身排在 keeper 之下，对 AP 几乎无害；其害处在分数统计上稀释 TP 信号并使任何桶级提分方案连带抬 FP。
- 小 GT 代表 det 的图内排名（0 基，≤100 个 det 内按分排）：<64 中位 49（p90=94，23% ≥90）；**64-256 中位 85**（36% ≥90）；256-1k 中位 83。这是 dump 后（已过 topk-100 切割）的排名——切割前的挤占只会更重。这是堵点 2（topk×低分耦合）的直接证据。

### 4. 打分链路代码核查（必做 4）

链路（`models/magformer/arch.py:1483-1572` `_inference` / `_inference_raw_gpu`，eval 走后者，bit-identical）：
`pred_logits.sigmoid()[..., :-1]` →（200 query × 1 类）**topk(100) 按 cls 分切割** → mask 双线性上采至 1024 → 0.5 阈值二值化 → `final = cls × mean(sigmoid_prob ∈ binary_mask)` → dump（`engine/eval_runtime.py`，score_threshold=0，evaluator maxDets=100）。

- **focal 实现**（`models/common/criterion.py:390-421`）：`torchvision.sigmoid_focal_loss(alpha=0.25, gamma=2)`，对真实类通道+eos 通道逐 logit；unmatched query 整行 × `eos_coef=0.1`。`class_weight=2.0` 只整体缩放 `loss_ce`（mask 5 / dice 20 仍主导）。
- **1 类极不平衡的均衡点**：本数据 181684/3276 ≈ 55.4 实例/图（含小目标图 66.5），Q=200 → **~28% query 为正**。逐 logit 权重：正类 0.25 vs 背景类通道 0.75×0.1=0.075（3.3:1 偏正）；每图梯度质量 正 13.9 vs 负 10.9（**1.27:1 偏正**；M2F 典型 Q=100/~10 实例是 1:2.7 偏负）。所以"alpha 压制正类"的整体质量论证不成立——问题在**每对象**粒度：小目标仅占实例 5.25% → ~5% 的正梯度质量，叠加归属翻动（AIM 之前）与 tiny mask 融合压分，其 cls 饱和最晚；观测结果（<64/64-256 分数 88K 步持平在 0.29-0.62）与此一致。**`balanced_ce:true` 只改 mask 点 BCE（`_loss_masks` 内 fg/bg 加权），分类 focal 的平衡从未被触碰**——"focal 平衡从未被直接攻击"在代码层属实。
- **topk 切割位置**：切割发生在 cls 分数上、mask-score 融合之前（`arch.py:1531-1532`）；200 query 只输出 100。小目标 query 的 cls（推算 ~0.45-0.55，=final/0.85）与背景 query 竞争最后 ~45 个名额。
- **EMA**（`engine/model_ema.py`、`trainer.py:1660-1699`）：decay 0.9999（τ=10K 步 ≈ 1.25 个 eval 窗口），warmup 2500；eval 与 best.pt 均用 EMA shadow（apply_shadow/restore 包住 eval）。以近 5 窗 APs 斜率 +0.21pt/8K 计，**EMA 滞后 ≈0.27pt AP_s**；cosine LR→0 后消失，但若 128K 守护停止后直接封盘，这 0.2-0.3pt 会留在桌上（补一次 raw 对照即可核实，AUD-3 有先例机制）。
- **AIM matcher**（`models/common/matcher.py:268-` `_aim_small_gt_costs`）：仅改训练期 Hungarian 代价的小 GT 列（确定性 bbox+30% 网格上的尺度无关 soft-Dice），推理 0ms、不碰打分链路任何环节。

### 5. 缺口归因：打分 vs 掩码 vs 召回（必做 5）

以标准口径 base AP_s=0.2925、stride-2 GT-oracle 天花板 0.768 为锚：

| 成分 | 量（pt） | 占比 | 依据 |
|---|---|---|---|
| stride-2 mask 量化（vs 全分辨率 1.0） | 23.2 | — | E3 oracle 表（已接受的架构成本） |
| **打分/排序/重复**（当前 det 集上重打分可达） | **≤1.6**（实测 +1.33~+1.61） | **~3%** | `a_plus_dedup_ranked` 0.3058 / AR_s 0.3086 |
| **缺检**（小 GT 在 IoU≥0.5 无任何 det） | **≈29**（区间 27-31） | **~62%** | 乘法分解：R@0.5_small 0.619 vs oracle 0.999 |
| **高 IoU mask 衰减**（0.5→0.95 存活率 0.50 vs oracle 0.77） | **≈17**（区间 15-19） | **~36%** | R@thr 阶梯：0.619@0.5→0.270@0.75→0.027@0.9→0.003@0.95 |

分解用乘法近似 AP≈presence×survival（对 oracle 与 F1 两端都自洽闭合，残差 <0.5pt）；给区间是因为 presence/survival 的线性转移假设在极端桶（<64）只精确到 ±2-3pt。结论稳健：**打分维度可兑现空间 ≤2pt，缺口结构仍是"检测不到（62%）+ mask 高阈值崩（36%）"**——与 c0（59.2/39.1/≤1.7）同构。

## AIM 覆盖度说明（打分维度）

- **AIM 解决了的**：匹配的确定性与边距（19×）、小 GT 代价列信噪比、推理零成本。对打分的意义是**间接**的：归属稳定 → 同一 query 持续吃到正 focal 梯度 → cls 才有成熟的可能。这是"可得分"的前置条件而非实现本身。
- **AIM 不覆盖的（本维度实测残余）**：(i) cls focal 平衡（alpha 0.25/eos 0.1 为 80 类时代的默认值，未按 1 类 + 28% 正率重标）；(ii) topk-100 输出切割对低 cls 小目标的存在性挤占；(iii) cls×mask-prod 融合的尺度结构性压分；(iv) EMA 评估滞后；(v) 喷洒式重复（14 det/命中 tiny GT）背后的 query 竞争缺失。
- **量级判断**：冻结模型 assignment headroom 本就只有 1.1pt（斗兽场结论）；本报告进一步表明**即使打分完全完美，当前 det 集也只多 1.3-1.6pt**——AIM 通过打分通道的残余收益上限与之一致地小。AIM 的真实价值在使召回增长可持续（§9 趋势、AIM 稳定性使能），打分维度的杠杆必须作用于**训练期分数成型速度**（堵点 2 的修复方向）而非 eval 期排序。

## 复现

```bash
cd /home/hdd3/zhanghaonan/magformer/output/aps_20260913/reviews_20260920/r1_score
CUDA_VISIBLE_DEVICES= /home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python analyze.py    # 主反事实+成熟度+FP结构 (~35min)
CUDA_VISIBLE_DEVICES= /home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python analyze2.py   # 修正dedup+干净上界 (~22min)
CUDA_VISIBLE_DEVICES= /home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python analyze3.py   # 排序最优上界 (~5min)
CUDA_VISIBLE_DEVICES= /home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python analyze4_subset_check.py  # 子集偏差核验 (~1min)
```
