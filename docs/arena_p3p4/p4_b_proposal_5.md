# P4-b 提案 #5：「出生 × 持有 × 变现」三明治（SGS：Seed-born, Gated-through, Scored-by-MAL）

提案人 #5（独立，互不可见）。判分公式同全场：理论收益 × 置信度 ÷ 开销。

**一句话主张**：38.1% 存在性缺口的修复必须同时打通三个环节——**出生**（种子 query 结构性地落位在小目标上，R2b 已证 92.1% 覆盖可解）、**持有**（出生后的 query 要穿过前 7 层无门控粗网格区不被冲散，P6 残余）、**变现**（持有者的 cls 分数脱离 0.22-0.29 带，P4-c MAL-CP+ 已裁决件打包）。R2b 的失败不是出生失败，是三明治缺了后两片；只复活种子必然重演 0.29 停滞，纯覆盖方案无资格——本提案从设计上就是捆绑件。

---

## (a) 机制因果链：三重成因 × 我攻击哪两个

R3 报告（reviews_20260920/r3_recall/report.md）把存在性缺口归因为两个根因 + 一个跨场接口：

| 成因 | 证据（出处） | 本提案的攻击组件 |
|---|---|---|
| **① 出生先验缺失**：200 可学习 query 的定位必须从粗网格 cross-attn 中"涌现"，无结构性机制保证任何 query 落位在小目标上 | <32px 目标 stride-4 仅 2-3 cell、stride-8 恰 1 cell、stride-16/32 亚 cell（简报 §5；R3 D4）；涌现所需的梯度信号被 matcher 均匀 12544 点稀释（400px² GT ≈4.8 前景点，SNR 噪声主导，R3 D5） | **B1 探针种子**（R2b 复活 + MAL 绑定） |
| **② 持有路径无门控**：topk 门控 mask（256²、ratio 2.0/min 64）只被第 8 层 cross-attn 消费一次；layer 0-6 的 key 是 stride 32/16/8 循环 + 硬 0.5 阈值，c0 的 attend-everywhere/全 mask 病理保留 7/8 层 | 代码级：`multiscale_decoder.py:992`（`topk_gate = ... (head_level == 3)`）+ 层调度 `:984`（`head_level = 3 if i >= L-2 else (i+1)%3`）；R3 D4 全文核查 | **B2 门控多层化**（stride 感知 min-K） |
| ③ 供给/分数（跨场）：0.85 可见小 GT/步、TP 分数 0.29 vs 0.80 | R3 D2 AR_s@10=0；SYNTHESIS P4 | **不重复攻击**——直接打包已裁决的 P4-c MAL-CP+（B3），供给侧归 P4-a |

**为什么 R2b 失败精确支持这个三明治设计**（EVIDENCE.md §5 机制读数，169 小 GT，raw 权重）：probe 定位 76.3% 小 GT ✓；种子槽（136-199）以 92.1% IoU≥0.5 覆盖率持有小目标（优于学习槽 87.0%）✓——**出生环节架构层全部工作**；但最优 query 的 cls 分数仅 0.294/0.219（学习/种子），且离线抬分 ×1.5-5 让 AP_s 单调变差（0.267→0.138），证明低分是"诚实校准"（分数已隐式回归到掩码质量），**不是可后处理修的排序问题——必须训练期把目标换成质量标量**（= MAL）。同时注意：R2b 的 92.1% 是在 ② 未修复（只有 layer-0 一次性先验 `seed_attn_prior`）、③ 完全缺失（focal 目标=1）的条件下带病达成的——②③ 补齐后覆盖与分数都应有裕量。

**因果链**（种子的一生）：probe 在 stride-4 直接收小 GT 并集的稠密监督（第一个不经过 1:200 query 稀释的小目标梯度）→ 峰值 topk 选出 64 个种子，content/query_pos 从峰值 cell 的 mask_features 投影生成（`multiscale_decoder.py:767-799`，已实现）→ layer-0 注意力先验把种子第一次 cross-attn 限制在 probe 前景内（`:834-854`，已实现）→ **[新] B2 让 layer 1-6 的粗网格注意力也按 stride 感知 top-K 聚焦**，种子不再穿过 7 层 attend-everywhere 区被背景稀释 → **[新] B3 MAL-CP+** 让赢下匹配的种子的 cls 目标 = q（AIM 同源 soft-Dice），持有质量直接进分数流 → 导出端 topk-100 按 cls×maskness 排序时小目标 TP 不再垫底（AR_s@10 从 0 起步）。

## (b) 与二分匹配 / set-prediction 的兼容性

**种子槽的匹配语义**：200 = 136 学习 + 64 种子，全部进入**同一个 Hungarian**。种子不是 CDN 式"绕过匹配的免费正样本"——它必须赢下匹配才收监督，set-prediction 的全局一对一语义不变。R2b 实测共同匹配下种子确实赢下小 GT 归属（92.1% vs 87.0%），且重复摊薄不是问题（每小 GT 1.30 det、oracle 去重 −0.25pt，R3 D3）——种子是 peak-NMS + 固定 topk 的空间分散候选（64 个），不是 DDQ 反面教训里"稠密铺满不去重"的形态。

**与 AIM 的交互**：AIM 列对全部 200 query 平等计算（matcher 对 query 来源无感知）；AIM 确定性窗口（bbox 膨胀 30% 的 R×R 网格，`matcher.py:296-320`）恰好覆盖种子出生的 stride-4 cell。**种子赢匹配的机制正是 AIM 设计的初衷**：出生即高 IoU → AIM dice 列低 → Hungarian 稳定选中，margin 从掷硬币（0.13σ）变为设计意图中的 2.43σ。AIM 的 `[:16]` 截断已修（现分块无帽，`matcher.py:322-354` 已核查），种子方案不增加 GT 数、不增加 AIM 点数。

**eos/focal 纪律**：种子槽输掉匹配时与学习槽同样吃 eos 目标；B3 的负样本逐位不动（MAL-CP+ 形态）保证分数纪律不被覆盖侧破坏。

**唯一性/容量**：每图 small GT p50=1 / max=26 ≪ 200 槽（R3 D2 容量不 binding）；64 种子对 26 个小 GT 的最密图仍有 2.5× 裕量。

## (c) 理论收益上界与依据

oracle 阶梯（R3 D6，segm rescore 口径）：never-hit 3634 小 GT 注入自身 mask，@0.99 → **+37.5pt**（物理上限），@0.45（现实分数档）→ **+6.7pt**。本方案落在两段之间，依据三锚：

1. **覆盖外推（R2b）**：基线覆盖率 = 1−38.1% = 61.9%；R2b 种子 = 92.1%（小模型 4K 步、②③缺失下达成），即补齐缺口的 **79.3%**（(0.921−0.619)/0.381）。覆盖上游受 probe 定位率 76.3% 限制，92.1% 是可信工作点而非天花板。
2. **分数档外推（P4-c 裁决门）**：MAL-CP+ 12K 门要求 TP 分数中位 0.29→≥0.40（强 ≥0.50）。@0.45 注入 = +6.7pt 锚下，覆盖 79.3% × 0.45 档 ≈ **+5.3pt 下界**；分数推到 0.6-0.7 档时变现率从 18%（6.7/37.5）向中段爬升，理论带 **+10~15pt**（插值，非实测——诚实标注）。
3. **持有侧（R3 堵点 3）**：门控多层化保守 +1~3pt（无直接消融，纯代码读 + 几何推算，R3 D4 自评低置信）。

**分阶段主张**：16K 短训门 **+1.5pt**（过门线；R2b 教训——4K 微调 AP_s 全家族落在 ±0.006 噪声带，16K 内不指望大兑现，先行指标才是判据）；32-48K 组合臂中心 **+5pt（带 +3~+9）**；分数流完全成型 + 覆盖稳态的长训上限 +8~15pt。全场最高物理上界（37.5pt 的 79%）× 中置信度（机制读数是直接先验，但 AP_s 兑现从未发生）。

## (d) 开销预算（算式）

基线 1.44-1.60 s/it（4 卡 DDP，取 1.52s 中值）；P4-c 已占 +3~6%（+45-95ms，主要 AIM 列随贴入增密 42M→100M 点/步，VERDICT 采 #3 口径）。

**训练增量**：
- B1 probe+种子：probe fwd（256²×256ch 输入，1×1+dw3×3+1×1 ≈0.3 GFLOP）~1ms + bwd ~2ms + GT union maxpool 目标构建 ~1ms + 峰值 NMS/topk64/gather/2 Linears ~0.5ms + probe loss ~1ms ≈ **+5-6ms**；
- B2 门控多层化：7 个粗 level head call 各加一次 per-query topk（最大 stride-8 = 128² = 16384 cell，B·Q = 200 行 → ~0.4ms/次）；bool attn mask 无梯度流（比较/scatter 产出，`multiscale_decoder.py:1086-1114` 现有路径），bwd 增量 0 ≈ **+3ms**；
- B3 MAL：q 目标构造 ~1ms（matcher dice 列搭车，P4-c 已计其账）；
- **本场合计 +9-10ms ≈ +0.6-0.7%**；与 P4-c 叠加 → **+3.7-6.7% ≤ +20% 红线，余量 2.7-5×**。

**推理增量**（目标 ~100ms，红线 +10ms）：probe fwd ~0.5ms + 种子构造 ~0.3ms + layer-0 先验 ~0.2ms（R2b/design C 同口径实测 ≈+1ms）+ 7 次粗层 topk ~2.8ms ≈ **+3.5-4ms ≤ +10ms**。**训练期专用规避方案**（简报要求热图类必须给）：probe 是固定 shape 卷积 + 固定 k topk、无 host sync，可被现有 `cuda_graph_runner.py` / TRT 全链捕获（F1 已有 TRT 路径）；若预算被挤占，粗层门控 topk 可换 sort-based 阈值近似（verify_V2 记录的 B 微优化 −1.5-3ms），probe 可挂 FP16——降档阶梯后最坏 +1.5ms。

**显存**：probe 激活 256²×128ch×4B ≈ 33MB + topk 瞬时 float（200×16384×4B ≈ 13MB，生命周期短）+ 种子中间 <1MB ≈ **+50-100MB/卡**，≤24GB 远未触顶（粗层 attn mask 本以同 shape 存在于 0.5 阈值版，不增）。

**参数量**：probe 0.06M + 种子投影 2×256×256 ≈ 0.19M，<0.25M（<0.1% 模型）。

## (e) 正确性验证方案

**单测（训练前，全可离线）**：
1. step-0 parity：probe 末层零初始化 + warmup 300 步内 seeding 关闭 → 冻结 F1 权重前向与基线**逐位一致**（G0 模式，防 warm-start 休克）；
2. 种子确定性：同输入两次前向 topk idx 一致（tie-break 缓存 `multiscale_decoder.py:438-446` 已有）；
3. 多层门控：每层每 query ≥K_min 个可 attend cell（防全 mask 退化回 attend-everywhere）；padding 合并无 NaN；K 调度形状断言（stride-8 min16 / stride-16·32 min4）；
4. matcher 无污染：种子槽参与的 indices 与学习槽分布同族（冻结权重同 GT 对比）；
5. mal-off 逐位回归 + AIM 不受泄漏（与 P4-c 单测共享）。

**冻结基座探针（纸面轮后，R2b 协议复用，简报实测门候选对齐）**：
- P-A（300-500 步只训 probe）：probe 定位率 ≥76.3%（R2b 锚）；**<60% → 否决 B1**（probe 在 F1 特征上不 work 则种子无源）；
- P-B（零训练离线）：冻结 F1 + 打开种子注入，val 上量种子槽覆盖率 ≥85%、q 分布满足 P4-c 前置门（q IQR≥0.15、p50∈[0.5,0.8]——不过则先回 P4-c 改 τ）。

**≤16K 短训门（封存 128K 基座 warm start，单 commit B1+B2+B3）**——先行指标 4K 步即读：
| 指标 | 基线 | 4K 先行 | 16K 门 |
|---|---|---|---|
| 种子槽小 GT 覆盖率 | —（R2b 92.1% 锚） | ≥85% | ≥88% |
| 38.1% 存在性缺口 | 38.1% | ≤25% | ≤20% |
| AR_s@100 | 0.309 | ≥0.36 | ≥0.42 |
| AR_s@10 | 0.0 | >0 | ≥0.05 |
| 小桶 TP 分数中位（含种子槽） | 0.29 | ≥0.35 | ≥0.40（MAL-CP+ G1 同源门） |
| AP_s（rescore，对 R4 退火对照线） | 0.2925 | — | **净 ≥+1.5pt** |
| AP_m / AP_l 回退 | — | — | ≤0.005 |

**失败判读树（预注册）**：覆盖率达标但分数不达标 → 移交 P4-c（MAL 温度/τ 复调）；分数达标但 AR_s@10=0 → 移交 P4-d（topk 切割/配额）；覆盖率不达标 → B1 内部诊断（probe_area_max 降到 512、峰值阈值 0.25→0.15、种子 64→96）；AP_m 回退 >0.005 → 粗层 min-K 加倍（stride-8 32 / 16·32 8）或退回 only_stride4（B2 单独降档，B1/B3 保留）。

## (f) 与其它场叠加兼容性

- **P4-c MAL-CP+（推荐打包路径）**：B3 即其 cls 目标件，q 用 matcher 现成 dice 列（种子槽无差别享受）；三桶贴入增密直接抬高种子槽的正样本曝光（0.85 可见/步 ×K）——**两场是同一枚硬币：P4-c 建分数流，本场建分数流的供给对象**。预算合并 +3.7-6.7%（见 d）。
- **P4-a（o2m/CDN/Group）**：种子槽属 regular query，进 Group 主组即兼容。**硬前提**：DN 路径若启用，`_forward_decoder_layer` 的 self-attention 现以 `tgt_mask=None` 调用（multiscale_decoder.py:580/602 已核查）——DN 与 regular query 互相可见 = 作弊，必须先补块对角隔离掩码（Mask DINO 有参考实现，~20 行），此为两场共同的预登记债。
- **P3-a 边界监督**：正交（损失 vs query 结构）；q 的 soft-Dice 上限被 P3-a 抬高 → 正向耦合，须分臂归因（P4-c VERDICT 同款规则）。
- **P4-d 导出**：种子产出的检测天然低分，topk-200/尺度配额对其保留率最高；CQTR 轨迹重打分对种子 query 尤其有效（出生即收敛、跨层位移小）——零训练变现叠加。
- **P3-c / P3-b**：无交互；probe/种子投影新模块按 `hires_lr_multiplier` 先例给 ×10 lr 组，不吃分组退火的账。

## 实施清单与风险

**代码挂点**（R2b 基础设施全部在树，config 打开即活）：
- B1：`arch.py:21-47`（probe 模块）、`:989-1013`（训练分支）+ `multiscale_decoder.py:767-854`（种子构建 + layer-0 先验）——**纯 config**：`probe_enabled/probe_area_max:1024/probe_loss_weight:2.0/seed_enabled/seed_queries:64/seed_warmup_steps:300/seed_attn_prior:true`（F1 config `f1_full_design_256k.yaml` 的 mask_former 节直接加）；
- B2（新 ~30 行）：`multiscale_decoder.py:992` 的 `topk_gate` 条件改为 `mask_attn_topk_all_levels`（config 新键）+ `:1086-1114` 的 K 计算按 level 缩放 min-K（stride-8:16 / stride-16:4 / stride-32:4，clamp≥2）；
- B3：按 P4-c VERDICT 形态接线 `criterion.py:_loss_labels`（q 目标、QFL β=2.0、warmup 1000、负样本逐位不动）。
- 合计新代码 <80 行 + config；1 人日。

**主要风险与对策**：(i) R2b 类 AP_s 兑现迟滞 → 先行指标门 + 32-48K 组合臂不在 16K 内下死刑；(ii) 粗层门控伤大目标 → per-query K 随 pred_area 缩放 + AP_m/l 门 0.005 + 降档开关；(iii) probe 与 fusion/AGPE 的特征耦合劣化 → P-A 探针先行否决权；(iv) 种子槽挤占学习槽 → 200 总数不变、R2b 已证学习槽持有率 87.0% 未受损；(v) Phase-2 卡（过门后才试）：DQ/HGSQ 式种子预算自适应（probe 密度回归 → 动态配额）与 CDN-mask 对比去噪组——均预登记、不在首发 commit。

## 诚实备注

- 92.1% 覆盖来自小模型 4K 步、169 小 GT 的机制读数，F1 128K 基座上需 P-B 探针复核；+10~15pt 长训上限是插值带非实测；
- 门控多层化收益（+1~3pt）是 R3 自评低置信的代码推论，本提案把它定位为"持有路径修复"的必要件而非独立收益源；
- 若 P4-c MAL-CP+ 的 12K 门失败（TP 分数不动），本方案的变现段塌回 +6.7pt 口径的现实下界附近——三明治任何一片缺失都回到 R2b 结局，这是打包论证的原因也是本场的风险结构。
