# P3-a 提案 5：BPS —— 边界钉定监督（Boundary-Pinned Supervision）

提案人 #5（独立，未见他人提案）。场：P3-a 掩码边界监督。
一句话：**给每个被匹配 GT 的点集加一条"以 GT 轮廓为几何锚、双侧对称、距离加权"的带内采样层（每 mask +1568 点，纯 loss 侧），并配带内 Dice**，把小目标边界梯度密度从泊松(<1 点/迭代) 提到构造保证的 1568 点，不改匹配、零推理开销。

---

## 0. 设计总览

| 组件 | 内容 | 改动面 |
|---|---|---|
| B1 带内分层采样 | 每个 matched mask 的点集追加 1568 个带内点：从 GT 内轮廓像素 + 双侧 ±δ 偏移生成，δ 在半径 r_i 圆盘内均匀采样，r_i 随目标尺度自适应 | `criterion.py:_loss_masks` |
| B2 距离加权 BCE | 带内点 BCE 权重 exp(−\|δ\|/r_i)，δ 在构造时免费精确已知（无需距离变换）；带内点豁免 balanced_ce 的背景降权 | `criterion.py:_sigmoid_ce_loss` |
| B3 带内 Dice | 每 mask 在带内点上单独算 soft-Dice（边界版 DoU 的点采样形态），权重 2.0，α ramp | `criterion.py` 新增 `_band_dice_loss` |
| 开关与配线 | `boundary_supervision` 配置组，α=0 时与基线逐位一致 | `schema.py` + `arch.py:785` |

两个子轴（点采样策略 + 损失形式）一次覆盖，且二者共享同一采样构造（带内点的偏移范数既是采样坐标又是损失权重），无重复开销。

---

## (a) 机制因果链：为什么攻根因

### a1. 根因的算术（来自我们自己的代码与数字）

**证据链起点**：CF1 反事实 +27.1pt（AP_s 0.2925→0.5636，检测集合/分数/其余掩码全冻结）——掩码内容是最大单项缺口（SYNTHESIS_20260920 §一；r2_maskiou/report.md §D5）。高 IoU 段崩塌：AP@0.75=0.2465、@0.9=0.0181（r2 §D1），坍缩的 87% 归掩码内容（r2 §D5）。±1px 膨胀/侵蚀都更差（<64 桶 0.652→0.476/0.256，r2 §D4）——误差是**边界抖动**而非整体偏移。

**逐点算术**（matcher.py:184-186 代码自带注释给出公式 E[前景点] = 12544×area/1024²）：
- 面积 400px² 的 GT：E ≈ 4.8 点；面积 64px²（<64 桶上限）：E ≈ 0.77 点；10px 目标（~78px²）：E ≈ 0.94 点。场简报的实测区间 ~4-47 点是含不确定性聚类的收敛后读数，但**无任何机制保证其中任何一点落在边界上**。
- M2F 原生 importance sampling（`get_uncertain_point_coords_with_randomness`, criterion.py:89-112）是**全图全局 top-9408/37632**：小 GT 的边界只有在恰好承载全图不确定性极大值时才分到点；训练早期 logit 处处小、不确定性≈平坦，边界从未被系统性发现。这是"富者愈富"分配，小目标结构性出局。
- 一个 10px 目标：边界带（±1px，面积 ~108px²@1024²）期望落点 12544×108/1048576 ≈ **1.3 点/迭代**，且逐迭代独立重掷（泊松）。边界位置上的监督是纯噪声 → head 学出软过渡带，0.5 穿越点位置欠定 → ±1-2px 抖动 → <64 桶 matched IoU p50 卡在 0.635（链可达 0.817）。

### a2. 一个只有在我们代码里才成立的加重因子（超出文献的针对性发现）

生产配置 `balanced_ce: true`（f1_full_design_256k.yaml:126）。`_sigmoid_ce_loss` 的实现（criterion.py:65-73）：fg_ratio 被 clamp 到 min 0.01，于是小 mask 的**背景点权重 = fg_ratio/bg_ratio ≈ 0.01，前景点权重 ≈ 99**。含义：恰好落在小目标边界外侧 1-2px 的点（正是钉住外缘、抑制概率晕圈的位置）被现有损失**主动降权 100 倍**。内侧强推、外侧几乎不罚 → 经验损失的最优解是软坡而不是锐边（坡内任何 0.5 穿越位置的训练损失相同，SGD+AMP+bilinear 的隐式平滑偏置选最平滑的）→ 抖动被损失形式直接许可。±1px 探针（双向都差）与该机制完全一致：最优形态在中间且尖锐，现损失对"尖锐"无约束力。**任何搬运文献的边界损失若不处理这个交互，外缘钉定力会被 balanced_ce 直接抵消**——这是本提案必须自带"带内点豁免 balanced 降权"的原因。

### a3. 因果闭环

BPS 把"边界无约束"的三个成因逐一对位：
1. **点不在边界** → B1 从 GT 内轮廓像素构造性生成带内点（几何锚=GT 本身，与预测无关，从 step-0 起就有保证的边界梯度）；
2. **外侧不罚 / 坡许可** → B1 双侧对称 ±δ（内外各半，推与拉的力对称，钉住而非单侧推）+ B2 豁免 balanced 降权 + B3 带内 Dice（边界对齐的集合论度量，分母按带宽归一，尺度自适应）；
3. **泊松方差** → 每 mask 固定 1568 带内点，边界梯度信噪比提升 ~√(1568/0.77) ≈ **45×**。

副产物：带内点经 `point_sample` 在 1024² GT 上双线性取标签（criterion.py:452 同路径），边界 1px 内自然获得亚像素软标签——免费的"链感知"监督形态，无需另造软标签管线。

---

## (b) 与二分匹配 / set-prediction 的兼容性论证

1. **匹配器逐字不动**：HungarianMatcher 的 cost 仍是 uniform 12544 点 BCE+Dice + AIM 小 GT 列（matcher.py:356-453）。AIM 已修好的"分得对"不受影响；本提案只在 indices 落定**之后**作用于 (query_i, GT_i) 配对损失。
2. **集合不变量保持**：损失仍只依赖匹配对；带内点集是 matched GT 几何的（随机化）函数，对 query 置换等变、不跨 query 泄漏、不引入新 cardinality 项；`num_masks` 归一化不变。B1 的随机数与现有采样同 RNG 纪律，DDP 各卡独立（ims_per_batch=1/卡），无跨卡同步需求。
3. **无匹配-损失反馈回路**：匹配逐迭代无滞回；损失形式变化不改变匹配输入。损失目标与匹配 cost 之间的轻度分歧（cost 用 uniform 点、损失多一条带内层）在 M2F 家族本已存在（AIM 列已经与 uniform cost 分歧，matcher.py:426-438），无不稳证据。
4. **DN 路径兼容**：`_compute_dn_loss`（criterion.py:584-726）的 DN query 归属由构造给定（无匹配），可选用同一 per-GT 带内点集（GT 相同则点集可复用，正样本多份时零额外构造成本）。
5. **无新参数**：纯 loss 改动，state_dict/EMA/resume（f1_seal_128k 续跑通道）全兼容，`find_unused_parameters=False` 不受影响。

---

## (c) 理论收益上界与依据

**硬上限**：27.1pt（CF1，r2 §D5）——掩码内容全部回收。本提案只主张其中"边界钉定"份额。

**结构分解**（数字全部注明出处）：
- AP@0.75 段的内容性空间：现状 0.2465 vs CF1（链完美掩码）@0.75 = 0.6023（r2 §D1/D5）→ **该段纯内容空间 0.356**。边界钉定正对这一段。
- 但可兑现性受渲染链表达力约束（r2 §D3 链 frac≥0.75@0.5：<32px² 仅 0.165（归 P3-b），32-64px² 0.592，64-128px² 0.770，128-256px² 0.863）→ **收益主体在 ≥32px²，尤其 64-256px² 桶**（其坍缩度 61.7% 与 <64 的 62.0% 同病，r2 §D1，但链可表达性高得多）。
- 现状 matched frac≥0.75：<64 0.242 / 64-256 0.272 / 256-1k 0.455（r2 §D2），链可达 0.817（small 桶整体）。

**主张区间**：回收 27.1pt 的 15-25% → **AP_s（rescore）+4.1~6.8pt，中心 +5pt（0.2925→~0.34）**；地板 +2pt（文献锚：M2F 系单类加边界损失的小目标 PQ +2.0，零推理开销 [调研A #21]）；天花板份额逻辑：AP@0.75 从 0.2465→0.30-0.34。伴随指标：matched IoU p50 <64 0.635→0.68-0.73，64-256 0.674→0.72-0.78。
**置信度**：中高。机制侧证据是冻结反事实+代码算术（非相关性推断）；迁移折扣来自度量差异（文献 +14.8pt DSC 是 ISLES 小病灶 Dice [调研A #1]，我们的是 AP 检测指标且受检测集合封顶——<64 桶 rec@0.5 仅 12.2% 决定了该桶绝对量小，主要兑现地在 64-256px²）。

**为什么优于直接搬文献**（针对性论证）：
- Kervadec 距离图边界损失 [调研A #1] 需要逐 GT 距离变换，且在纯 CE 基线上无增益（WMH UNet-CE 0.757→0.756，反面教训 3）——我们的 BCE 点形态里距离在采样构造中**免费精确已知**（δ 就是采样偏移），且我们同时处理了 balanced_ce 交互（a2）与塌缩防护（保留全部原生区域损失块 + α ramp，B1 是**加法**不是替换）。
- M2F importance sampling [调研A #9] 保留不动的理由：其收益（不掉点省显存）建立在全图不确定性分配上，对大目标有效；小目标的问题是分配结构性缺席，正交可叠加。
- PointSup ~10 点/目标≈全监督 94-98% [调研A #8] 证明**位置>数量**；BPS 把"位置"从涌现性质变成构造保证。

---

## (d) 开销预算

**训练 ms/it**（现 1.44-1.60 s/it，估算式给出）：
- 现状损失侧点采样流量/卡/迭代：9 层 × N_mask(≈50) × (37632 候选 + 12544 logits + 12544 标签) = 28.2M 次 grid_sample ≈ 3.6GB 显存流量 @~400GB/s ≈ **9ms ≈ 0.6% 迭代**。损失侧本就是小头。
- B1 构造（每 GT 每迭代一次，9 层复用）：内轮廓 = m − maxpool(−m, k=3)；N×1024²×9 FLOP ≈ 0.45 GFLOP ≈ 1ms；chunk=16 时瞬态 16×1024²×4B = 67MB。带内点生成（固定 k 的 topk-over-random-keys 技巧，免 `nonzero()` 免 `.item()` 免 CPU 同步）+ δ 采样 ≈ 1ms。
- 追加采样流量：1568 点 × 2（logit+标签）× N × 9 层 = 1.4M 次 ≈ +5% 损失侧 ≈ 0.5ms。B3 带内 Dice 归约可忽略。
- **合计 ≈ +3~5ms/it = +0.3%（上限按 10× 误估也仅 +3%），红线 +20% 有 6 倍以上余量**。
- 实现纪律：全程 GPU（带内点构造、权重、损失都在 `targets[b]["masks"]` 所在设备，collate.py 已把掩码随 batch 上卡）；唯一禁区是 `nonzero().shape` 类隐式同步，单测 (e)-4 专门卡。

**推理**：**+0ms**。改动全部在 criterion（仅训练路径），`_inference_raw`/`gpu_postprocess`/CUDA graph 零触碰。

**显存**：瞬态 +67MB（轮廓提取 chunk）+ 点张量 N×1568×2×4B×9 ≈ 5.6MB ≈ **+75MB 峰值（0.3% of 24GB）**，无持久增量。

---

## (e) 正确性验证方案

**单测**（CPU 可跑，pytest 进 `tests/`）：
1. `test_band_membership`：合成 10px 方形 GT，带内点 100% 满足 \|δ\|≤r_i（构造保证）；内外比例 50%±5%；标签与 GT 双线性采样一致。
2. `test_boundary_density_floor`：面积 64px² GT，原生块 E[前景点]≈0.77 vs 带内 1568——断言边界监督密度 ≥200×。
3. `test_alpha0_bit_identity`：开关启用但 ramp α=0 → 损失字典与基线逐位一致（1e-6）——保证"纯加法"回归安全。
4. `test_no_cpu_sync`：torch profiler 断言损失前向无 cudaMemcpyD2H（GPU 纪律）。
5. `test_band_dice_gradcheck`：双精度 gradcheck。
6. `test_adaptive_bandwidth`：A∈{16,64,1024,10000}px² → r_i={1,1,2,6}（r_i=clamp(round(√A/16),1,8)；尺度自适应带宽规避 Boundary-IoU 固定带宽失真 [调研A #23 反面 5]）。

**冻结基座探针**（封存 F1 128K，eval-only，~15min GPU）：带内错分率 = 带内点中"内侧预测<0.5 或外侧≥0.5"的比例（边界可训练质量的直接读数）；记录 <64 / 64-256 / 256-1k 三桶基线值，作为短训门的同族对照量。

**≤16K 短训门**（从 f1_seal_128k 续跑，单因子开关 `boundary_supervision.enabled`，α 前 2K it 线性升，其余全冻）：
- 主判据 @8K：**matched IoU p50（<64）0.635 → ≥0.68（方向门）；带内错分率较探针基线 −30%**。
- 主判据 @16K：**AP@0.75(small) 0.2465 → ≥0.30；AP_s(rescore) 0.2925 → ≥0.32；净差 vs R4 退火对照臂（同期 +1.0~1.25）≥ +1.5pt**。
- 护栏：AP@0.5(small) ≥0.59（不拿粗形态换精度）；空掩码率 ≤ 基线+0.5%（Kervadec 塌缩教训，反面 3）；loss_mask/dice 曲线无发散；hires 头梯度范数 ≤2× 基线（P3-c 已知 20× lr 噪声源监控点）。
- 判读规则（失败归因分流）：IoU 平 + 错分率平 → 采样 bug（回单测 1/2）；IoU 升 + AP 平 → 渲染链封顶（<32px² 子桶单独看，转交 P3-b）；AP@0.5 降 → 降 B3 权重或扩均匀份额。

---

## (f) 与其它场胜出方案的叠加兼容性

| 场 | 方向 | 兼容性 |
|---|---|---|
| P3-b | DCT 头/软掩码/面积自适应阈值 | 正交且互促：我们练内容，他们管表示与解码；更锐的概率坡使 0.5 穿越稳定，面积自适应阈值更易调。均零推理开销家族。 |
| P3-c | lr 分组退火/边界 EMA | 正交旋钮；已知风险（边界加权 × hires 20× lr 噪声）恰是 P3-c 的治疗对象——Gate A 的 hires 梯度护栏就是联合判读点，冲突可解不互斥。 |
| P4-a | o2m/CDN 去噪 | DN 损失路径可原样采用 per-GT 带内点集；同 GT 的多正样本共享同一带内点集（GT 几何唯一），无重复计账。 |
| P4-b | 种子/覆盖 | 正交（query 供给侧）；带内点构造读的是增广后最终 targets，copy-paste 场景自动生效。 |
| P4-c | MAL 分数/copy-paste 参数 | 正交（cls 损失侧）；更准的掩码同时抬高 maskness 内均概率，与分数成型同向。 |
| P4-d | 导出/重打分 | 零训练侧改动，天然兼容；掩码变锐使 maskness 结构性压分的幅度收窄，轻微正协同。 |

依赖声明：本提案不依赖任何其它场先行；唯一外部约束是 <32px² 子桶的收益被渲染链表达力封顶（r2 §D3），该部分归 P3-b。

---

## 实施接入点（对照真实实现）

1. `models/common/criterion.py`
   - `_loss_masks`（L423-502）：在 `get_uncertain_point_coords_with_randomness`（L446-451）产出原生 12544 点后，`torch.cat` 带内点坐标 (N,1568,2)、逐点权重 (N,14112)、带内掩码 flag；标签仍走同一 `point_sample(target_masks, ...)`（L452）。带内点集每迭代每 GT 构造一次，9 层监督复用（归一化坐标与网格无关）。
   - 新增 `_build_boundary_band_points(self, gt_masks, ...)`：maxpool 腐蚀取内轮廓 → 固定 k topk 选轮廓像素 → 双侧 ±δ（圆盘均匀）→ 返回 (coords, \|δ\|, r_i)；chunk=16。
   - `_sigmoid_ce_loss`（L43-74）：签名加 `point_weights`、`band_mask`；带内点豁免 balanced 降权（自带 50/50 平衡）；非带内点行为逐位不变。
   - 新增 `_band_dice_loss`：带内子集 per-mask soft-Dice，键 `loss_bdice`。
   - 退役路径不复活：`_resample_small_object_points`（L504-582，bbox 内填充非边界监督）维持 `small_object_sample_threshold=0`（f1 config L38）。
2. `models/magformer/arch.py`：SetCriterion 构造（L785-802）透传新 kwargs；weight_dict 增加 `loss_bdice` 及全部 `{k}_{i}` 辅助层条目（同 L770-773 的 boxes 模式）。
3. `magformer/config/schema.py`：`MaskFormerConfig`（L270-349）新增 `boundary_supervision_enabled`（False）、`boundary_band_points`（1568）、`boundary_r_base`（16.0）、`boundary_r_max`（8）、`boundary_w_bce`（1.0，ramp 至）、`boundary_w_bdice`（2.0）、`boundary_ramp_iters`（2000）、`boundary_bypass_balanced`（True）。
4. matcher.py：零改动。gpu_postprocess.py / arch._inference_raw：零改动。

默认超参：B1 +1568 点/mask；r_i=clamp(round(√A/16),1,8)@1024²；B2 权重 exp(−\|δ\|/r_i)；B3 权重 2.0（对照 dice_weight 20 的 1/10，保守起步）；α 0→1 前 2K it。

## 风险与反面教训对照

- 塌缩（Kervadec 单独用边界损失塌成空掩码，反面 3）：B1-B3 全部为加法，原生区域损失块（9408 不确定性 + 3136 均匀 + 12544 原生 Dice）逐位保留；α ramp；空掩码率护栏。
- 固定带宽失真（Boundary-IoU 论文自证对 小目标 失真，反面 5）：r_i ∝ √A 尺度自适应。
- 精修头收益集中大目标（反面 1）：BPS 是损失侧非头侧；按构造给小目标每单位边界长度更高密度（10px 轮廓 ~30px 得 ~52 点/px，100px 轮廓 400px 得 3.9 点/px）。
- focal 类崩塌（反面 9）：不引入 focal 项。
- 推理期精修超预算（反面 6）：零推理改动。
