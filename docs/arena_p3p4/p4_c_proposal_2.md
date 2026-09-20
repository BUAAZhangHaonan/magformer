# P4-c 提案 #2｜QS-Package：质量目标分类损失 + 图内排序正则 + 尺度分层贴入增密（三件同 commit）

提案人：独立提案人 #2 ｜ 2026-09-21 ｜ 场：P4-c 曝光与分数成型
一句话：**把分类分数从"存在性 0/1"改造成"匹配质量读数"（MAL-D，度量与 AIM 同源），用图内 matched-vs-unmatched 排序正则直接训练 AP 所需的排序结构（反喷洒、压 hard-FP），用深度同步 + 质量门控的尺度分层 copy-paste 把 <64px 正样本供给 ×2.3——三者按 DEIM"增密必配质量感知损失"的铁律打包上線，全部训练期专用、推理 +0 ms。**

组件表（三因子，单因子可开关，共用一套验证门）：

| 因子 | 内容 | 攻击的根因 | 训练开销 | 推理开销 |
|---|---|---|---|---|
| A | MAL-D：matched query 分类目标 1→q（soft-Dice，detach + warmup 混合） | 分数语义不含可达质量；one-hot 目标下小目标 cls 均衡点结构性贴地 | ≤4.5 ms/it（上界，实取复用路线 ≈0） | 0 |
| B | 图内排序正则：matched（按 q 加权）必须高于全部 unmatched 一个 margin | 14.05 det/GT 喷洒、64-256 桶 TP/FP AUC 0.49（随机） | <1 ms/it | 0 |
| C | 尺度分层贴入：bank 4000 + 深度 SNR 入库门控 + "每图小目标补足到 4"自适应预算 | 0.85 可见 <64 GT/步的正样本饥饿 | GPU 0；CPU worker ~100-140 ms/图（完全被 1.44 s 步时隐藏） | 0 |

---

## (a) 机制因果链：为什么攻击根因而非症状

### 根因诊断（三层）

**根因 1｜分数语义错位（A 的靶子）**。`criterion.py:_loss_labels`（390-421 行）对 matched query 的分类目标是 one-hot 1.0：`sigmoid_focal_loss(alpha=0.25, gamma=2)` + `query_weights[eos]×0.1`，从未按"1 类 + ~28% query 正率"重标定（R1 §4 代码事实）。one-hot 目标下，cls 头被要求"只要匹配上就输出 1"，但小目标的掩码证据天然弱（matched maskness p50 = 0.613/0.723/0.821 随桶单调；masked-pooling 特征少、0.85 次曝光/步），模型无法在弱特征上兑现 1.0 的目标，均衡点只能停在"特征所能支撑的置信度"——这就是 <64 TP 中位 0.29、且 24K→112K 斜率 -0.003/8K（持平 88K 步，R1 §2）的机理：**不是没训够，是目标本身与可达质量矛盾，损失在均衡点互相抵消**。cosine lr 在 112K 仍处峰值 85%，排除"优化衰减"解释——是均衡点问题。事后变换救不了：分数是模型的诚实输出，改输出端不动模型 = 只重排现状（分位重标定 -13.1、score=IoU -11.9，R1 §1），因为 <64 det 池 80.5% 是 hard-FP、64-256 桶 AUC 0.49，TP/FP 在分数轴上不可分——**可分性必须被训练出来**。

**根因 2｜排序结构缺失（B 的靶子）**。AP 的最优排序不是"按质量排"而是"每 GT 一个代表、代表按质量排、冗余沉底"——R1 §1 的 `a_plus_dedup_ranked`（+1.33pt，达自身检测集合上限的 95.5% 才被反超）正是这个构造。当前模型每个被命中 <64 GT 平均挂 **14.05 个** IoU≥0.5 的 det（70% 的 GT ≥2 个），喷洒出的 13 个 unmatched 副本只吃 eos×0.1 的弱负梯度，徘徊在中低分形成 hard-FP 池；而 matched 与 unmatched 之间**没有任何损失项直接约束它们的相对次序**。64-256 桶 AUC 0.49（随机）证明分数对"谁是被匹配的正样本"几乎零信息。

**根因 3｜正样本供给饥饿（C 的靶子）**。batch=4 每步 0.85 个可见 <64 GT、小目标像素份额 0.124%、损失点份额 0.1-0.4%（R5 §1）。分类流上小目标只占 ~5% 正梯度质量。copy-paste 通道已修复（deposit 存 `crop_depth`、贴入同步写深度、`prefill_images` 预填已实装，transforms.py:1131-1141 / dataset.py:495-516），但默认参数是 COCO 量级（bank 200 / max_paste 5 → E[贴入小目标/图]≈1.2，密度仅 ×1.13，R5 §5）：**通道通了，流量没开**。

### 因果链（组件→机制→观测量的映射）

- **A（MAL-D）**：q = matched pair 的 soft-Dice（与 AIM 匹配代价同度量）。分类目标变为 q 后，BCE 最小解 σ(logit)→q，**分数按构造收敛到"该 query 此刻可达的掩码质量"**：(i) 小目标 TP 的 cls 从均衡点 ~0.48（=0.29/0.613）抬到 ≈q≈0.61，最终分 cls×maskness≈0.37，TP 中位 0.29→0.37-0.45；(ii) 更关键的是**方差**：低质量匹配（q≈0.4）不再被推向 1.0，高质量匹配（q≈0.8）获得全额正梯度（DEIM-MAL 凸加权 q^α）——TP 内部按质量分层、TP 与未匹配之间的目标间距拉开，这正是训练出 AUC 的机制；(iii) 排除"质量目标反而压低小目标分"的担忧：小目标 q p50≈0.61 仍远高于当前 cls 0.48，且 hard-FP（unmatched）目标恒 0——分离面加宽而非收窄。
- **B（排序正则）**：图内 hinge `relu(m + s_neg − s_pos)`，按 q^α 加权（高质量正样本强制更大间隔）。直接把 `a_plus_dedup_ranked` 的排序结构**训练进模型**：keeper 压到喷洒副本之上、副本沉底。这是"改变排序"的合法机制（P4-d 反面教训只封杀事后单调变换，训练期损失改变的是模型本身）。作用点恰是 topk 切割病灶：小 GT 代表 det 图内排名中位 85/100、36% ≥90（R1 §3）——排名上移直接减少被 topk-100 吞掉的存在性。
- **C（贴入增密）**：深度 SNR 门控入库（过滤 |Δdepth|<1.0σ 的实例，X-Paste"入库质量过滤是承重组件"）+ 自适应预算"每图可见小目标补足到 4"（AD-Det 缺什么贴什么）+ bank 4000。可见 <64 GT 0.85→≈1.9-2.1/步（算式见 (d)），小目标正曝光 ×2.1-2.3。**按场规铁律与 A 同 commit**：DEIM 消融明确"增密无 MAL = 分数通胀"（naive o2m 32.6→8.4 的同族教训）——A 恰好保证增量正样本的分数按质量成型，而非"小目标一律给分"。

### 为什么这是"最短路径"

A+B 是损失函数级改动（criterion.py 内 ~60 行，零新参数、零新模块、零推理变化）；C 是已修复通道的参数与两处小改（deposit 门控 + 预算逻辑）。三者都不动塔、不动 decoder、不动匹配器本体。

---

## (b) 与二分匹配 / set-prediction 的兼容性论证

1. **匹配不变**：Hungarian/AIM 匹配、o2o 指派、200 query set 输出、`linear_sum_assignment_gpu` 全部原样。A 只改 matched query 的**分类目标数值**，B 只在匹配结果**之后**对 cls logit 加正则——都不参与代价矩阵（`matcher.py:forward` 零改动）。
2. **set 语义自洽**：每个 GT 仍被恰一个 query 认领；unmatched 仍是 no-object（eos 0.1 不动）。改变的是"被认领者学什么"，不是"谁被认领"。
3. **度量闭环且无循环性**：q 用 soft-Dice——与 AIM 代价列同度量（matcher.py:426-437 的 grounded soft-Dice）。Stable-DINO 的处方是"位置度量进 cost **且**进 cls 目标"；我们已走了前一半（AIM），A 补上后一半，分数成为匹配度量的一致读数。无循环论证：q **detach**，分类梯度不回流掩码；掩码梯度不经过分类目标。
4. **与 cost_class 的相互作用是良性的**：`cost_class = -sigmoid(logit)`（matcher.py:375）。MAL 训练后高分 ⇔ 高匹配质量 ⇔ 匹配器更倾向选它——分数与匹配互相强化而非打架；Rank-DETR（NeurIPS23）在 DETR o2o 上验证了"排序损失 + 匹配"共存的正确性，RS Loss（ICCV21 oral）在 o2o Faster R-CNN 上同理。
5. **B 与 o2o 的副本结构正交且同向**：o2o 下"每个 GT 一个 keeper、其余沉底"恰是 AP 最优结构（R1 §1 判读）；B 把这个结构从"匹配的副产品"变成"显式受监督的输出"。不会出现"把第二好匹配压死后无人兜底"的风险：匹配每步重算，keeper 由 AIM 全覆盖决定，副本的沉底不改变下一step的匹配输入。
6. **深度监督兼容**：9 层 aux 各自匹配、各自算各自的 q 与排序项（同一代码路径，forward 的 aux 循环 250-270 行处同样调用）。
7. **不踩场的禁区**：不引入 o2m 解码支路（那是 P4-a 的领地）；本提案的"增密"全部走数据侧（copy-paste），且按铁律与 MAL 打包——即 DEIM 的"Dense O2O（数据增密）+ MAL"组合形态，而非 Group/H-DETR 的 query 增密形态。

---

## (c) 理论收益上界与依据

### 分项锚点

**A+B（损失侧）**：
- 上界锚 1（内部，最硬）：R1 §1 反事实 `a_plus_dedup_ranked` = **+1.33pt**（当前检测集合上"keeper 按质量排 + 冗余沉底"的完美实现）。B 训练的正是这个结构、A 提供质量读数——二者合并的理论兑现 ≤ +1.3-1.6pt（AR_s 锚 +1.61）。注意此锚只覆盖"现有 det 集合上的重排"；训练期改动还会改变 det 集合本身（分数成熟 → keeper 进 topk、曝光增加 → 召回），故它是下界意义上的保守锚而非上界。
- 上界锚 2（文献）：VFL（CVPR21）质量目标 +2.0 AP over FCOS+ATSS（调研C 总表 #1）；RS Loss LVIS+RFS **+3.5 mask AP、稀有类 ~+7**（#6）——"稀有类"对应我们的稀有尺度桶，按单类单桶人口占比折算取 +1.5-2pt 保守档；Rank-DETR 高 IoU 档 AP 增益更明显（与 P3-a 协同）。
- 上界锚 3（变现门）：R3 oracle 补全 +37.5（oracle 分）vs +6.7（现实分）——变现率 18%。A 让 cls 携带质量信息正是打开这道门的钥匙，但门的另一侧（覆盖）属 P4-b，本场只主张分享其收益：门开到 40% 时折算 +2-3pt（与 P4-b 分摊后本方案记 1/3）。

**C（供给侧）**：
- R5 §5 内部估算（Ghiasi CVPR21 +1.2 AP COCO / LVIS +2.4-3.7 / 小目标专用线 +3-7；Kisantal 反复贴入 +9.7% 相对）：修复深度通道 + max_paste 12-20 启用 = **AP_s +2~+5pt**。
- 增密算术（我们的数字）：E[贴入小目标/图] = prob 0.5 × 预算 12 × prefer_small 抽中率 0.75 × 通过率 0.65 ≈ 2.9/图 → 可见小目标 2.37→~5.3/图（×2.2）；其中 <64 桶按 bank 桶构成（7001/69013≈10%）贡献 ≈0.29/图 × 4 图 ≈ +1.16/步 → **可见 <64 GT 0.85→≈2.0/步（×2.4）**，损失点份额 0.1-0.4%→~1% 量级。c0 全量 300K 仍 0.236 vs 设计臂密度驱动的持续放大（R5 §5 数据量旁证）证明**密度才是变量**。
- 深度 SNR 门控的必要折算：门控丢 ~30-40% 源池（56,123→~34K）仍 ≫ bank 4000，不伤供给；换来的是规避 SOC/InstaBoost 指出的上下文失配 FP 侧信道。

### 合成上界（本提案对 AP_s 的主张）

- **保守（12K 短训可判）**：+1.5~+2.5pt（A+B ~+1-1.5 早发型 + C 密度红利初段）。
- **预期（128K 等价训练）**：**+3~+6pt**（trainer 口径 0.2675 → 0.30-0.33）。
- **理论上界**：~+8pt——需 P4-b 覆盖同时落地、变现门 18%→40%+（含分摊）；相对 16.5pt 的"存在性+分数"缺口，本方案独立可主张约 1/3。
- 判读数字的**构造性下限**（不依赖文献）：A 上线后小目标 TP cls 按构造收敛到 q≈0.61 → 最终分中位 ≥0.37（vs 0.29）；这不是预测而是 BCE 最小解的性质——短训若连这个都到不了，说明实现有 bug（验证门用它做 sanity）。

### 明确不主张的

- 不主张纯打分维度 >2pt（R1 已证当前 det 集上重排上界 +1.6）；本方案 AP 增量的主体来自"排序结构被训练进模型 + 供给增加改变 det 集合"，不是校准。
- 不主张解决缺检主体（38.1% 无检测是 P4-b 的山）；本方案让"已检出者"变现、让"供给增加者"被检出。

---

## (d) 开销预算（给估算算式）

**训练 ms/it**（基线 1.44-1.60 s/it，红线 +20% = +288-320 ms）：

- **A（q 计算）**：每层 aux 用 2048 点独立采样（不复用，保证与 P3-a 因子解耦）：N_matched ≈ 55.4 实例/图 × 4 = 222 对；grid_sample 222×2048 双线性 ≈ 222×2048×4 texel 读 ≈ 1.8M 次访存 ≈ 0.4-0.5 ms/层 × 9 层 ≈ **≤4.5 ms/it（+0.3%）**。备选复用路线（读 `_loss_masks` 已算的 per-dice，`_dice_loss` 加 `reduction='none'` 选项）：**≈0 ms**。分类损失算术在 (4,200,2) 张量上，<0.1 ms。
- **B（排序正则）**：pairs = Σ图 (正×负) ≈ 4 × (55×145) ≈ 32K 元素的 hinge + 归约 ≈ **<0.5 ms/it**；显存增量 32K float ≈ 128 KB 瞬时。
- **C（copy-paste）**：GPU 侧 0。CPU 侧每图 12 次贴入 × (40² 量级 crop 的 img/mask/depth resize + 区域写 ≈ 8-12 ms) ≈ **100-140 ms/图**。布局：ims_per_batch=1/GPU、步时 1.44 s，每 worker 的 1.44 s 预算内只需产 1 图 → **完全被流水线隐藏，ms/it 增量 ≈0**；最坏情形（worker 数不足）+140 ms = +8.8-9.7%，仍 <红线一半。
- **合计**：≤ +5 ms/it GPU（复用路线 ≈+1 ms）；含 CPU 最坏 +145 ms = **+9-10% ≪ +20%**。

**推理**：A/B/C 全部训练期专用（损失与数据管线），`gpu_postprocess.py` 打分链零改动 → **+0.00 ms**（目标 ~100 ms 不受影响）。

**显存**：
- GPU：损失侧新增张量合计 <1 MB；bank 在 CPU。**GPU 显存增量 ≈ 0 MB**（当前占用不变，红线 24GB 无压力）。
- CPU RAM：bank 4000 条 × (crop_img 40²×3 B + crop_mask 40² B + crop_depth 40²×4 B ≈ 12.8 KB) ≈ **51 MB**；`prefill_images=600` 在 fork 前填满 → 4 个 DataLoader worker 经 COW 共享只读副本，实际物理内存 ≈51 MB + 写时复制页。可忽略。

**GPU 侧原则合规**：所有张量计算（q、hinge、focal）在 GPU 上随损失图执行；仅贴入留在 CPU worker——理由：贴入是 numpy 图像空间操作，若搬 GPU 需每步 1024²×3 图像 CPU→GPU 往返（≈8 ms×2 + 同步点），比计算收益更贵，符合"仅当传输时延超过计算收益时才允许 CPU"的豁免条款。

---

## (e) 正确性验证方案

### 1. 单元测试（不占 GPU）

- **A**：(i) 软目标 focal 数值——q∈{0, 0.5, 1.0} 手算 vs 实现（torchvision `sigmoid_focal_loss` 的 p_t = p·t+(1-p)(1-t) 支持软目标，逐值断言）；(ii) **等价回退测试**——warmup 权重 w=1 且 λ_rank=0 时，loss_ce 与基线逐位一致（保证不是"顺手改了别的"）；(iii) q detach——`q.requires_grad == False` 且掩码损失梯度与基线一致；(iv) per-instance dice 的一致性：`_dice_loss` 的 sum/num_masks == per-instance 均值。
- **B**：合成分数下 hinge 的梯度只达 cls logit（对 pred_masks 梯度为零）；margin/加权单调性（q 大 → 惩罚大）。
- **C**：(i) 入库门控——构造 |Δdepth| = 0.5σ/1.0σ/2σ 的合成实例，只有后两者入库；(ii) 贴入后 masks/boxes/labels/depth 四者长度与形状一致、贴入区 depth 与 crop_depth 逐位相等（有效像素）；(iii) 自适应预算——构造可见小目标 {0,2,5} 的图，验证贴入数 {4,2,2}（补足到 4）；(iv) seed 固定下 100 图贴入确定性。

### 2. 冻结基座探针（F1 128K 封存权重，~30 min GPU，CPU 亦可）

- **q 分布标定**：在 val 子集上重算 matched query 的 soft-Dice 分桶分布，验证 q p50 ≈ 0.61/0.72/0.82（maskness 0.613/0.723/0.821 的紧代理）。这是 A 的设计前提：若 q 实测显著低于 0.5，需先降 q_min 或延长 warmup 再短训。
- **AUC 可分性基线复核**：复用 R1 脚本口径锁定 0.753/0.490 两个门基线。

### 3. 12K 短训门（从封存 128K 基座续训，seed 固定，4 臂：control / +A / +A+B / +A+B+C）

主判据（12K 步时点，trainer 口径 + 标准 COCOeval 双报）：

| 指标 | 基线 | 过门值（+A+B+C 臂） | 判读 |
|---|---|---|---|
| <64 TP 分数中位（best-IoU 口径，全 val） | 0.29 | **≥0.42**（构造下限 0.37） | A 的直接效应 |
| TP-vs-hardFP AUC（64-256 / <64 det 桶） | 0.490 / 0.753 | **≥0.70 / ≥0.85** | B 的直接效应 |
| 固定池 FP 率（等池容量 100/图，<64 桶 hard-FP 数） | — | **增幅 ≤ +10%**（同时 TP 中位 +≥0.13） | 反搭车门（区别于事后抬分的判据） |
| AP_s（vs 同步数 control 臂差分） | 0.2675 | **+A+B ≥ +1.0pt；+A+B+C ≥ +2.0pt** | 净效应 |
| recall@0.5 small | 0.619 | 不低于 control −0.005 | 安全门 |
| 贴入区边缘 FP 计数（新监控） | 0 | 无异常尖峰 | C 的上下文失配哨兵 |

失败判读树：TP↑ 且 FP 同幅↑（搭车）→ λ_rank 0.5→1.0、mal_alpha 1.5→2.0 重跑 6K；AP_s 反降 → 查贴入边缘 FP → 收紧 SNR 门 1.0σ→1.3σ 或 max_paste 12→8；小目标 cls 仍贴地但 q 正常 → 检查 masked-pooling 表征（转 P3-b 领地，勿再调损失——调研C 反面教训 8）。

单因子判读：+A 臂单独看"TP 中位 0.29→≥0.37 且 AUC 微升"；+B 臂增量看 AUC；+C 臂增量看小桶 recall 与正曝光计数（训练日志加 <64 GT/步统计）。

---

## (f) 与其它场胜出方案的叠加兼容性

| 场 | 胜出方案族 | 兼容性 | 说明 |
|---|---|---|---|
| P3-a 边界监督 | 边界带过采样 / 距离图损失 | **正协同** | A 的 q 用 soft-Dice：边界损失抬 dice → q 自动抬 → 分数目标自动升。良性闭环；但需监控 dice 上升期 q 的 EMA 稳定性（q 的 warmup 混合已缓冲）。实现上正交：A 刻意不复用 `_loss_masks` 采样路径，P3-a 改点采样不互相牵连。 |
| P3-b 小掩码表示 | DCT-Mask / 软导出 / 面积自适应阈值 | **正协同** | 表示改善 → maskness 上限 0.95→更高 → 最终分 = 更高 cls(q)×更高 maskness 双重受益。P3-b 动 head/导出，A 动损失目标，无交叠。 |
| P3-c 边界稳定性 | lr 分组退火 / EMA 策略 | **正交** | 日程层 vs 损失层；A/B 的 warmup 计数与退火日程独立计时。短训门建议与 P3-c 对照臂（128K→160K 收尾）分跑，勿混臂。 |
| P4-a o2m 增密 | Group/H-DETR 辅助支路 | **打包关系** | 本提案已自带"增密×质量损失"绑定（C×A）；若 P4-a 胜出，其 o2m 支路的 matched query 直接走同一 `_loss_labels` 路径拿 q 目标、组内走同一 rank 正则——即"o2m 必配 MAL"铁律的现成满足。排期上 A 先于 o2m 落地（P4-a 依赖 P4-c 的分数绑定结论，BRIEFS 执行顺序已如此规定）。 |
| P4-b 覆盖/种子 query | 种子注入 + 分数绑定 | **共享组件** | P4-b 的"分数绑定"正是本提案因子 A（R2b 探针协议复用时直接用 A 的实现）。种子 query 改出生不改目标：A 的 q 按 matched 对计算，与 query 来源无关。若两场都选 A，同一份代码。 |
| P4-d 导出融合 | topk200 / maskness 解耦 / CQTR 重打分 | **互补** | P4-d 修融合公式（maskness 尺度归一），A 修 cls 语义——两侧独立且相乘受益：P4-d 归一 maskness 后，桶内排序由 cls(q) 主导，恰是 A 训练出的信号。topk200 已 config 化，A 上线后 topk 切割损失进一步缩小。注意：A 训练的 cls 分数语义变了（质量读数），P4-d 的离线重打分需在 A 短训门之后重新标定基线。 |

全局叠加顺序建议：P4-d（零训练）→ **本提案 A+B 先行（C 同 commit 但可单因子关）** → P4-b/P4-a 在 A 的分数绑定结论上展开。

---

## 附：代码接入点清单（对照真实实现）

1. `models/common/criterion.py`
   - `SetCriterion.forward`（195 行）：`indices = self.matcher(...)`（218 行）之后与 aux 循环（259 行）各调一次新增 `_compute_quality_targets(outputs, targets, indices, num_points=2048)`，结果经 `_get_loss` 新参 `quality` 传入。
   - `_loss_labels`（390-421 行）：matched 行（`idx`）的 `target_classes_onehot[..., 0]` 由 1 改 `w + (1-w)·q`（w 由 `_mal_step` buffer 按 `mal_warmup_steps` 衰减 1→0）；matched 行损失乘 `clamp((q/0.5)^mal_alpha, 0.25, 2.0)`；负样本路径（eos 0.1）不动。
   - 新增 `_rank_loss`：按图 `relu(margin + s_neg − s_pos)·q^α` 均值，key `loss_rank`，warmup 线性升 λ。
   - 可选：`_dice_loss`（31 行）加 `reduction='none'`（复用路线）。
2. `models/magformer/arch.py::_sync_criterion_from_config`（727-802 行）：透传新 kwargs；`weight_dict` 加 `loss_rank: rank_weight`。
3. `config/schema.py::MaskFormerConfig`（270 行）：`mal_enabled/mal_alpha(1.5)/mal_q_min(0.2)/mal_warmup_steps(2000)/quality_num_points(2048)/rank_loss_enabled/rank_weight(0.5)/rank_margin(0.15)/rank_warmup_steps(2000)`。
4. `data/dataset.py::_InstanceBank.deposit`（207 行）：入库前算 crop 内外 |Δdepth|，< `deposit_depth_snr_min`(1.0σ=0.025 norm) 拒收（`crop_depth` 字段已存在，101-166 行）。
5. `data/transforms.py::CopyPasteTransform.__call__`（1007 行）：贴入前数 `result["masks"]` 中面积 <1024px² 的可见实例 n_small；预算 = `min(max_paste_instances, max(2, target_small_visible − n_small))`；`bank.sample(预算, prefer_small=True)`（1034 行）。
6. `config/schema.py::CopyPasteConfig`（71 行）运行值：`enabled: true, prob: 0.5, max_paste_instances: 12, bank_capacity: 4000, prefill_images: 600, prefer_small: true` + 新键 `deposit_depth_snr_min: 1.0, target_small_visible: 4`。

## 附：风险登记

| 风险 | 概率 | 缓解 |
|---|---|---|
| q 早期噪声大（掩码未成型时 dice 抖动） | 中 | warmup 混合 w:1→0（2K 步）+ q detach + q_min 0.2；从 128K 封存基座续训时掩码已成型（dice p50>0.6），风险减半 |
| 排序正则与 eos 0.1 的负梯度叠加过强 | 低 | λ_rank 0.5 起步 + 2K 线性 warmup；单测验证只作用于 cls logit |
| 贴入上下文失配制造 FP | 中 | SNR 入库门控 + 贴入边缘 FP 哨兵指标 + AD-Det 式按需预算（不均匀撒） |
| 深度弱对比（1.5σ 本底）限制贴入质量上限 | 确定存在 | 这是上限约束非 bug（R5 §3）；门控丢最差 30-40% 正是为不放大它 |
| 小目标 cls 因 masked-pooling 表征不足而到不了 q | 中 | 短训门判读树已设出口：转 P3-b，不再调损失（调研C 反面教训 8） |
