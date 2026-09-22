# 两方案收敛续训战役 — 预注册判读协议 (2026-09-22)

> **命名规则（用户指令 2026-09-22）**：实验名禁用无意义代号（S1/V1 类）；
> 一律 `YYYYMMDD_描述性名称`，重启用 `_r2/_r3` 尾缀区分。
> 物理运行根 `output/aps_20260913/g2_runs/` 为目录名（含在跑产物，不可中途改名），
> 其内各实验目录均为描述性名称。
> 例外记录：锚点臂在命名规则下达前启动（16:39），物理目录为
> `g2_runs/g2_s0_sanity_r1/`（对应配置 `20260922_winners_off_anchor.yaml`），
> 该目录名保留不动以免破坏运行中写入，映射关系以本条为准。

> **v2 修订（用户指令 2026-09-22 晚）**：按理论增益重排——"<1pt 级方法果断删，
> ~10pt 级优先保留，降低方法复杂度"。据此：**DN（+0.5~1.5pt）降级为 fallback**，
> **SCB+（+4~9pt，P4 唯一大增益）升格接受单变量审判**；HDA+ 退火（+0.2~0.6pt）
> 不作为独立方法，仅以 2 个 config 键作为 bass 的保护性日程保留（判决书：边界加权
> 放大 hires 噪声，二者同栈），零推理开销、可一键剥离。ACB/TREX+ 维持删除。

> 背景：B1' 五方案合并启动硬失败（AP −16.6pt / APs −90% / bbox −69pt vs A0@4K），
> 归因候选排序 seed拼接 > copy-paste > 交互。用户决策（本会话）：
> 收敛到**恰好两个方案**，在 128K 封存权重上继续续训，目标 AP_s > 0.27（理论上限 0.768）。

## 1. 选型（定稿）

| 问题 | 保留方案 | config 键 | 弃用 | 弃用理由 |
|---|---|---|---|---|
| P3 掩码质量 | **BAS-CL+ + HDA+退火**（P3-a + P3-c 裁决钦定同栈） | `bass_enabled: true` + `hires_lr_multiplier: 20.0` + `hires_mult_anneal: true` | ACB（推理侧，依赖 bass 校准探针，过门后再议） | 零训练依赖 |
| P4 小目标存在性 | **SCB+（单变量审判中）**；DN 降 fallback | `probe_enabled/seed_enabled: true/seed_queries: 64/seed_attn_prior/seed_prior_all_layers/seed_ramp_iters: 4500/seed_dropout: 0.15` | MAL-CP+（嫌疑#2 + bank 是三次 OOM 元凶）、TREX+（推理侧） | B1' 失败信号 + 内存红线 + <1pt/复杂度规则 |

**关键协议事实（本次核实）**：F1（封存权重）与 A0 均在**恒定 hires 20×** 下训练
（EVIDENCE §12-3：退火当时从未执行）。故 "winners-off" = 20× 恒定；
HDA+ 的唯一增量 = 退火。交接文档"B1' 头从零"说法有误：A0/B1' 的
`finetune_weights` 均指向 f1_seal_128k/best.pt（warm-start）。

## 2. 运行序列（GPUs 4-7，单 4-rank 束串行，内存红线 ≤90%≈225GB，看门狗 88%×3 击杀）

| 实验（配置名） | 内容 | 步数 | 目的 |
|---|---|---|---|
| `20260922_winners_off_anchor` | 全关，20×恒定 | 16K 日程，@4K 读数后手工停 | 干净锚点：修复后代码 + 128K warm-start 的 winners-off 基线（旧 A0@4K 0.8675/0.2656 受 EMA temp bug 污染） |
| `20260922_bass_boundary_supervision` | bass+退火 | 同上 | P3 包单独门（10pt 级保留者） |
| `20260922_seed_query_splicing_trial` | probe+seed，单变量 | 同上 | **SCB+ 平反审判**（v2 升格）：B1' 头号嫌疑的干净单变量复现 |
| 合并续训（过门后定名） | bass(+退火) [+ SCB+ 若过门] | 全 16K | 如 `20260922_bass_plus_seed_16k`；`20260922_bass_plus_dn_16k` 保留作 fallback |
| `20260922_dn_small_gt_denoising` | dn 单变量 | 不排程 | fallback（v2 降级，<1pt 级） |

所有臂：128K warm-start（finetune_weights=seal best.pt）、bs1/rank×4、lr 1e-5 cosine
16K、warmup 500、eval/checkpoint@4000（门臂@4K 读数后停）、workers 4 +
MALLOC_TRIM/MMAP/ARENA、无 copy-paste bank。

## 3. 预注册判读门（读数落地前锁定）

- **锚点臂@4K**：segm AP ∈ [0.850, 0.895] 且 APs ∈ [0.24, 0.30]。
  出界 ⇒ 代码基线漂移或 warm-start 链路坏了，停止战役查因。
- **bass 臂@4K vs 锚点**：硬门 = segm AP ≥ 锚点 − 0.5pt（掩码质量收益 4K 内未必
  显现，先证无害）；健康信号 = 带 Dice 损失分量有限且下行、其余损失不平坦。
  硬门失败 ⇒ P3 包退场（战役主收益消失，回报用户重议）。
- **seed 审判臂@4K vs 锚点**（v2）：**早期击杀线**——健康 probe 损失自 ~1.24 下行
  （r2 冒烟锚）；若 opt@1500 时 `train/loss_probe` 相对 @500 无下行（差 > −0.03）
  或仍 ≥0.98 平坦 ⇒ 即杀（省 1.5h GPU），SCB+ 定罪删除，B1' 归因案告破。
  AP 硬门 = segm AP ≥ 锚点 − 0.5pt；**平反线 = APs ≥ 锚点 + 1.0pt**（覆盖率通道
  兑现证据）。过平反线 ⇒ 合并续训加入 SCB+；仅过硬门不过平反线 ⇒ SCB+ 无增益
  证据，删除，合并续训 = bass 单方案；触发击杀线/硬门失败 ⇒ SCB+ 定罪。
- **合并续训@16K 终局**：成功 = segm APs ≥ 0.28（超 seal 0.2698 +1pt）且
  segm AP ≥ 0.87（不牺牲大目标）。灰区 [0.27, 0.28) ⇒ 记录并考虑
  ACB/TREX+ 推理侧叠加或延长训练。
- 判读一律用 `metrics_log.jsonl` 的 `val/segm_AP*`（EMA 口径），bbox 同看。

## 4. 基础设施

- 启停：`ops/launch_4rank.sh <config-name> [port]` / `ops/stop_bundle.sh <agent_pid>`
  （杀整棵进程树；torchrun rank 各自成组，组杀无效——本次实测发现）。
- 看门狗：`ops/watch_host_memory.sh <agent_pid> <name>` 每 20s 记 JSONL；
  used% ≥88 持续 3 拍 ⇒ TERM→KILL 整树（90% 红线前动手）。
- 运行目录：`output/aps_20260913/g2_runs/<实验名>_r1/`。

## 5. 读数记录

### 锚点臂 20260922_winners_off_anchor @4K（2026-09-22 18:20 判读）

| 指标 | 读数 | 参照 | 判定 |
|---|---|---|---|
| segm AP | **0.8704** | seal 0.8744 / 旧污染 A0@4K 0.8675；门 [0.850, 0.895] | ✅ 过门 |
| segm APs | **0.2692** | seal 0.2698；门 [0.24, 0.30] | ✅ 过门 |
| segm APm / APl | 0.8619 / 0.9756 | seal 0.8655 / 0.9769 | 持平 |
| bbox AP / APs | 0.8566 / 0.3331 | seal 0.8596 / 0.3296 | 持平 |

- 结论：修复后代码 + 128K warm-start 的 winners-off 基线**无漂移**（vs seal −0.4pt/−0.06pt，
  噪声级）；旧 A0@4K 锚确实被 EMA temp bug 压低 ~0.3pt，弃用正确。
- 本臂即后续所有门的对照锚。AMP 跳步 8 次全在前 800 步（warmup 期），之后零。
- 权重已存档：g2_runs/g2_s0_sanity_r1/{best,best_aps,last}.pt。
- 后续门判读基准（最终锁定）：**bass/seed 臂硬门 = segm AP ≥ 0.8654（锚点−0.5pt）**。

### bass 边界监督臂 @4K（2026-09-22 20:35 判读）— 硬门失败（灾难级）

| 指标 | bass@4K | 锚点@4K | 差 | 硬门 ≥0.8654 |
|---|---|---|---|---|
| segm AP | **0.7510** | 0.8704 | **−11.9pt** | ❌ 差 11.4pt |
| segm APs | **0.0484** | 0.2692 | **−22.1pt（−82%）** | 小目标几乎全灭 |
| segm APm / APl | 0.7024 / 0.9610 | 0.8619 / 0.9756 | −15.9 / −1.5pt | 大目标基本无恙 |
| bbox AP / APs | 0.4595 / 0.0530 | 0.8566 / 0.3331 | −39.7 / −28.0pt | 检测层被拖垮 |
| segm ARs | 0.0481 | 0.3154 | −26.7pt | 小目标召回崩 |

**损失层证据（vs 锚点同键@4000）**：
- `loss_mask` 0.367 vs 0.049（**7.5×**）、`loss_dice` 0.151 vs 0.038（4×）——掩码头被主动劣化
- `loss_dice_band`（新增带 Dice）随 ramp **不降反升**（0.025→0.45）——目标不可满足
- `loss_ce_0` 3.44 vs 3.46 两臂相同——层 0 CE 高是基线固有，与 bass 无关（排除）
- AMP 跳步 10 次（与锚点 9 次同级，排除 AMP 因素）

**归因结论：B1' 嫌疑排序被推翻**——bass(+退火) 单独即可复现 B1' 崩溃形态
（本臂无 seed、无 copy-paste、无 mal、无 dn）。交接文档"头号嫌疑 SCB+/次嫌
copy-paste"不成立；bass 的 ON 路径（软标签/带加权/豁免三件套之一或组合）在
128K warm-start 体制下是主要破坏源。候选毒性成分（按嫌疑）：软标签
（bass_soft_label mix 1.0 全量替换硬标签）、带 BCE 豁免改变 CE 平衡、
λ_b=3 带 Dice 与主损失打架。裁决书自己预注册过回退形态（#4 纯采样、无软标签）。

**处置**：按预注册门，P3 包退场待用户重议；seed 审判臂照常推进（P4 侧独立）。
seed 臂读数后合并续训的组合视两门结果与用户决策而定。

### bass 毒性成分离线诊断（2026-09-22 21:40，读 criterion.py `_loss_masks_bass`/`_build_bass_pack`）

全形态 bass 三件套的实现级毒性分析（按嫌疑排序）：

1. **软标签全量替换（mix=1.0）**：coverage 软图标签教头部输出亚 0.5 概率
   （<1 cell 小目标的 cell 覆盖率天然 <0.5 → 教出来 0.4 → 推理 0.5 二值化
   直接抹掉）；且两个 Dice 目标都指向软图——CF1 自己证明的表达力缺口
   （matched IoU 0.635-0.737）使该目标在 lr 1e-5 warm-start 体制下**不可满足**
   （带 Dice 0.025→0.45 爬升即为证据）。
2. **λ=3 带权 + 归一化**：非带点权重被压到 0.625×，监督重心移到最难表达区域。
3. **balanced_ce 与软标签交互**：fg 权重按软标签均值计算，语义漂移。

**打捞形态（裁决书预注册回退 #4 纯采样）已备好**：
`20260922_bass_sampling_only_trial.yaml` = 锚点 + 仅改采样分布
（三段采样/地板32/带内豁免保留；软标签关、λ=1、带Dice权重=0、退火维持删除）。
与锚点的唯一差异 = 点采在哪里。seed 审判臂结束后排队跑 4K 门。

### seed 种子审判臂 @4K（2026-09-22 22:15 判读）— 硬门失败（回归型），SCB+ 定罪删除

| 指标 | seed@4K | 锚点@4K | 差 |
|---|---|---|---|
| segm AP | 0.8501 | 0.8704 | **−2.0pt**（硬门 0.8654 差 1.5pt）❌ |
| segm APs | 0.2332 | 0.2692 | **−3.6pt**（平反线 +1pt 遥不可及）❌ |
| segm APm / APl | 0.8442 / 0.9631 | 0.8619 / 0.9756 | −1.8 / −1.3pt |
| bbox AP / APs | 0.8212 / 0.2738 | 0.8566 / 0.3331 | −3.5 / −5.9pt |
| segm ARs | 0.2570 | 0.3154 | −5.8pt |

- 击杀线未触发（probe 1.25→0.39 健康学习）——**机制活但净效应为负**：
  64 种子槽对 query 池的扰动成本 > 覆盖收益。
- 按预注册门：SCB+ 删除。P4 侧训练方案归零（DN 已降级、MAL-CP+ 已弃）。
- **B1' 归因案闭合**：主杀手 = bass 全形态（单独 −11.9pt 灾难）；次级 = seed
  （单独 −2.0pt 回归）；copy-paste 无需入场解释。交接文档嫌疑排序（seed>paste>交互）证伪。
- 打捞臂 20260922_bass_sampling_only_trial 已接跑（22:20 起，4K 门 ~00:20 出）。
