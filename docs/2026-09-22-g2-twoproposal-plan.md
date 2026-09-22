# G2 战役：两方案收敛续训 — 预注册判读协议 (2026-09-22)

> 背景：B1' 五方案合并启动硬失败（AP −16.6pt / APs −90% / bbox −69pt vs A0@4K），
> 归因候选排序 seed拼接 > copy-paste > 交互。用户决策（本会话）：
> 收敛到**恰好两个方案**，在 128K 封存权重上继续续训，目标 AP_s > 0.27（理论上限 0.768）。

## 1. 选型（定稿）

| 问题 | 保留方案 | config 键 | 弃用 | 弃用理由 |
|---|---|---|---|---|
| P3 掩码质量 | **BAS-CL+ + HDA+退火**（P3-a + P3-c 裁决钦定同栈） | `bass_enabled: true` + `hires_lr_multiplier: 20.0` + `hires_mult_anneal: true` | ACB（推理侧，依赖 bass 校准探针，过门后再议） | 零训练依赖 |
| P4 小目标存在性 | **mCDN+** | `dn_enabled/dn_scalar: 3/dn_box_noise_scale: 0.3/dn_noise_ladder: [0.05,0.15,0.30]/dn_small_gt_area: 4096/dn_max_gt_cap: 8` | SCB+（主嫌疑：probe 损失全程平 1.00、64/200 池扰动）、MAL-CP+（嫌疑#2 + bank 是三次 OOM 元凶；q 目标按"增密必配 MAL"铁律随弃）、TREX+（推理侧，留作零成本叠加） | B1' 失败信号 + 内存红线 |

**关键协议事实（本次核实）**：F1（封存权重）与 A0 均在**恒定 hires 20×** 下训练
（EVIDENCE §12-3：退火当时从未执行）。故 "winners-off" = 20× 恒定；
HDA+ 的唯一增量 = 退火。交接文档"B1' 头从零"说法有误：A0/B1' 的
`finetune_weights` 均指向 f1_seal_128k/best.pt（warm-start）。

## 2. 运行序列（GPUs 4-7，单 4-rank 束串行，内存红线 ≤90%≈225GB，看门狗 88%×3 击杀）

| 臂 | 配置 | 步数 | 目的 |
|---|---|---|---|
| S0 | `g2_s0_sanity.yaml`（全关，20×恒定） | 16K 日程，@4K 读数后手工停 | 干净锚点：修复后代码 + 128K warm-start 的 winners-off 基线（旧 A0@4K 0.8675/0.2656 受 EMA temp bug 污染） |
| S1 | `g2_s1_bass.yaml`（bass+退火） | 同上 | P3 包单独门 |
| S2 | `g2_s2_dn.yaml`（dn） | 同上 | P4 包单独门 |
| C1 | `g2_c1_two_16k.yaml`（bass+退火+dn） | 全 16K | 两方案合并续训（= B1' 减去三个弃用方案） |

所有臂：128K warm-start（finetune_weights=seal best.pt）、bs1/rank×4、lr 1e-5 cosine
16K、warmup 500、eval/checkpoint@4000（门臂@4K 读数后停）、workers 4 +
MALLOC_TRIM/MMAP/ARENA、无 copy-paste bank。

## 3. 预注册判读门（读数落地前锁定）

- **S0@4K**：segm AP ∈ [0.850, 0.895] 且 APs ∈ [0.24, 0.30]。
  出界 ⇒ 代码基线漂移或 warm-start 链路坏了，停止战役查因。
- **S1@4K vs S0**：硬门 = segm AP ≥ S0 − 0.5pt（掩码质量收益 4K 内未必显现，
  先证无害）；健康信号 = 带 Dice 损失分量有限且下行、其余损失不平坦。
  硬门失败 ⇒ P3 包退场，C1 改单 dn。
- **S2@4K vs S0**：硬门 = segm AP ≥ S0 − 0.5pt；健康信号 = dn_ce/dn_dice
  有限且下行。硬门失败 ⇒ P4 包退场，C1 改单 bass。
- **C1@16K 终局**：成功 = segm APs ≥ 0.28（超 seal 0.2698 +1pt）且
  segm AP ≥ 0.87（不牺牲大目标）。灰区 [0.27, 0.28) ⇒ 记录并考虑
  ACB/TREX+ 推理侧叠加或延长训练。
- 判读一律用 `metrics_log.jsonl` 的 `val/segm_AP*`（EMA 口径），bbox 同看。

## 4. 基础设施

- 启停：`ops/g2_launch.sh <cfg> [port]` / `ops/g2_stop.sh <agent_pid>`（杀整棵
  进程树；torchrun rank 各自成组，组杀无效——本次实测发现）。
- 看门狗：`ops/g2_watch.sh <agent_pid> <name>` 每 20s 记 JSONL；
  used% ≥88 持续 3 拍 ⇒ TERM→KILL 整树（90% 红线前动手）。
- 运行目录：`output/aps_20260913/g2_runs/<arm>_r1/`。
