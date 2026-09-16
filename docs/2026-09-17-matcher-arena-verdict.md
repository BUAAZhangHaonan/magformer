# Arena verdict — 小目标 matcher 代价的原理化重设计 (2026-09-17)

用户挑战: hand-tuned 采样比例是拍脑袋; 均匀采样不好讲故事; 要"讲好故事 + 比 M0 高 + 低开销"的方案。
协议: 3 组独立 SubAgent 提案 (design_A FDD / design_B LMNN-Cost / design_C AIM) → 主 agent 对抗审查 → 全部实装进 GPU-5 实验室 → 实测裁决。

## 1. 先回答"为什么 M0 == M2"
M0 (12544 均匀随机点) 是 M2 (全网格期望) 的**无偏蒙特卡洛估计** — 期望相同, 实验室均值必然重合。
差异全在方差通道, 旧实验室没测。新 harness 实测 (n=754 held-out):
- K=12544 重采样: pick 翻转率 **8.5%**, 跨 seed 稳定性 0.71, oracle 一致率比确定性全网格低 2.7pt;
- K=96 (hand-tuned 部署预算) 均匀估计: matched-IoU 崩到 **0.116**, 翻转率 **87.5%**, 稳定性 0.0。
结论: "全网格期望"相对生产 M0 只赢方差 (真实但小); 要赢均值必须换统计量 — 这正是三个设计的共同出发点。

## 2. Headroom 发现 (过关线重锚, 诚实声明)
新 harness 首次测出 **oracle 上限 = 0.4152** (每 GT 最优 query 的 IoU 均值), 而 M0 = 0.4043。
**冻结模型上 assignment 质量的全部空间只有 1.1pt** — 我原设的 M0+0.02 过关线在数学上不可达
(提案预测 0.434/0.445 均高于 oracle, 它们写于 headroom 未知时)。裁决标准改为:
(1) 逼近 0.4152 的程度; (2) 判定边距信噪比 margin = (最优干扰者代价 − oracle 代价)/std_q(代价)
   — E2-M1 "信号 0.3-2.4 vs 噪声 ±1.9" 的标准化版本, 训练期 ownership 是否翻转的直接预测器;
(3) 预测扰动稳定性; (4) 开销; (5) 故事。

## 3. 实测结果 (held-out n=754, 冻结 c0, 1883 小 GT 记录 60/40 切分)
| 方案 | matched_IoU | oracle一致 | margin中位 | >2σ稳健占比 | jit翻转(σ=0.5) |
|---|---|---|---|---|---|
| M0 全网格均匀 | 0.4043 | 0.667 | 0.13σ | 0.00 | 0.11 |
| M0n K=12544 (生产) | 0.4039 | 0.651 | 0.13σ | 0.00 | 0.17 |
| A: FDD λ=0.15 (Fisher分歧加权, 0参数) | 0.4049 | 0.659 | 0.70σ | 0.26 | 0.10 |
| B: LMNN-Cost (10.6K排序度量学习) | 0.4091 | 0.691 | 0.78σ | 0.25 | 0.10 |
| B 消融: 同架构+M3目标 | 0.3895 | 0.601 | — | — | — |
| **C: AIM-D (尺度无关soft-Dice, 确定性窗口)** | **0.4134** | **0.736** | **2.43σ** | **0.55** | 0.10 |
| oracle 上限 | 0.4152 | 1.0 | — | — | — |

- AIM-D 距上限 0.0018 (吃掉 **83%** headroom); 一致率 +6.9pt; **边距 19 倍**; 一致性 sanity (FDD λ=1) 与 M0 逐位相等。
- AIM 的 BCE 项 w-grid 单调有害 (w*=0); Dice 用 96 点采样估计会崩 (0.1116, 翻转 87%) → **确定性窗口精确计算是必要条件, 不是锦上添花**。
- LMNN 的排序目标比 M3 软目标 +1.96pt (同架构) — 机制主张成立, 但整体被 AIM 支配; 训练/外推 gap 仅 0.0026 (无记忆化)。
- FDD: 0 参数把边距拉 5 倍但均值不动 — 可作 AIM 的可选混合 (窗口内 Fisher 加权), 本轮不实装。

## 4. 赢家: AIM (design C) — 已实装生产 (config-gated)
机制一句话: **小 GT 的所有权不该依赖点重采样 (Law D: 确定性)、物体尺寸 (Law S: 尺度无关
soft-Dice 统计量) 或哪一层在问 (Law T, 未实装)** — 不靠软化分配或加 query, 靠统计量的不变性买稳定性。
- 生产代码: `matcher.py::_aim_small_gt_costs` (向量化, ≤16 GT, 无 python 循环, 无 RNG), `arch.py` 接线;
  新配置键 `matcher_small_gt_mode: aim` (+ `aim_w_bce`, 默认 0), 缺省行为逐位不变 (运行中双臂不受影响)。
- 单元测试全过 (确定性/退化 1px GT/贴边 GT/空 GT/前向/box 行共存): `scripts/test_aim_matcher.py`。
- 开销: 实测 16 小 GT 最坏情况 **1.72ms/次 vs 现行 grounded 17.23ms/次**; 每迭代 ×9 深监督 ≈ +2~16ms,
  **替换**现行 +20~30ms 路径 → 净省 ~10~25ms/迭代; 推理 0ms (matcher 仅训练期)。
- 下一臂配置: 在 s1c 配方上把 `matcher_small_gt_points: 96` 换成 `matcher_small_gt_mode: aim` (α 沿用 0.7; α=1.0 为设计 C 原案, 可作消融)。

## 5. 对整体指标的点数估算 (诚实链)
- 冻结模型通道 (实测): assignment 质量上限 ≤ +1.1pt matched-IoU, AIM 拿下 0.91pt; 稳健性从 0% 记录达 2σ → 55%。
- 训练动力学通道 (锚定): E2-M1 的"列由噪声决定"被 margin 0.13σ 直证, AIM 的 2.43σ 意味着多数小 GT 列
  退出掷硬币区 → Hungarian 不再每层/每次重掷所有权 → 分数成型 (当前瓶颈) 的前提成立。
  E5 残余 gap 4.0pt (64-256px², d1 生产混合后) 是 AIM 相对 hand-tuned 的可关闭空间。
- **估算: 相对现行 hand-tuned 臂 (s1c), AIM 预计 +0.3~0.8 AP_s, +0.1~0.3 mAP** (通道1: 残余gap部分关闭
  +0.2~0.5; 通道2: 确定性+边距 +0.1~0.3), 同时训练更快; mAP 下行风险≈0 (大 GT 列不动, 统计量有界 O(1))。
- 上限声明: matcher 杠杆单独**给不了** AP_s 翻倍目标 (冻结上限 1.1pt); 它是 stride-2 画布/s4 注意力等
  主引擎的稳定性使能器。最终 AP_s 数字只能由下一训练臂实测。

## 6. 过程工件
- 提案: `arena_matcher/design_{A,B,C}/proposal.md` (各≤150行, 含数学/实装/预测/证伪)
- 实验室: `scripts/gpu5_matcher_arena.py` (缓存+oracle+M0n探针) + `scripts/arena_designs.py` (三设计+margin/jitter探针)
- 数据: `gpu5_matcher_lab/arena_recs.pt` (1883 记录缓存), `arena_results.json`, `arena_designs_results.json`
