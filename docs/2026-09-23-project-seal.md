# MAGFormer 项目封档 — 2026-09-23

> 用户指令（2026-09-23）：优化战役终止（30+ 优化模块全部无效），深度清理后封档。
> 本文档是封档后的唯一入口：终版定义、最终指标、战役终审、保留/删除清单、复现指南。

## 1. 终版定义（唯一保留的版本）

**Swin-T（RGB 塔）+ MobileNetV3-L（深度塔）双编码器 MagFormer**，v317 谱系
（DCCG 跨模态融合 + 置信门控 + DPE 深度位置编码 + AGPE + AIM 匹配 + stride-2 hires 头）。

| 资产 | 位置 |
|---|---|
| 代码 | 仓库根 `magformer/`（全部探索模块 config-gated 默认关，关=逐位基线，no-op 回归测试锁定） |
| 最终训练配置 | `configs/full_design_256k.yaml` |
| **128K 封存权重（基线）** | `output/aps_20260913/p5_runs/f1_seal_128k/`：`best.pt`（校准 EMA）/ `last.pt`（完整训练态，可 resume）/ 指标与校准记录；`seal_verify/`（CALIBRATED.md + 4 个校准配置） |
| 预训练编码器权重 | `pretrained_weights/`（swin_tiny_d2_format / mobilenetv3_large_100_depth / model_init） |
| 数据 | `magformer_datasets/20260318_1K_32254`（train 25654 图 / val 3276 图） |
| 环境 | conda env `magformer`（`environment.magformer.yml`，torch 2.5.1+cu124） |

## 2. 最终指标

| 口径 | segm AP | segm APs | bbox AP |
|---|---|---|---|
| **F1@128K 封存（校准 EMA，全 val 3276）** | **0.8744** | **0.2698** | 0.8596 |
| 同对塔 + 标准 M2F concat 头（300K 级训练） | 0.91 | **0.354** | — |

全家族历史读数表见 `docs/EXPLORATION_ROADMAP.md`；基线对比（cellpose/yolov8/maskrcnn/
msmformer/ucn/uoais 等）见 `docs/BASELINE_RESULTS.md` 与 `output/analysis/`。

**结构性结论**：本架构族 7 月收敛在 0.84-0.86 带（c0_corrected 0.8603 峰值），
F1 长训至 128K 达 0.8744/APs 0.2698。**唯一持续有效的两个杠杆 = 更换 RGB-D 编码器
（双塔 v317）与延长训练（128K）**；标准 concat 融合头在 mAP 与 AP_s 上均优于本族
全部融合设计。AP_s 理论上限（GT-oracle，E3 stride-2）= 0.768。

## 3. 优化战役终审（2026-09-13 → 2026-09-23，全部判负）

30+ 优化模块（含斗兽场七场胜出：BAS-CL+、ACB、HDA+、mCDN+、SCB+、MAL-CP+、TREX+
及更早的 R1 设计包、探针/种子、copy-paste、RDIs、蒸馏族、VC-SUDA 等）**无一产生
净正增量**。关键终审读数（全部 128K warm-start、16K 日程 @4K、GPUs 4-7、同协议）：

| 臂 | segm AP | APs | vs 锚点 | 结论 |
|---|---|---|---|---|
| winners-off 锚点 | 0.8704 | 0.2692 | — | 基线无漂移（vs seal 持平） |
| 五方案合并 B1'（历史） | 0.7011 | 0.0261 | −16.6 / −24pt | 灾难，F2 封锁 |
| bass 边界监督全形态 | 0.7510 | 0.0484 | −11.9 / −22pt | 灾难 |
| bass 纯采样形态（软标签关/λ=1/带Dice=0） | 0.7157 | 0.0353 | −15.5 / −23pt | 更灾难 ⇒ 毒在采样重分布+balanced_ce 豁免 |
| seed 种子拼接（单变量） | 0.8501 | 0.2332 | −2.0 / −3.6pt | 回归，SCB+ 定罪 |

- **B1' 归因案闭合**：主杀手 = bass（软标签实为缓冲，真毒 = 带采样重分布 + 带内豁免
  的边界梯度战争）；次级 = seed（64 槽 query 池扰动 > 覆盖收益）；copy-paste 洗清。
- **体制结论**：冻结基座上 4K 步微调无法移动 AP_s（噪声带 ±0.006，EVIDENCE §7 早已
  记录，本轮四臂独立复现）。正增量若存在也只能在长训/从头训头体制表达——
  已按用户指令停止探索。
- **DPE 验证（封档前指令）**：历史消融 `dpe_off_corrected` 0.8510 vs `c0_corrected`
  0.8603 = **+0.9pt 轻度有效**（EXPLORATION_ROADMAP.md 家族表），保留于终版。
- 证据链：`output/aps_20260913/EVIDENCE.md`（战役全程 §1-§18）+
  `docs/2026-09-22-twoproposal-continuation-plan.md`（终局四臂判读）+
  `docs/data/two_proposal_20260922/`（五份指标 jsonl）+ `docs/arena_p3p4/*_verdict.md`
  （七场裁决书，方案机制与设计思路的最终记录）。

## 4. 封档清理记录（2026-09-23 执行）

**已删除**（git 已跟踪部分可从历史恢复；未跟踪部分已物理删除）：
- `archive_20260906/`（53G / 15029 文件，v317 前工作区，唯一活引用 swin 权重已迁出）
- `experiments_archive/`（4 个前代实验族 + v317 源码残留；终版树已提升至仓库根）
- `output/`：g2_runs 四失败臂全部权重（17G）、diagnostics/experiments/upper_bound/
  eval/logs/cache（~15G）、过程日志与小证据目录；`p5_runs/full_design_256k/`
  仅留 metrics（最终权重即 f1_seal_128k）
- `configs/`：75 个迭代配置（v14-v317 / finetune_* / ablation / next_stage /
  g1/f2/两提案战役臂），仅留 base + full_design_256k + golden_recipe 模板
- `docs/`：35 份斗兽场提案、五路评审原始底稿、文献调研、refactor/security 过程文档、
  9 份过程报告；7 份裁决书与全部终局记录保留
- 旧根包（提升前）、.trash 死代码、MSDeformAttn egg-info/build 编译产物、
  tools/aps_diag + tools/trt 一次性诊断工具、基线集成测试（移入 baselines/）

**保留**：终版代码树、128K 封存权重、基线对比权重（`output/pretrained/` 1.7G，
被 baselines/ 引用）、编码器预训练权重、全部终局文档与指标证据、数据集。

## 5. 复现指南

```bash
conda activate magformer
# 训练（从编码器预训练开始；4 卡 GPUs 4-7，物理索引，勿设 CUDA_VISIBLE_DEVICES）
python -m torch.distributed.run --nproc_per_node=4 --master_port=29611 \
  tools/train.py --config configs/full_design_256k.yaml
# 评估（--eval-only）
python tools/train.py --config configs/full_design_256k.yaml --eval-only
# 从 128K 封存权重 warm-start 微调：model.finetune_weights 指 f1_seal_128k/best.pt
```

硬规则：只用 GPUs 4-7；251GB 主机同时仅一个 4-rank 束；系统内存 ≤90%
（运维脚本 `ops/launch_4rank.sh` / `ops/watch_host_memory.sh` / `ops/stop_bundle.sh`，
看门狗 88%×3 拍杀整棵进程树——torchrun rank 各自成进程组，须按树杀）。
