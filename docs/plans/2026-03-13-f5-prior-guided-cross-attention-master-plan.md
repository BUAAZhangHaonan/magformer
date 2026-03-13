# F5 Prior-Guided Cross-Attention Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在当前 `master` 工作区内把 `prior_guided_cross_attn` 做成独立、生产级的轻量 RGB-D 融合模式，并完成独立 F5 实验轨、全仓 smoke 资产清理、结果汇总与 handoff 更新。

**Architecture:** 保留现有 `cross_attn` 作为历史对照，在 `magformer/models/magformer/fusion.py` 中新增独立的 `prior_guided_cross_attn` 路径，使用显式 prior-conditioned K/V。实验侧新增独立 Stage B F5 track，对 `ConvNeXt-lite` 和 `MobileNetV3` 两条轻量 backbone 各跑 3 组 prior 组合。仓库清理分两段执行，先替换 runner/test 验证链路，再删除全仓 smoke 配置与文档，最后统一做 full experiment、benchmark、宽表与 handoff 收口。

**Tech Stack:** PyTorch, Detectron2-style training utilities, bash experiment runners, pytest, COCOeval-based postprocess, repo-local analysis scripts.

---

## 固定决策

- 新融合模式名固定为 `prior_guided_cross_attn`
- F5 v1 只做显式 prior-conditioned K/V，不引入 dense attention bias，不新增 public config 字段
- 候选包固定为两条 backbone 共 6 个 full run：
  - `mobilenetv3_priorguidedcrossattn_edge`
  - `mobilenetv3_priorguidedcrossattn_edge_validhole`
  - `mobilenetv3_priorguidedcrossattn_edge_validhole_variance`
  - `convnextlite_priorguidedcrossattn_edge`
  - `convnextlite_priorguidedcrossattn_edge_validhole`
  - `convnextlite_priorguidedcrossattn_edge_validhole_variance`
- 新实验根目录固定为 `output/experiments/0831_1k_20ep_1024_lightdepth_stage_b_f5`
- 比较锚点固定为：
  - `magformer_nodpth_ref`
  - `magformer_lightdepth_convnextlite_spatialgate_edge_validhole`
  - `magformer_lightdepth_mobilenetv3_spatialgate_edge_validhole`
- 全仓 `smoke` 资产清理为硬约束，最终必须满足：
  - `rg -n "smoke" configs docs scripts tests -S`
  - 结果为空

## 子计划顺序

1. 落盘本总计划与 5 份执行子计划文档
2. 按 TDD 实现 `prior_guided_cross_attn` 核心融合逻辑与单测
3. 接入 F5 独立 runner、track summary 入口与配套测试
4. 清理 `scripts/experiments/` 与 `tests/` 中的全仓 smoke 资产
5. 清理 `configs/`、`docs/` 和相关辅助脚本中的全仓 smoke 资产
6. 运行 6 个 F5 full experiments，刷新 summary、benchmark、宽表和中文 handoff 文档

## 每阶段统一要求

- 所有代码改动必须先写失败测试，再实现，再跑通过
- 每个子计划结束前都要重新阅读对应子计划文档，逐项核对完成判据
- 每个子计划单独提交并推送，不堆叠到下一阶段
- 不保留一次性脚本、过程草稿、临时配置或 repo-tracked 垃圾文件
- 最终提交前执行全量 `pytest -q`

## 子计划文档

- `docs/plans/2026-03-13-f5-subplan-01-fusion-core.md`
- `docs/plans/2026-03-13-f5-subplan-02-experiment-track.md`
- `docs/plans/2026-03-13-f5-subplan-03-smoke-runner-test-cleanup.md`
- `docs/plans/2026-03-13-f5-subplan-04-smoke-config-doc-cleanup.md`
- `docs/plans/2026-03-13-f5-subplan-05-full-runs-and-reporting.md`
