# F5 Full Runs And Reporting Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 跑完 6 个 F5 full experiments，刷新 suite summary、可视化、推理 benchmark、宽表和最新中文 handoff 文档，并将结论固定到仓库。

**Architecture:** 使用独立 F5 all-runner 在 `0831_1K / 1024 / 20 epochs` 协议下完成 6 个候选 full run。每个 candidate 完成后立即刷新 summary，全套完成后统一跑 `visualize_suite.py`、`benchmark_inference_suite.py`、`write_extended_metrics_table.py`，并把结论写入最新中文总结文档与 handoff 文档。

**Tech Stack:** bash, conda `magformer` environment, summarize/visualize/benchmark analysis scripts, Markdown.

---

## 涉及文件

- Run: `scripts/experiments/run_0831_1k_20ep_1024_lightdepth_stage_b_f5_all.sh`
- Modify/Create: `docs/experiments/2026-03-13-f5-prior-guided-cross-attention-final-summary.md`
- Modify: `docs/experiments/2026-03-12-lightdepth-plan-status-and-metrics.md`
- Modify: `docs/plans/2026-03-13-project-status-and-next-steps.md`
- Modify: `docs/experiments/2026-03-13-extended-metrics-table.md`

## 步骤

1. 运行 F5 all-runner，确保 6 个 candidate 全部完成
2. 每个 candidate 结束后刷新 F5 suite summary
3. 全部完成后执行：
   - `python scripts/experiments/summarize_suite.py --output-root output/experiments/0831_1k_20ep_1024_lightdepth_stage_b_f5 --write`
   - `conda run -n magformer python scripts/experiments/visualize_suite.py ...`
   - `python scripts/analysis/benchmark_inference_suite.py ...`
   - `python scripts/analysis/write_extended_metrics_table.py ...` 合并 `depth_revisit`、`stage_a`、`stage_b`、`stage_b_f5`
4. 写 `2026-03-13-f5-prior-guided-cross-attention-final-summary.md`
5. 更新计划状态文档、最新 handoff 与扩展宽表文档
6. 运行全量 `pytest -q`
7. 阅读本子计划，核对 6 个候选目录、summary、宽表、中文结论三者一致后提交

## 完成判据

- 新 suite root 下 6 个 F5 模型目录完整
- 每个候选至少有 `metrics.cocoeval.json`、`params_trainable.txt`、`wall_time_sec.txt`
- benchmark 完成后有 `inference_speed.json`
- 文档、summary、宽表中的 model id、AP、wall time、peak memory 一致
- 最终 `pytest -q` 全绿

## 提交信息

- `docs: publish F5 experiment results and handoff updates`
