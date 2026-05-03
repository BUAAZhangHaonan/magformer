# F5 Experiment Track Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 为 F5 新增独立的 Stage B experiment track、6 个固定 variant、对照锚点接线和 dry-run 可复现测试。

**Architecture:** 复用现有 MobileNetV3 与 ConvNeXt-lite base config，不复制 6 份 YAML。通过新的 bash runner 注入 `mode=prior_guided_cross_attn` 与 prior 组合，所有 full run 输出写入 `output/experiments/0831_1k_20ep_1024_lightdepth_stage_b_f5`，并在 suite root 中挂入 Stage A 胜出模型和 RGB-only 基线的软链接作为比较锚点。

**Tech Stack:** bash, pytest, repo-local render/summary scripts.

---

## 涉及文件

- Create: `scripts/experiments/run_0831_1k_20ep_1024_lightdepth_stage_b_f5_magformer.sh`
- Create: `scripts/experiments/run_0831_1k_20ep_1024_lightdepth_stage_b_f5_all.sh`
- Modify: `scripts/analysis/write_extended_metrics_table.py`
- Create: `tests/test_lightdepth_stage_b_f5_runner_metadata_cmd_reproducible.py`
- Create: `tests/test_lightdepth_stage_b_f5_all_script.py`
- Optional modify: `tests/test_lightdepth_configs.py`

## 步骤

1. 先写失败测试，验证：
   - F5 runner dry-run 能输出正确 variant、base config、mode、priors
   - F5 all-runner 列出全部 6 个 variant 与 3 个锚点
   - 默认宽表 summary 列表可包含新 F5 suite
2. 运行：
   - `pytest -q tests/test_lightdepth_stage_b_f5_runner_metadata_cmd_reproducible.py tests/test_lightdepth_stage_b_f5_all_script.py`
   - 预期失败
3. 实现两个新 runner，要求：
   - 不含任何临时缩减预算分支
   - 含 OOM fallback
   - 每个 candidate full run 完成后刷新 summary
4. 如默认宽表入口需要更新，则在本阶段一并接线，但不写结果文档
5. 运行：
   - `pytest -q tests/test_lightdepth_stage_b_f5_runner_metadata_cmd_reproducible.py tests/test_lightdepth_stage_b_f5_all_script.py tests/test_lightdepth_configs.py`
6. 阅读本子计划，确认 variant、路径、锚点、输出目录完全一致后提交

## 完成判据

- F5 独立 runner 和 all-runner 存在且只支持 full run / dry-run
- 6 个固定 variant 名称无漂移
- 新 suite root 与锚点命名固定
- 默认宽表入口能够感知 F5 summary

## 提交信息

- `feat: add F5 lightdepth experiment track`
