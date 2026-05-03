# Config And Doc Cleanup Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 删除 repo-tracked 的临时缩减预算专用配置、实验文档、JSON 过程文件和相关残留引用，使当前工作树不再保留这类资产。

**Architecture:** 在 runner/test 清理完成后，再删除 `configs/` 中的临时缩减预算 YAML、`docs/experiments/` 中的相关文档与 JSON 过程文件，并同步清理 `scripts/analysis/cleanup_temp_artifacts.py`、代码规范说明或 runbook 中的旧文本残留。历史留痕交给 git，不在当前工作树保留。

**Tech Stack:** apply_patch, ripgrep, pytest.

---

## 涉及文件

- Delete: 历史临时缩减预算 YAML
- Delete: 历史临时缩减预算 Markdown / JSON
- Modify: `scripts/analysis/cleanup_temp_artifacts.py`
- Modify: 任何 `docs/`、`scripts/` 中仍提及旧临时预算资产的 tracked 文件

## 步骤

1. 先用关键字扫描记录清理目标
2. 删除所有临时缩减预算专用 YAML、Markdown、JSON 过程文件
3. 改写仍残留旧术语的辅助脚本与规范文档
4. 运行：
   - 对 `configs docs scripts tests` 做旧临时预算关键字扫描
   - 预期为空
5. 如有受影响的测试，补跑对应 pytest 文件
6. 阅读本子计划并确认 repo-tracked 旧临时预算资产归零后提交

## 完成判据

- `configs/` 中没有任何临时缩减预算 YAML
- `docs/experiments/` 中没有任何临时缩减预算过程文档或 JSON
- 关键字扫描结果为空

## 提交信息

- `refactor: remove temporary-budget configs and docs`
