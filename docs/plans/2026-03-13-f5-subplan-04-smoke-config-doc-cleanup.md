# Smoke Config And Doc Cleanup Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 删除 repo-tracked 的 smoke 专用配置、实验文档、JSON 过程文件和相关残留引用，使当前工作树不再保留 smoke 资产。

**Architecture:** 在 runner/test 清理完成后，再删除 `configs/` 中的 smoke YAML、`docs/experiments/` 中的 smoke 文档与 JSON 过程文件，并同步清理 `scripts/analysis/cleanup_temp_artifacts.py`、代码规范说明或 runbook 中的 smoke 文本残留。历史留痕交给 git，不在当前工作树保留。

**Tech Stack:** apply_patch, ripgrep, pytest.

---

## 涉及文件

- Delete: `configs/*smoke*.yaml`
- Delete: `docs/experiments/*smoke*`
- Modify: `scripts/analysis/cleanup_temp_artifacts.py`
- Modify: 任何 `docs/`、`scripts/` 中仍提及 smoke 的 tracked 文件

## 步骤

1. 先用 `rg -n "smoke" configs docs scripts tests -S` 记录清理目标
2. 删除所有 smoke 专用 YAML、Markdown、JSON 过程文件
3. 改写仍残留 smoke 文本的辅助脚本与规范文档
4. 运行：
   - `rg -n "smoke" configs docs scripts tests -S`
   - 预期为空
5. 如有受影响的测试，补跑对应 pytest 文件
6. 阅读本子计划并确认 repo-tracked smoke 资产归零后提交

## 完成判据

- `configs/` 中没有任何 `*smoke*.yaml`
- `docs/experiments/` 中没有任何 smoke 过程文档或 JSON
- `rg -n "smoke" configs docs scripts tests -S` 结果为空

## 提交信息

- `refactor: remove smoke configs and docs`
