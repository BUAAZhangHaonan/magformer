# Smoke Runner And Test Cleanup Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 清除 `scripts/experiments/` 与 `tests/` 中的全仓 smoke 资产，并建立不依赖 smoke 的 dry-run/常规验证链路。

**Architecture:** 删除所有 runner 中的 `SMOKE=0`、`--smoke`、smoke 输出根目录和 smoke 元数据逻辑；测试侧统一改写为普通 `--dry-run`、候选参数、路径规范化与 override 注入断言，不再依赖 smoke 分支。Lightdepth/F5 runner 优先清理，再清到全仓。

**Tech Stack:** bash, pytest, ripgrep.

---

## 涉及文件

- Modify: `scripts/experiments/*.sh` 中所有包含 `--smoke` 或 `SMOKE=0` 的 runner
- Modify: `tests/test_*runner*.py`
- Modify: `tests/test_*smoke*.py`

## 步骤

1. 先写或改失败测试，确保删掉 smoke 后仍能验证：
   - dry-run 命令内容
   - 输出目录 canonicalization
   - candidate/image-size/override 逻辑
   - batch/epoch/OOM fallback 逻辑
2. 运行一组代表性测试，确认因 smoke 断言失效而失败
3. 批量移除 runner 中的 smoke 分支与参数
4. 批量改写 tests，不再出现 `--smoke` 断言或 smoke 文件名
5. 运行：
   - `pytest -q` 针对所有曾含 smoke 断言的测试文件
   - `rg -n "smoke" scripts tests -S`
6. 阅读本子计划并确认 scripts/tests 下 smoke 已清零后提交

## 完成判据

- `rg -n "smoke" scripts tests -S` 结果为空
- 所有受影响的 runner 测试通过
- F5、Stage A、Stage B、revisit、trackp、tracks 等 runner 仍具备 dry-run 可验证性

## 提交信息

- `refactor: remove smoke runner code and tests`
