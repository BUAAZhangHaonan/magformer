# Runner And Test Cleanup Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 清除 `scripts/experiments/` 与 `tests/` 中的全仓临时缩减预算资产，并建立统一的 dry-run/常规验证链路。

**Architecture:** 删除所有 runner 中的临时缩减预算参数、专用输出根目录和相关元数据逻辑；测试侧统一改写为普通 `--dry-run`、候选参数、路径规范化与 override 注入断言，不再依赖快速验证分支。Lightdepth/F5 runner 优先清理，再清到全仓。

**Tech Stack:** bash, pytest, ripgrep.

---

## 涉及文件

- Modify: `scripts/experiments/*.sh` 中所有包含临时缩减预算逻辑的 runner
- Modify: `tests/test_*runner*.py`
- Modify: 历史上的快速验证测试文件

## 步骤

1. 先写或改失败测试，确保删掉临时缩减预算分支后仍能验证：
   - dry-run 命令内容
   - 输出目录 canonicalization
   - candidate/image-size/override 逻辑
   - batch/epoch/OOM fallback 逻辑
2. 运行一组代表性测试，确认因旧断言失效而失败
3. 批量移除 runner 中的临时缩减预算分支与参数
4. 批量改写 tests，不再出现旧分支断言或旧文件命名
5. 运行：
   - `pytest -q` 针对所有受影响的 runner 测试文件
   - 对 `scripts tests` 做旧临时预算关键字扫描
6. 阅读本子计划并确认 scripts/tests 下旧临时预算资产已清零后提交

## 完成判据

- 关键字扫描结果为空
- 所有受影响的 runner 测试通过
- F5、Stage A、Stage B、revisit、trackp、tracks 等 runner 仍具备 dry-run 可验证性

## 提交信息

- `refactor: remove temporary-budget runner code and tests`
