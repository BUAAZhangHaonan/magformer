# 架构重构提案（2026-09-06，未实施）

## 现状问题
1. experiments_archive/*/source/ 是三份近乎完整的仓库快照（cdti / E18 / arm_ce 系），单文件级 diff 约 60 处——代码漂移已经造成过实际事故（c0 补测时 E18 树缺失 depth_pe 模块，8 月中旬后该模块被删除，导致差点用错误架构评测）。
2. baselines/ 内 30 个脚本 + vendored 库混居，RAM-diet 修复散落在运行脚本里。
3. 配置体系三套并存：根 configs/、scripts/E*/config.yaml、audit yaml，同一架构开关（dpe/agpe/point_refiner）在不同树里默认值不同。

## 建议（按优先级）
1. **单一源码树**：以 cdti 树（最新完整功能）为基础合入主包 magformer/，用 git 三方合并逐文件处理 60 处 diff；合并后删除 experiments_archive/*/source/ 快照，只保留 runner 脚本与配置。
2. **配置收敛**：所有架构开关进 schema.py 单一定义，实验 yaml 只允许覆盖 solver/data；默认值与 v317（c0）对齐。
3. **baselines 拆分**：vendored 库（ultralytics/cellpose/stardist/uois/msmformer）移 third_party/ 并 gitignore；运行脚本按"官方库 + 我们的 RAM 适配层"两文件制拆分。
4. **权重管理**：archive_20260906 稳定后精编为 checkpoints/（描述性命名 + sha256 清单），大文件继续留 git 外。

## 迁移门槛
合并源码树后必须跑的回归门：1566 lightdepth 三变种复评（历史数字对齐 ±0.1）+ c0 全量复评 0.7565 复现。
