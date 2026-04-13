# 2026-03-24 Post-Suite Audit

## Findings

1. `msmformer` 的全零指标是本次训练配置塌陷，不是汇总脚本误报。
   - 训练期 `metrics.json` 与最终 `metrics.cocoeval.json` 都是全零。
   - 导出结果里存在大量 `score=0.0`、`bbox=[0,0,0,0]`、空 mask 的退化预测。
   - 关键风险点是 scratch 训练仍沿用了 `MODEL.BACKBONE.FREEZE_AT: 2`，与 vendor 基线 `FREEZE_AT: 0` 不一致。

2. `magformer/depth_sanity.py` 位置合理，不属于误放脚本。
   - 它是训练前 depth 预检模块，被 `tools/train.py` 和对应测试直接引用。
   - 如果未来要重构，更适合移动到 `magformer/training/` 或 `magformer/utils/`，而不是移出 Python 包。

3. `20260318_1K_1566` 的真实数据集统计参数已经存在，且与仓库中的参考 stats 一致。
   - 正式 cache 位于 `output/cache/dataset_stats/20260318_1k_1566_e60cae5edb/`。
   - RGB: `mean_rgb=[116.70696,114.12934,113.43611]`，`std_rgb=[27.95885,28.67968,28.81314]`。
   - Depth: `p1=1.0015300512`，`p99=2.0956230164`。

4. 新数据集归一化的消费链路并不完全一致。
   - 已消费新 stats: `magformer`、`mgm_mask2former`、`mask2former`、`maskrcnn`、`msmformer`、`uoais`。
   - 未消费 dataset-specific normalization: `yolov8_*`、`unet_*`。
   - 这属于当前训练栈的真实不一致，不是文档问题。

5. 工作区存在可清理的运行时垃圾。
   - 根目录 YOLO 权重原先混在仓库顶层。
   - `output/cache/dataset_stats/` 中混入了 pytest 临时数据集 cache。
   - `.pytest_cache/`、`magformer.egg-info/`、`__pycache__/` 为可删除生成物。

## Cleanup Applied

- 根目录 YOLO 权重已清空，并统一归入 `output/pretrained/`。
- pytest/fake dataset 产生的 stats cache 已从 `output/cache/dataset_stats/` 移除。
- `.pytest_cache/`、`magformer.egg-info/`、`__pycache__/` 已清理。
- `scripts/experiments/run_20260321_ddp_smoke_canary.sh` 已改为优先使用 `output/pretrained/yolov8n-seg.pt`。
- `scripts/analysis/write_extended_metrics_table.py` 已补齐：
  - `bbox/AP50`
  - `bbox/AP75`
  - `P@50`
  - `R@50`
  - `F1@50`

## Metric Note

- `P@50`、`R@50`、`F1@50` 基于 COCO `segm` evaluator 的 IoU=0.50 PR 曲线计算。
- 表中取最大 F1 对应的 operating point，并对类别维度做平均。

## Residual Risk

- `msmformer` 当前这次 suite 结果仍应视为真实失败样本，不能离线“修成”有效 AP。
- `yolov8_*` 与 `unet_*` 若要真正对齐 dataset-specific normalization，需要后续改训练数据管线并重新训练。
