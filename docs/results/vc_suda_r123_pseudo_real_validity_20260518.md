# VC-SUDA R123 pseudo_real_512 数据有效性审计

日期：2026-05-18

## 范围

- Host: `4029`
- Repo: `/home/hdd3/zhanghaonan/magformer`
- Dataset root: `magformer_datasets/pseudo_real_512`
- Diagnostics: `output/diagnostics/r123_pseudo_real_validity_20260518/`
- 约束：CPU-only；未使用 GPU；未启动训练；未创建新分支；未使用 `.worktree`。

## 依赖检查

可用 Python 为 `/usr/bin/python3`，版本 `3.10.12`。

| dependency | status |
| --- | --- |
| `PIL` | OK |
| `pycocotools` | MISSING: `No module named 'pycocotools'` |

因此，本次没有生成 mask 边界/半透明 mask contact sheet。按任务约束，没有写任何简化 fallback mask 渲染逻辑。

依赖错误记录：

- `output/diagnostics/r123_pseudo_real_validity_20260518/DEPENDENCY_ERROR.txt`

## 统计方法

- `mask area` 使用 COCO annotation 的 `area` 字段。
- `bbox area` 使用 `bbox[2] * bbox[3]`。
- 小目标比例使用 COCO 常用定义：`area < 32^2`，也就是 `< 1024 px^2`。
- 分位数为排序后 nearest rounded index。

完整机器可读结果：

- `output/diagnostics/r123_pseudo_real_validity_20260518/annotation_stats.json`
- `output/diagnostics/r123_pseudo_real_validity_20260518/selected_image_ids.json`

## Annotation 统计

| annotation | images | anns | empty images | inst min/p10/p50/p90/max | mask area min/p10/p50/p90/max | bbox area min/p10/p50/p90/max | small ratio |
| --- | ---: | ---: | ---: | --- | --- | --- | ---: |
| `instances_target_unlabeled.json` | 200 | 11750 | 0 | 25 / 25 / 50 / 100 / 100 | 1 / 101 / 452 / 653 / 883 | 1 / 414 / 744 / 1050 / 1480 | 1.0000 |
| `instances_target_labeled.json` | 25 | 1697 | 0 | 50 / 50 / 50 / 100 / 100 | 2 / 90 / 401 / 641 / 832 | 6 / 384 / 726 / 1026 / 1406 | 1.0000 |
| `instances_val.json` | 28 | 1892 | 0 | 50 / 50 / 50 / 100 / 100 | 1 / 82 / 399 / 629 / 831 | 1 / 384 / 725 / 1023 / 1369 | 1.0000 |
| `instances_target_labeled_r114_balanced_plus125.json` | 150 | 9083 | 0 | 25 / 50 / 50 / 100 / 100 | 1 / 98 / 441 / 651 / 883 | 1 / 405 / 736 / 1044 / 1480 | 1.0000 |
| `instances_target_unlabeled_r114_balanced_minus125.json` | 75 | 4364 | 0 | 25 / 25 / 50 / 100 / 100 | 1 / 102 / 459 / 656 / 859 | 1 / 416 / 744 / 1054 / 1443 | 1.0000 |

## 结构一致性检查

| annotation | zero/negative mask area anns | zero/negative bbox area anns | missing segmentation anns | missing image refs | missing image files | categories |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `instances_target_unlabeled.json` | 0 | 0 | 0 | 0 | 0 | 1 |
| `instances_target_labeled.json` | 0 | 0 | 0 | 0 | 0 | 1 |
| `instances_val.json` | 0 | 0 | 0 | 0 | 0 | 1 |
| `instances_target_labeled_r114_balanced_plus125.json` | 0 | 0 | 0 | 0 | 0 | 1 |
| `instances_target_unlabeled_r114_balanced_minus125.json` | 0 | 0 | 0 | 0 | 0 | 1 |

## Contact sheet 状态

计划抽样集合已经写入：

- random 24: `selected_image_ids.json` 的 `random_24`
- instance count 最低 24: `selected_image_ids.json` 的 `lowest_instance_count_24`
- tiny/area 最小样本所在图 24: `selected_image_ids.json` 的 `tiny_min_area_image_24`
- high-density 24: `selected_image_ids.json` 的 `high_density_24`

实际 contact sheet 未生成。原因是当前 Python 环境缺少 `pycocotools`，无法按要求解码 mask 并叠加 mask 边界/半透明 mask。这里没有使用 bbox-only、polygon-only 或其他简化替代图。

## 结论

从 annotation 层面看，`pseudo_real_512` 这批模拟真实/后 20% 数据不是空场景，也不是无有效电子元件数据：

- 5 个 annotation 的 `empty images` 全部为 `0`。
- 每图 instance count 的最小值为 `25` 或 `50`，没有 0-instance 图。
- 所有 annotation 都有 segmentation；没有缺 image ref；没有缺图像文件路径；没有非正 mask area 或非正 bbox area。

主要风险是：图像级人工可视化复核没有完成。缺少 `pycocotools` 后，按任务约束不能生成 mask overlay contact sheet，所以本次结论只覆盖 annotation 结构与面积字段，不覆盖人工肉眼查看 mask 叠加效果。
