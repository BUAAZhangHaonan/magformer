# VC-SUDA R123 pseudo_real_512 数据有效性审计

日期：2026-05-18

## 范围

- Host: `4029`
- Repo: `/home/hdd3/zhanghaonan/magformer`
- Dataset root: `magformer_datasets/pseudo_real_512`
- Diagnostics: `output/diagnostics/r123_pseudo_real_validity_20260518/`
- 约束：CPU-only；未使用 GPU；未启动训练；未创建新分支；未使用 `.worktree`。

## 依赖检查

本次补齐视觉复核时使用指定 conda Python：

- `/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python`

| dependency | status |
| --- | --- |
| `PIL` | OK (`12.1.1`) |
| `pycocotools` | OK |

上一轮 `/usr/bin/python3` 缺少 `pycocotools`，所以没有生成 contact sheet。本次没有使用 bbox-only fallback，所有可视化 mask 都由 COCO `segmentation` 解码得到。

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

本次已生成 mask overlay contact sheet，全部使用真实 image 文件，并用 COCO `segmentation` 解码得到半透明 mask 填充和 mask 边界：

| sample type | contact sheet | visual review |
| --- | --- | --- |
| random 24 | `output/diagnostics/r123_pseudo_real_validity_20260518/contact_random24.png` | 24 张图均可打开，目标和 mask 覆盖在元件区域内，未见空图或无效图。 |
| instance count 最低 24 | `output/diagnostics/r123_pseudo_real_validity_20260518/contact_low_count24.png` | 低计数组样本每图仍有 25-49 个实例，mask 和目标主体对齐，未见空图或无效图。 |
| tiny/area 最小样本所在图 24 | `output/diagnostics/r123_pseudo_real_validity_20260518/contact_tiny24.png` | tiny 样本可视化显示密集小目标，mask 边界可见，未见空图或无效图。 |
| high-density 24 | `output/diagnostics/r123_pseudo_real_validity_20260518/contact_high_density24.png` | 高密度样本每图 100 个实例，mask 覆盖密集但仍落在目标区域，未见空图或无效图。 |

生成摘要：

- `output/diagnostics/r123_pseudo_real_validity_20260518/contact_sheet_generation_summary.json`

疑似空图/无效图：无。

## 结论

从 annotation 层面看，`pseudo_real_512` 这批模拟真实/后 20% 数据不是空场景，也不是无有效电子元件数据：

- 5 个 annotation 的 `empty images` 全部为 `0`。
- 每图 instance count 的最小值为 `25` 或 `50`，没有 0-instance 图。
- 所有 annotation 都有 segmentation；没有缺 image ref；没有缺图像文件路径；没有非正 mask area 或非正 bbox area。

补齐图像级视觉复核后，4 组 contact sheet 都能看到真实图像上的 segmentation mask 半透明填充和边界。肉眼复核未发现空图、无效图、整图无目标或明显 mask 错位样本。
