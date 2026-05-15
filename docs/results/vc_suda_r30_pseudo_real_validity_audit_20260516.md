# VC-SUDA R30 pseudo_real/AP-min 数据有效性审计

日期：2026-05-16

## 范围

- Host: `WS-4029GP-TRT`
- Repo: `/home/hdd3/zhanghaonan/magformer`
- Dataset root: `magformer_datasets/pseudo_real_512`
- Annotation files: `instances_target_labeled.json`, `instances_target_unlabeled.json`, `instances_val.json`
- Diagnostics: `output/diagnostics/r30_pseudo_real_validity_20260516`
- 约束：未训练，未修改代码；只写文档和轻量诊断产物。

## Annotation 统计

| split | images | annotations | inst min | inst p50 | inst p90 | inst max | empty images | bbox abnormal | area abnormal | category abnormal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `target_labeled` | `25` | `1697` | `50` | `50` | `100.00` | `100` | `0` | `0` | `0` | `0` |
| `target_unlabeled` | `200` | `11750` | `25` | `50.00` | `100.00` | `100` | `0` | `0` | `0` | `0` |
| `val` | `28` | `1892` | `50` | `50.00` | `100.00` | `100` | `0` | `0` | `0` | `0` |

类别集合为单类 `component`，有效 `category_id` 为 `{1}`。三个 split 均未发现类别异常。

## RGB/depth 抽样检查

每个 split 固定随机抽样 `20` 张，随机种子 `20260516`。

| split | sampled | missing RGB | missing depth | RGB size mismatch | depth shape mismatch | depth valid min/p50/p90/max | depth NaN | depth Inf |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| `target_labeled` | `20` | `0` | `0` | `0` | `0` | `0.9489` / `0.9499` / `0.9503` / `0.9504` | `0` | `0` |
| `target_unlabeled` | `20` | `0` | `0` | `0` | `0` | `0.9491` / `0.9499` / `0.9502` / `0.9507` | `0` | `0` |
| `val` | `20` | `0` | `0` | `0` | `0` | `0.9494` / `0.9501` / `0.9506` / `0.9507` | `0` | `0` |

抽样 depth 形状均为 `512x512`。完整抽样明细见 `audit_summary.json`。

## 实例数最低和最高样本

### Lowest 20

| split | image_id | instances | file_name |
| --- | ---: | ---: | --- |
| `target_unlabeled` | `116` | `25` | `27,5x31x18x26_25_scene_000000_000328_v0.png` |
| `target_unlabeled` | `157` | `25` | `27,5x31x18x26_25_scene_000001_001094_v0.png` |
| `target_unlabeled` | `212` | `25` | `27,5x31x18x26_25_scene_000003_001035_v0.png` |
| `target_unlabeled` | `182` | `25` | `37,5x41x26x37_25_scene_000003_000776_v1.png` |
| `target_unlabeled` | `150` | `25` | `37,5x41x26x37_25_scene_000004_000450_v1.png` |
| `target_unlabeled` | `113` | `25` | `37,5x41x26x37_25_scene_000006_000789_v0.png` |
| `target_unlabeled` | `127` | `25` | `37,5x41x26x37_25_scene_000008_000878_v0.png` |
| `target_unlabeled` | `162` | `25` | `615008160321_25_scene_000005_000728_v0.png` |
| `target_unlabeled` | `49` | `25` | `679303124022_25_scene_000004_000693_v1.png` |
| `target_unlabeled` | `114` | `25` | `687106149022_25_scene_000003_000131_v0.png` |
| `target_unlabeled` | `50` | `25` | `687106149022_25_scene_000003_000132_v1.png` |
| `target_unlabeled` | `35` | `25` | `74437625201002_25_scene_000003_000571_v0.png` |
| `target_unlabeled` | `207` | `25` | `74437625201002_25_scene_000005_001254_v1.png` |
| `target_unlabeled` | `46` | `25` | `74437625201002_25_scene_000006_001238_v1.png` |
| `target_unlabeled` | `145` | `25` | `74437625201002_25_scene_000007_000647_v1.png` |
| `target_unlabeled` | `193` | `25` | `C12401832E402A_25_scene_000002_001255_v1.png` |
| `target_unlabeled` | `201` | `25` | `C12401832E402A_25_scene_000006_000014_v1.png` |
| `target_unlabeled` | `224` | `25` | `IndLQS_4012_25_scene_000000_000458_v0.png` |
| `target_unlabeled` | `69` | `25` | `IndLQS_4012_25_scene_000004_001091_v0.png` |
| `target_unlabeled` | `98` | `25` | `IndLQS_4012_25_scene_000007_000930_v0.png` |

### Highest 20

| split | image_id | instances | file_name |
| --- | ---: | ---: | --- |
| `target_labeled` | `11` | `100` | `490107670612_100_scene_000008_000003_v1.png` |
| `target_labeled` | `1` | `100` | `615002138421_100_scene_000001_000714_v1.png` |
| `target_labeled` | `12` | `100` | `618025231421_100_scene_000006_000389_v0.png` |
| `target_labeled` | `24` | `100` | `658410821024_100_scene_000007_000031_v0.png` |
| `target_labeled` | `13` | `100` | `686106183822_100_scene_000006_000169_v0.png` |
| `target_labeled` | `23` | `100` | `687106149022_100_scene_000009_001248_v1.png` |
| `target_labeled` | `9` | `100` | `SOIC127P1030X265-18N_100_scene_000007_000400_v0.png` |
| `target_labeled` | `16` | `100` | `SOIC127P1030X265-18N_100_scene_000008_001177_v0.png` |
| `target_unlabeled` | `99` | `100` | `1779205141_100_scene_000003_000324_v0.png` |
| `target_unlabeled` | `16` | `100` | `490107670612_100_scene_000004_000825_v1.png` |
| `target_unlabeled` | `83` | `100` | `490107670612_100_scene_000008_000002_v0.png` |
| `target_unlabeled` | `39` | `100` | `615002138421_100_scene_000006_000677_v1.png` |
| `target_unlabeled` | `183` | `100` | `618025231421_100_scene_000004_000256_v1.png` |
| `target_unlabeled` | `64` | `100` | `618025231421_100_scene_000005_001166_v1.png` |
| `target_unlabeled` | `27` | `100` | `618025231421_100_scene_000006_000390_v1.png` |
| `target_unlabeled` | `154` | `100` | `658410821024_100_scene_000005_000947_v1.png` |
| `target_unlabeled` | `219` | `100` | `686106183822_100_scene_000003_001210_v1.png` |
| `target_unlabeled` | `223` | `100` | `687106149022_100_scene_000000_000286_v0.png` |
| `target_unlabeled` | `48` | `100` | `68710814522_100_scene_000007_000044_v1.png` |
| `target_unlabeled` | `111` | `100` | `6_100_scene_000000_000959_v0.png` |

最低实例数样本仍有 `25` 个标注实例；没有发现空图，也没有发现“没有有效电子元件”的图。

## Contact sheet

- `lowest_instances`: `output/diagnostics/r30_pseudo_real_validity_20260516/contact_sheet_lowest_instances.png`
- `random_samples`: `output/diagnostics/r30_pseudo_real_validity_20260516/contact_sheet_random_samples.png`

## 结论

R30 pseudo_real/AP-min 使用的 target_labeled、target_unlabeled、val 三个 split 在本次审计中通过基础有效性检查。annotation 中没有空图、异常 bbox、异常 area 或类别异常；每 split 20 张 RGB/depth 抽样均存在且尺寸匹配，depth 抽样没有 NaN/Inf，非零有效比例约为 `0.9489` 到 `0.9507`。实例数最低的样本仍有 `25` 个有效元件标注，因此未看到无有效电子元件图。
