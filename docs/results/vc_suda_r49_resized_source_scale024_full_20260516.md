# VC-SUDA R49 Resized Source Scale 0.24 Full Build, 2026-05-16

R49 materializes the full resized 32K source COCO RGB-D dataset at scale `0.24`. No training and no model eval were run.

## Output

- dataset root: `magformer_datasets/20260318_1K_32254_scale024`
- annotation: `magformer_datasets/20260318_1K_32254_scale024/annotations/instances_train_scale024.json`
- diagnostics: `output/diagnostics/r49_resized_source_scale024_full_20260516`
- validation summary: `output/diagnostics/r49_resized_source_scale024_full_20260516/validation_summary.json`
- protocol summary: `output/diagnostics/r49_resized_source_scale024_full_20260516/protocol_summary.json`

Build command:

```bash
/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python tools/build_resized_coco_rgbd_dataset.py \
  --source-root magformer_datasets/20260318_1K_32254 \
  --source-ann annotations/instances_train.json \
  --output-root magformer_datasets/20260318_1K_32254_scale024 \
  --output-ann annotations/instances_train_scale024.json \
  --scale 0.24 \
  --split train \
  --summary-json output/diagnostics/r49_resized_source_scale024_full_20260516/summary.json
```

## Build Summary

| metric | value |
|---|---:|
| images | 25,654 |
| annotations | 1,398,531 |
| RGB files | 25,654 |
| depth `.npy` files | 25,654 |
| RGB missing | 0 |
| depth missing | 0 |
| mask area p50 | 436.1472 |
| area<=256 ratio | 0.2271604991 |
| directory size | `22G` |
| `/home/hdd3` usage after build | `48%` (`7.3T` available) |
| build wall time | `1:09:11` |
| build peak RSS | `129,212,564 KB` |

Resize policy stayed the same as R48: RGB bilinear, depth linear with float values preserved, polygon geometry scaled numerically, and RLE masks resized with nearest-neighbor before bbox/area recomputation.

## Protocol Summary

| name | images | instances | inst/img p50 | inst/img p90 | mask area p50 | area<=256 | density 25-30 | density 46-60 | density >90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| r49_full_scale024 | 25,654 | 1,398,531 | 50.000 | 99.000 | 436.1472 | 0.2271604991 | 4,145 | 15,258 | 5,192 |

Target split basename overlap:

| split | overlap |
|---|---:|
| target_labeled | 0 |
| target_unlabeled | 0 |
| val | 0 |
| total | 0 |

## Validation

The full annotation validation was streamed instead of loading the large JSON all at once. It checked annotation counts, image/category refs, bbox format, positive area, duplicate image ids, target basename overlap, and actual file counts.

| gate | result |
|---|---|
| validation summary readable | pass |
| protocol summary readable | pass |
| RGB missing=0 | pass |
| depth missing=0 | pass |
| target basename overlap=0 | pass |
| bbox/area/image_id/category_id validation | pass |
| p50 in target-scale band 339-678 | pass |
| area<=256 close to target | pass |
| CocoRgbdDataset random 5 readable | pass |
| overall data gate | pass |

Field validation details:

| check | value |
|---|---:|
| bad bbox | 0 |
| bad area | 0 |
| bad image_id | 0 |
| bad category_id | 0 |
| duplicate image ids | 0 |
| missing image refs | 0 |
| missing category refs | 0 |

Loader random sample check:

| idx | image_id | image | depth | masks | boxes |
|---:|---:|---|---|---|---|
| 2191 | 2192 | (246, 246, 3) | (246, 246) | (246, 246, 49) | (49, 4) |
| 3621 | 3622 | (246, 246, 3) | (246, 246) | (246, 246, 50) | (50, 4) |
| 10600 | 10601 | (246, 246, 3) | (246, 246) | (246, 246, 25) | (25, 4) |
| 11283 | 11284 | (246, 246, 3) | (246, 246) | (246, 246, 50) | (50, 4) |
| 13541 | 13542 | (246, 246, 3) | (246, 246) | (246, 246, 49) | (49, 4) |

## Validation Commands

```bash
/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python -m pytest tests/test_build_resized_coco_rgbd_dataset.py tests/test_diagnose_target_scale_source_pool.py -q
/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python tools/build_resized_coco_rgbd_dataset.py --help
git diff --check
```

These commands are the required pre-commit checks for this R49 documentation commit.

## Gate

R49 passes the data-build gate. It only validates the resized dataset as data. It does not approve or report any training run or model eval.
