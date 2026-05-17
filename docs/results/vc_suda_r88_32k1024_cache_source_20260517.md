# VC-SUDA R88 32K_1024 cache source-scale, 2026-05-17

R88 did not pass.

It tests the faithful 32K_1024 source-scale setup on top of R80. The source dataset stays at the original 1024 geometry through the cache-backed annotation path, but this did not open the target main AP ceiling.

## Scope

- Host: `4029`.
- Project: `/home/hdd3/zhanghaonan/magformer`.
- Branch: `feature/vc-suda-sim2real`.
- Base config: `configs/vc_suda_stage_c_r80_unsup_schedule_1024.yaml`.
- R88 config: `configs/vc_suda_stage_c_r88_32k1024_cache_source_1024.yaml`.
- Source root: `magformer_datasets/20260318_1K_32254`.
- Source annotation: `cache/coco_loader/instances_train.sqlite`.
- Output dir: `output/vc_suda/stage_c_r88_32k1024_cache_source_1024`.
- Checkpoint: `output/vc_suda/stage_c_r88_32k1024_cache_source_1024/checkpoint_iter_0000750.pth`.

Kept fixed from R80: R12 `ckpt499` true-resume, `solver.max_iter=750`, `vc_suda.unsupervised_weight=0.5`, `vc_suda.unsupervised_warmup_epochs=1`, balanced CE with `balanced_ce_min_fg_ratio=0.05`, no offline pseudo bank, no target sampling, and no training-code change.

## External Eval

Target eval:

- Path: `output/diagnostics/r88_32k1024_cache_source_target_unlabeled200_20260517`.
- Protocol: `pseudo_real_target_unlabeled200`, 200 images, 1024 backmap, `bbox,segm`, `inference_topk=200`, `max_dets=200`.
- Protocol checker: pass.

Source sanity eval:

- Path: `output/diagnostics/r88_32k1024_cache_source_original_first50_20260517`.
- Protocol: original source `first50`, 1024 backmap, `bbox,segm`, `inference_topk=100`, `max_dets=100`, `--allow-nondefault-weights`.
- Protocol checker: pass.

## R88 vs R80

| eval | metric | R80 | R88 | delta |
|---|---|---:|---:|---:|
| target_unlabeled200 | bbox AP | `38.99` | `38.90` | `-0.09` |
| target_unlabeled200 | segm AP | `33.64` | `33.31` | `-0.33` |
| target_unlabeled200 | segm AP50 | `65.59` | `65.08` | `-0.51` |
| target_unlabeled200 | segm AP75 | `31.73` | `31.35` | `-0.38` |
| original first50 | bbox AP | `47.48` | `31.98` | `-15.50` |
| original first50 | segm AP | `43.91` | `26.48` | `-17.43` |

Required headline values:

- R88 target segm AP is `33.31`; bbox AP is `38.90`.
- R80 target segm AP is `33.64`.
- R88 source first50 segm AP is `26.48`; R80 source first50 segm AP is `43.91`.
- R88 target segm AP50 is `65.08`, but AP50 is not the main AP gate and does not mean the run passed.

## Hard Buckets

| bucket | R88 bbox AP/AP75 | R88 segm AP/AP75 | R88 oracle R@50/R@75 |
|---|---:|---:|---:|
| overall | `0.389007 / 0.373474` | `0.333131 / 0.313528` | `0.687915 / 0.370809` |
| normal | `0.462606 / 0.471329` | `0.409060 / 0.409346` | `0.779670 / 0.470134` |
| dense | `0.235638 / 0.175759` | `0.175931 / 0.115134` | `0.505217 / 0.178055` |
| dense_tiny | `0.263479 / 0.199866` | `0.214363 / 0.154632` | `0.550218 / 0.208734` |
| tiny_area_le_256 | `0.079154 / 0.010872` | `0.009982 / 0.000357` | `0.175869 / 0.017042` |
| bottom20_area | `0.060904 / 0.006504` | `0.003144 / 0.000075` | `0.118298 / 0.008085` |

## Decision

R88 未超过 R80，32K_1024 忠实 source-scale 没打开 target 主 AP，上限仍在 33-34 AP 区间。

The target bbox AP is nearly tied with R80, but the main target segm AP is lower. The source first50 sanity also falls sharply, so 32K_1024 source-scale is not a replacement for R80.

## Artifacts

- Target metrics: `output/diagnostics/r88_32k1024_cache_source_target_unlabeled200_20260517/metrics.cocoeval.json`.
- Source metrics: `output/diagnostics/r88_32k1024_cache_source_original_first50_20260517/metrics.cocoeval.json`.
- Target protocol checker: `output/diagnostics/r88_32k1024_cache_source_target_unlabeled200_20260517/protocol_check.log`.
- Source protocol checker: `output/diagnostics/r88_32k1024_cache_source_original_first50_20260517/protocol_check.log`.
- Bucket AP: `output/diagnostics/r88_32k1024_cache_source_target_unlabeled200_20260517/r88_bucket_ap.md`.
