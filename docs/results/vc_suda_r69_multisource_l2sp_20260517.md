# VC-SUDA R69 multisource L2-SP diagnostic - 2026-05-17

## Conclusion

R69 fails the gate. L2-SP on decoder and pixel decoder keeps target scale below `2.0x`, but source first50 is still below `0.55` and target_unlabeled200 is below both `0.30` hard target and `0.22` short-run target.

## Setup

- Config: `configs/vc_suda_stage_b_r69_multisource_l2sp_1024.yaml`
- Warm start: `output/experiments/20260510_1k_finetune_full_1024_v13/checkpoint_iter_0008499.pth`
- Stage: `B`
- Source mix: `original:pseudo = 1:1`
- `target_labeled_weight: 1.0`
- `unsupervised_weight: 0.0`
- Iterations: `500`
- Train output: `output/vc_suda/vc_suda_stage_b_r69_multisource_l2sp_1024`
- Checkpoint used for external eval: `checkpoint_iter_0000499.pth`

## L2-SP Setting

`vc_suda.source_retention` is enabled with:

- `include_prefixes: [decoder, pixel_decoder]`
- `exclude_prefixes: []`
- `weight: 0.1`
- `normalize: true`

The R69 smoke on the real config/model reported:

```text
source_retention True 0.1 True 292 19931842 0.000000000000
```

That means 292 selected reference parameter tensors and 19,931,842 selected parameter elements. The same smoke starts at zero L2-SP, as expected for the warm-start reference. The training log reports total model parameters `50,068,863`; it does not separately print the L2-SP selected tensor/element count, so the selected count comes from the R69 smoke, while the total parameter count comes from the training log.

Warm-start load in training reported missing/unexpected keys `0/0`. Target eval strict load reported matched `774/774`, missing `0`, unexpected `0`, and shape mismatch `0`.

## External Eval

The original first50 eval was already complete. The target_unlabeled200 directory only had `protocol_summary.json`, so target eval was rerun with the established protocol:

- Entry point: `tools/evaluate_1024_backmap.py`
- Backend: `MAGFORMER_MS_DEFORM_ATTN_BACKEND=pytorch`
- Flag: `--force-pytorch-msda`
- Base config: `configs/eval_vc_suda_stage_b_1024_teacher8499_segm.yaml`
- Dataset: `magformer_datasets/pseudo_real_512/annotations/instances_target_unlabeled.json`
- Split: `train`
- Images: `200`
- Image size: `1024`
- Score/mask threshold: `0.05/0.5`
- Inference topk/maxDets: `200/200`
- Protocol checker status: `pass`

| Split | Eval dir | bbox AP/AP50/AP75 | segm AP/AP50/AP75 |
|---|---|---:|---:|
| original first50 | `output/diagnostics/r69_multisource_l2sp_ckpt0499_original_first50_20260517` | `0.542072 / 0.794411 / 0.592747` | `0.493580 / 0.787959 / 0.541100` |
| target_unlabeled200 | `output/diagnostics/r69_multisource_l2sp_ckpt0499_target_unlabeled200_20260517` | `0.212557 / 0.547368 / 0.128738` | `0.150892 / 0.451722 / 0.039892` |

## Target Scale And Oracle Diagnostics

Area ratio uses the R61/R65 image-level p50 convention: median image-level prediction p50 divided by median image-level GT p50.

| Signal | Value |
|---|---:|
| GT bbox area p50 over images | `779.75` |
| Pred bbox area p50 over images | `1409.25` |
| Target bbox area ratio | `1.807310` |
| GT mask area p50 over images | `501.75` |
| Pred mask area p50 over images | `864.50` |
| Target mask area ratio | `1.722970` |
| Predictions | `18,534` |
| Prediction count p50/p90 over images | `85.5 / 146.1` |

Oracle/tiny brief:

- Mask oracle recall: R@50 `0.502979`, R@75 `0.089021`, R@90 `0.000255`.
- One-to-one oracle recall: R@50 `0.502723`, R@75 `0.089021`.
- Matched oracle-score AP: segm `0.191089`, bbox `0.241491`.
- Tiny bucket `area <= 256`: `2934` GT, oracle R@50 `0.003067`, R@75 `0.0`; one-to-one matched@50 `9`, matched@75 `0`.
- Dense bucket `>90` instances: oracle R@50 `0.294434`, R@75 `0.030948`.

Diagnostic outputs:

- Area ratio JSON: `output/diagnostics/r69_multisource_l2sp_ckpt0499_target_unlabeled200_20260517/area_ratio_summary.json`
- Oracle summary: `output/diagnostics/r69_multisource_l2sp_ckpt0499_target_oracle_20260517/r53_prediction_union_oracle_summary.json`

## R65 Comparison

| Run | Source first50 segm AP | Target_unlabeled200 segm AP | Target bbox area ratio |
|---|---:|---:|---:|
| R65 `ckpt499` | `0.501789` | `0.186416` | `1.619x` |
| R69 `ckpt499` | `0.493580` | `0.150892` | `1.807x` |

R69 is worse than R65 on both source and target AP, while still passing the coarse target-scale area limit.

## Gate Decision

Fail.

| Gate | Required | R69 ckpt0499 | Result |
|---|---:|---:|---|
| Source first50 segm AP | `>=0.55` | `0.493580` | fail |
| Target_unlabeled200 segm AP hard | `>=0.30` | `0.150892` | fail |
| Target bbox area ratio | `<=2.0` | `1.807310` | pass |
| Short-run target segm AP | `>=0.22` | `0.150892` | fail |

R69 does not justify extension. L2-SP on decoder/pixel_decoder does not recover source sanity and does not improve target AP over R65.
