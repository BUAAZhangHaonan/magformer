# VC-SUDA R82 4-view TTA candidate bank diagnosis, 2026-05-17

R82 is a no-train diagnostic for the R80 teacher. It checks whether the fixed 4-view TTA union bank improves tiny and bottom20 candidate coverage enough to justify training.

## Scope

- Host: `4029`.
- Project: `/home/hdd3/zhanghaonan/magformer`.
- Config: `/home/hdd3/zhanghaonan/magformer/configs/vc_suda_stage_c_r80_unsup_schedule_1024.yaml`.
- Weights: `/home/hdd3/zhanghaonan/magformer/output/vc_suda/stage_c_r80_unsup_schedule_1024/checkpoint_iter_0000750.pth`.
- Target ann: `/home/hdd3/zhanghaonan/magformer/magformer_datasets/pseudo_real_512/annotations/instances_target_unlabeled.json`.
- GPU: `cuda:0` with `CUDA_VISIBLE_DEVICES=4`.
- Views: `1.0 noflip`, `1.0 hflip`, `1.25 noflip`, `1.25 hflip`.
- Union: same class, mask IoU `>=0.55` or bbox IoU `>=0.75`; quality is max; kept threshold is `0.1`.

No training was run. Training logic was not changed.

## Candidate Coverage

| bucket | images | GT/proxy | candidates | kept | keep-rate | candidate cov@50/@75 | kept cov@50/@75 | quality mean/p10/p50/p90 | kept area mean/p10/p50/p90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 200 | 11750 | 27409 | 19863 | `0.725` | `0.837 / 0.478` | `0.829 / 0.473` | `0.615 / 0.006 / 0.894 / 0.971` | `364 / 109 / 342 / 648` |
| normal | 158 | 7634 | 21527 | 13981 | `0.649` | `0.908 / 0.578` | `0.895 / 0.570` | `0.534 / 0.004 / 0.767 / 0.972` | `372 / 103 / 360 / 659` |
| dense | 30 | 2971 | 4176 | 4176 | `1.000` | `0.698 / 0.282` | `0.698 / 0.282` | `0.908 / 0.819 / 0.938 / 0.967` | `365 / 129 / 344 / 632` |
| dense_tiny | 12 | 1145 | 1706 | 1706 | `1.000` | `0.728 / 0.320` | `0.728 / 0.320` | `0.910 / 0.821 / 0.931 / 0.966` | `296 / 120 / 250 / 538` |
| tiny_area_le_256 | 185 | 2934 | 2085 | 1850 | `0.887` | `0.540 / 0.118` | `0.519 / 0.111` | `0.792 / 0.057 / 0.935 / 0.967` | `335 / 149 / 287 / 607` |
| bottom20_area | 181 | 2355 | 1512 | 1334 | `0.882` | `0.483 / 0.090` | `0.462 / 0.084` | `0.782 / 0.048 / 0.929 / 0.968` | `330 / 135 / 274 / 620` |

## R82 Gate

| bucket | metric | R81 | R82 | delta | gate | pass |
|---|---|---:|---:|---:|---:|---:|
| tiny_area_le_256 | candidate_gt_coverage_iou50 | `0.326` | `0.540` | `+0.214` | `0.400` | `True` |
| tiny_area_le_256 | candidate_gt_coverage_iou75 | `0.050` | `0.118` | `+0.068` | `0.080` | `True` |
| tiny_area_le_256 | kept_gt_coverage_iou50 | `0.211` | `0.519` | `+0.308` | `0.270` | `True` |
| tiny_area_le_256 | kept_gt_coverage_iou75 | `0.031` | `0.111` | `+0.080` | `0.045` | `True` |
| bottom20_area | candidate_gt_coverage_iou50 | `0.286` | `0.483` | `+0.197` | `0.350` | `True` |
| bottom20_area | candidate_gt_coverage_iou75 | `0.039` | `0.090` | `+0.051` | `0.065` | `True` |
| bottom20_area | kept_gt_coverage_iou50 | `0.186` | `0.462` | `+0.276` | `0.240` | `True` |
| bottom20_area | kept_gt_coverage_iou75 | `0.025` | `0.084` | `+0.059` | `0.040` | `True` |

## Decision

R82 passes the coverage gate. The next step may train from this TTA bank.

## Artifacts

- Bucket JSON: `/home/hdd3/zhanghaonan/magformer/output/diagnostics/r82_tta_candidate_bank_full200_20260517/r82_tta_candidate_buckets.json`.
- Summary JSON: `/home/hdd3/zhanghaonan/magformer/output/diagnostics/r82_tta_candidate_bank_full200_20260517/r82_tta_candidate_summary.json`.
- Candidate JSON: `/home/hdd3/zhanghaonan/magformer/output/diagnostics/r82_tta_candidate_bank_full200_20260517/r82_tta_candidate_predictions.json`.
- Runtime config: `/home/hdd3/zhanghaonan/magformer/output/diagnostics/r82_tta_candidate_bank_full200_20260517/r82_tta_candidate_runtime.yaml`.
