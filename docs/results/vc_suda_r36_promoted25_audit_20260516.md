# VC-SUDA R36 Promoted 25 Distribution Audit - 2026-05-16

## Conclusion

R31 promoted 25 is biased and too hard for a one-shot `+25` supervised expansion.

The promoted set is not just "25 more labeled target images". It is a dense and difficult subset: instance-count p50 is `99` versus `50` for the remaining target_unlabeled 175, and R12/R33 teacher segm AP on promoted images is only `0.231670` versus `0.336575` on the remaining 175. Depth and RGB statistics are effectively the same, so the drop is not explained by bad depth files, NaN/Inf, or image brightness.

This directly explains the R34/R35 split: R35 original-split true-resume keeps the old distribution and stays near R12 (`0.320048 -> 0.319300` segm AP), while R34 true-resume injects the biased/hard promoted set and drops to `0.301879`.

## Scope

- No training was run.
- No code was changed.
- Temporary audit script: `/tmp/r36_promoted25_audit.py`.
- Untracked diagnostics: `output/diagnostics/r36_promoted25_audit_20260516/`.
- Contact sheet: `output/diagnostics/r36_promoted25_audit_20260516/promoted25_rgb_gtmask_contact_sheet.png`.
- Metrics JSON: `output/diagnostics/r36_promoted25_audit_20260516/audit_metrics.json`.

## Promoted Set Recovery

The promoted set was recovered two ways:

1. `instances_target_labeled_r31_plus25.json - instances_target_labeled.json` by `file_name`.
2. `instances_target_unlabeled.json - instances_target_unlabeled_r31_minus25.json` by `file_name`.

Both methods returned the same 25 images. The recovered original target_unlabeled image ids also match the R31 plan list exactly.

Promoted original image ids:

```text
1, 2, 4, 5, 6, 7, 8, 9, 14, 15, 16, 20, 27, 31, 35, 36, 39, 41, 46, 48, 62, 64, 70, 77, 83
```

## Distribution

| group | images | anns | inst min/p50/p90/max | ann area p10/p50/p90 | small mask ratio |
|---|---:|---:|---|---|---:|
| original target_labeled 25 | 25 | 1697 | 50 / 50 / 100 / 100 | 90 / 401 / 641.4 | 1.0000 |
| promoted 25 | 25 | 1941 | 25 / 99 / 100 / 100 | 72 / 372 / 604 | 1.0000 |
| remaining target_unlabeled 175 | 175 | 9809 | 25 / 50 / 99 / 100 | 108 / 468 / 661 | 1.0000 |
| val28 | 28 | 1892 | 50 / 50 / 100 / 100 | 82.1 / 399 / 629 | 1.0000 |

Readout:

- Promoted 25 is much denser than the remaining 175. Its instance p50 is `99`, not `50`.
- Promoted 25 has smaller masks than the remaining 175: area p50 `372` vs `468`, and p10 `72` vs `108`.
- After promotion, the new labeled set has `3638` annotations; promoted images contribute `1941/3638 = 53.35%` of labeled masks while being only half the images.

## Bbox Shape

| group | bbox width p10/p50/p90 | bbox height p10/p50/p90 | bbox area p10/p50/p90 |
|---|---|---|---|
| original target_labeled 25 | 17 / 27 / 34 | 18 / 28 / 34 | 384 / 726 / 1026 |
| promoted 25 | 16 / 27 / 34 | 16 / 27 / 34 | 338 / 676 / 990 |
| remaining target_unlabeled 175 | 19 / 28 / 34 | 19 / 29 / 34 | 436.6 / 754 / 1054 |
| val28 | 18 / 27 / 34 | 18 / 28 / 34 | 384.6 / 725 / 1023 |

Promoted objects are slightly smaller by bbox too. This matches the lower annotation-area distribution and makes the set more mask-sensitive.

## Depth And RGB

| group | depth valid p10/p50/p90 | depth hole p10/p50/p90 | max NaN | max Inf | luma mean p50 | luma std p50 |
|---|---|---|---:|---:|---:|---:|
| original target_labeled 25 | 0.94941 / 0.94999 / 0.95041 | 0.04959 / 0.05001 / 0.05059 | 0 | 0 | 0.4397 | 0.0940 |
| promoted 25 | 0.94947 / 0.94992 / 0.95038 | 0.04962 / 0.05008 / 0.05053 | 0 | 0 | 0.4498 | 0.0902 |
| remaining target_unlabeled 175 | 0.94945 / 0.94999 / 0.95047 | 0.04953 / 0.05001 / 0.05055 | 0 | 0 | 0.4476 | 0.0967 |
| val28 | 0.94959 / 0.95008 / 0.95060 | 0.04940 / 0.04992 / 0.05041 | 0 | 0 | 0.4452 | 0.0925 |

Depth and RGB do not separate promoted from the rest. Valid depth stays around `0.95`, hole ratio around `0.05`, and NaN/Inf are `0` in all audited images.

## Teacher Performance

Teacher source: R33 current-code R12 `ckpt499` sanity eval, using `output/diagnostics/r33_r12_ckpt499_sanity_unlabeled200_20260516/coco_instances_results.json`.

The subgroup COCOeval uses the original target_unlabeled200 GT and R12/R33 predictions, then filters by promoted 25 versus remaining 175.

| group | images | GT | predictions | bbox AP/AP50/AP75 | segm AP/AP50/AP75 |
|---|---:|---:|---:|---|---|
| promoted25 | 25 | 1941 | 2254 | 0.307086 / 0.629242 / 0.270460 | 0.231670 / 0.522647 / 0.185690 |
| remaining175 | 175 | 9809 | 11552 | 0.412118 / 0.762832 / 0.400486 | 0.336575 / 0.671570 / 0.303495 |

Teacher proxy metrics show the same pattern:

| group | pred/image p50/p90 | score p10/p50/p90 | P/R@50 | P/R@75 | pred mask area / GT area p50 |
|---|---|---|---|---|---:|
| promoted25 | 108 / 115.6 | 0.5349 / 0.9233 / 0.9579 | 0.4916 / 0.5708 | 0.2072 / 0.2406 | 1.4239 |
| remaining175 | 62 / 103 | 0.5048 / 0.9377 / 0.9648 | 0.6002 / 0.7068 | 0.3134 / 0.3690 | 1.2800 |

Readout:

- Promoted segm AP is `-0.104905` below remaining (`0.231670` vs `0.336575`).
- Promoted bbox AP is `-0.105032` below remaining (`0.307086` vs `0.412118`).
- Promoted proxy recall is lower at both thresholds: R@50 `0.5708` vs `0.7068`, R@75 `0.2406` vs `0.3690`.
- Promoted predictions have larger over-coverage: predicted mask area / GT area p50 `1.4239` vs `1.2800`.

This is a teacher-quality difference on the exact images promoted into supervision. It is not just a global metric artifact.

## Promoted Sample Table

| original image_id | instances | area p50 | small ratio | file_name |
|---:|---:|---:|---:|---|
| 1 | 50 | 565.5 | 1.0000 | `15414194BA211_50_scene_000027_000201_v0.png` |
| 2 | 50 | 489.5 | 1.0000 | `RB44145_50_scene_000026_000908_v0.png` |
| 4 | 50 | 474.0 | 1.0000 | `76030110x_50_scene_000020_000512_v0.png` |
| 5 | 50 | 461.5 | 1.0000 | `7449150079_50_scene_000028_000948_v0.png` |
| 6 | 50 | 504.0 | 1.0000 | `RC-MASTER-SMD_C6032-28_T_50_scene_000029_001068_v1.png` |
| 7 | 50 | 497.5 | 1.0000 | `687106149022_50_scene_000004_000376_v0.png` |
| 8 | 50 | 485.0 | 1.0000 | `76030110x_50_scene_000029_001148_v0.png` |
| 9 | 50 | 426.0 | 1.0000 | `658410821024_50_scene_000014_000288_v1.png` |
| 14 | 100 | 345.0 | 1.0000 | `7448262510_XXL_100_scene_000000_000224_v0.png` |
| 15 | 99 | 263.0 | 1.0000 | `658410821024_100_scene_000004_000695_v0.png` |
| 16 | 100 | 325.0 | 1.0000 | `490107670612_100_scene_000004_000825_v1.png` |
| 20 | 99 | 270.0 | 1.0000 | `SOIC127P1030X265-18N_100_scene_000011_000957_v0.png` |
| 27 | 100 | 275.0 | 1.0000 | `618025231421_100_scene_000006_000390_v1.png` |
| 31 | 100 | 347.5 | 1.0000 | `A-DF15A_KG-T2S_1_100_scene_000003_000969_v1.png` |
| 35 | 25 | 610.0 | 1.0000 | `74437625201002_25_scene_000003_000571_v0.png` |
| 36 | 97 | 315.0 | 1.0000 | `679303124022_100_scene_000003_000488_v0.png` |
| 39 | 100 | 396.5 | 1.0000 | `615002138421_100_scene_000006_000677_v1.png` |
| 41 | 100 | 413.5 | 1.0000 | `76030110x_100_scene_000004_000139_v0.png` |
| 46 | 25 | 640.0 | 1.0000 | `74437625201002_25_scene_000006_001238_v1.png` |
| 48 | 100 | 274.0 | 1.0000 | `68710814522_100_scene_000007_000044_v1.png` |
| 62 | 97 | 367.0 | 1.0000 | `749014013_100_scene_000005_000867_v0.png` |
| 64 | 100 | 281.0 | 1.0000 | `618025231421_100_scene_000005_001166_v1.png` |
| 70 | 100 | 359.0 | 1.0000 | `7448262510_XXL_100_scene_000004_000142_v1.png` |
| 77 | 99 | 396.0 | 1.0000 | `615008160321_100_scene_000000_000630_v0.png` |
| 83 | 100 | 337.0 | 1.0000 | `490107670612_100_scene_000008_000002_v0.png` |

## Answer To The R34/R35 Question

R34 dropped because the `+25` split changed the supervised target distribution, not because true-resume continuation is inherently harmful.

- R35 keeps the original split and changes only true-resume continuation: segm AP changes from `0.320048` to `0.319300`, a small `-0.000748` delta.
- R34 keeps true resume but adds the promoted set: segm AP drops to `0.301879`, a `-0.018169` delta from R12/R33.
- The promoted set is exactly the kind of data that can destabilize a short continuation: dense images, smaller objects, worse teacher masks, and more predicted mask over-coverage.

So R31 promoted 25 is not suitable as a one-shot `+25` supervised expansion. The next labeled expansion should be balanced by instance density and teacher quality, or it should be introduced through a smaller staged ablation rather than adding this promoted set all at once.

## Validation

- Promoted set recovered from both JSON-diff methods and matched the R31 plan ids exactly.
- Distribution audit covered original target_labeled 25, promoted 25, remaining target_unlabeled 175, and val28.
- Teacher audit used existing R33/R12 predictions; no new inference or training was run.
- Contact sheet was generated under `output/diagnostics/r36_promoted25_audit_20260516/` and is intentionally untracked.
