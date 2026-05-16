# VC-SUDA R40 Oracle Miss/Error Atlas - 2026-05-16

## Conclusion

The R12/R15 prediction pool is capped by missing or poor masks, not by score ordering. R40 splits the R38 oracle miss source into two main blockers: tiny-object recall and dense-scene separation.

Against `11,750` GT masks, `31.57%` are no-cover (`best IoU <0.5`) and another `33.65%` are low-quality (`0.5<=IoU<0.75`). Only `34.78%` reach IoU `>=0.75`, and only `3.69%` reach IoU `>=0.90`. The next Teacher-side experiment should first improve small-object recall and dense separation, then mask alignment/shape. Depth holes and query count are weaker explanations for this pool.

## Scope

- No training was run.
- No repo code was changed.
- R15 and R33/R12 prediction JSONs were verified byte-identical with `cmp -s`.
- Metrics were computed by temporary script `/tmp/r40_oracle_miss_atlas.py` on `4029`.
- Diagnostic files were written under `output/diagnostics/r40_oracle_miss_atlas_20260516/` and are intentionally not committed.

## Inputs

GT:

- `magformer_datasets/pseudo_real_512/annotations/instances_target_unlabeled.json`
- Images: `200`
- GT instances: `11,750`

Prediction pool:

- `output/diagnostics/r15_topk200_maxdets200_20260515/coco_instances_results.json`
- Same bytes as `output/diagnostics/r33_r12_ckpt499_sanity_unlabeled200_20260516/coco_instances_results.json`
- Predictions: `13,806`

Readable side channels:

- RGB images found/read: `200 / 200`
- Depth files found/read: `200 / 200`
- Contact sheets generated: `contact_worst_oracle_r75_20.jpg`, `contact_most_no_cover_20.jpg`, `contact_dense_gt90_worst_10.jpg`

## Overall Instance Classes

| Class | Definition | Count | Rate |
|---|---|---:|---:|
| no-cover | best IoU `<0.5` | `3,709` | `31.57%` |
| low-quality | `0.5<=IoU<0.75` | `3,954` | `33.65%` |
| good | `0.75<=IoU<0.9` | `3,653` | `31.09%` |
| high-quality | IoU `>=0.9` | `434` | `3.69%` |

Oracle recalls match R38: R@50/R@75/R@90 = `0.684340 / 0.347830 / 0.036936`. Best-IoU p50/p90 = `0.660811 / 0.857863`.

## Density Buckets

| Density bucket | GT | no-cover | low-quality | good | high-quality | R@75 | R@90 | IoU p50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `25-30` | `550` | `2.73%` | `19.45%` | `55.45%` | `22.36%` | `77.82%` | `22.36%` | `0.847431` |
| `46-60` | `6,547` | `20.90%` | `35.48%` | `39.21%` | `4.41%` | `43.62%` | `4.41%` | `0.723810` |
| `>90` | `4,653` | `49.99%` | `32.75%` | `16.78%` | `0.47%` | `17.26%` | `0.47%` | `0.500000` |

Dense scenes are the clearest failure source. In `>90` density images, about half of all GT masks have no prediction above IoU `0.5`.

## Area Buckets

| GT area bucket | GT | no-cover | low-quality | good | high-quality | R@75 | R@90 | IoU p50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `<=256` | `2,934` | `84.59%` | `14.42%` | `0.99%` | `0.00%` | `0.99%` | `0.00%` | `0.240273` |
| `257-452` | `2,943` | `26.78%` | `52.77%` | `19.84%` | `0.61%` | `20.46%` | `0.61%` | `0.620690` |
| `453-579` | `2,945` | `7.95%` | `40.07%` | `47.37%` | `4.62%` | `51.99%` | `4.62%` | `0.755793` |
| `>579` | `2,928` | `7.00%` | `27.25%` | `56.18%` | `9.56%` | `65.74%` | `9.56%` | `0.800245` |

Small masks are nearly absent from the usable oracle pool. The `<=256` bucket has only `0.99%` R@75 and no high-quality masks.

## Depth And RGB Readout

Depth valid buckets do not explain the upper bound drop:

| Depth valid bucket | GT | no-cover | low-quality | R@75 | R@90 | IoU p50 |
|---|---:|---:|---:|---:|---:|---:|
| `95-99%` | `5,402` | `30.62%` | `33.78%` | `35.60%` | `3.28%` | `0.666137` |
| `<95%` | `6,348` | `32.37%` | `33.54%` | `34.09%` | `4.05%` | `0.655558` |

Depth hole buckets are the same split (`1-5%` vs `>5%`) and show only a small R@75 difference (`35.60%` vs `34.09%`). RGB brightness/contrast was readable for all images; the worst dense examples sit in normal brightness ranges, so this atlas does not support RGB exposure as the main cause.

## No-Cover / Low-Quality Failure Shape

All `7,663` no-cover or low-quality GT instances are in images that have prediction candidates. The issue is not a literal empty image-level candidate pool.

| Failure subtype | Count | Share of bad | Area ratio p50 / p90 | Center distance norm p50 / p90 | Best IoU p50 / p90 |
|---|---:|---:|---:|---:|---:|
| mask-alignment-shape | `4,262` | `55.62%` | `1.153 / 1.405` | `0.177 / 0.428` | `0.639 / 0.733` |
| mask-too-large | `2,966` | `38.71%` | `2.478 / 13.051` | `0.588 / 2.355` | `0.250 / 0.552` |
| mask-too-small | `270` | `3.52%` | `0.591 / 0.651` | `0.299 / 0.754` | `0.415 / 0.545` |
| mask-offset | `165` | `2.15%` | `1.147 / 1.429` | `0.959 / 1.987` | `0.244 / 0.497` |

The largest bad-mode is near-sized but not high-IoU masks, followed by masks that are too large. This points to separation and boundary/mask alignment more than pure score calibration.

## Image-Level Worst Lists

Worst oracle R75 images start with:

| Rank | image_id | file | density | oracle R75 | no-cover | low-quality |
|---:|---:|---|---:|---:|---:|---:|
| 1 | `99` | `1779205141_100_scene_000003_000324_v0.png` | `100` | `0.0300` | `69` | `28` |
| 2 | `169` | `615008160321_100_scene_000008_000207_v0.png` | `92` | `0.0543` | `69` | `18` |
| 3 | `14` | `7448262510_XXL_100_scene_000000_000224_v0.png` | `100` | `0.0900` | `55` | `36` |
| 4 | `164` | `UX60SC-MB-5ST(80)_100_scene_000019_000334_v0.png` | `100` | `0.1000` | `60` | `30` |
| 5 | `152` | `IndLQS_4012_50_scene_000002_000553_v1.png` | `50` | `0.1000` | `21` | `24` |

Most no-cover images start with the same dense failures: image `99` and image `169` each have `69` no-cover GT masks. The dense `>90` worst-10 list is also headed by image `99`, `169`, `14`, and `164`.

Full ranking CSVs and contact sheets are in `output/diagnostics/r40_oracle_miss_atlas_20260516/`.

## Teacher-Side Decision

Prioritize these changes next:

1. Small-object recall: the `<=256` px bucket is the strongest single blocker, with `84.59%` no-cover and `0.99%` R@75.
2. Dense separation: `>90` density has `49.99%` no-cover and only `17.26%` R@75, even with oracle matching.
3. Mask alignment/shape: among bad instances, `55.62%` are near-sized alignment/shape errors and `38.71%` are too large.

Do not make depth holes or query count the first Teacher-side bet for this prediction pool. Depth valid/hole buckets differ only slightly, and every bad GT is in an image with prediction candidates. Query diversity may still matter as a way to split dense objects, but raising maxDets/topk alone already failed in R15/R38 and cannot close the gap.

## Output Files

Generated but not committed:

- `output/diagnostics/r40_oracle_miss_atlas_20260516/r40_oracle_miss_atlas_summary.json`
- `output/diagnostics/r40_oracle_miss_atlas_20260516/instance_metrics.csv`
- `output/diagnostics/r40_oracle_miss_atlas_20260516/image_metrics.csv`
- `output/diagnostics/r40_oracle_miss_atlas_20260516/worst_oracle_r75_20.csv`
- `output/diagnostics/r40_oracle_miss_atlas_20260516/most_no_cover_20.csv`
- `output/diagnostics/r40_oracle_miss_atlas_20260516/dense_gt90_worst_10.csv`
- `output/diagnostics/r40_oracle_miss_atlas_20260516/contact_worst_oracle_r75_20.jpg`
- `output/diagnostics/r40_oracle_miss_atlas_20260516/contact_most_no_cover_20.jpg`
- `output/diagnostics/r40_oracle_miss_atlas_20260516/contact_dense_gt90_worst_10.jpg`

## Validation

```bash
cmp -s \
  output/diagnostics/r15_topk200_maxdets200_20260515/coco_instances_results.json \
  output/diagnostics/r33_r12_ckpt499_sanity_unlabeled200_20260516/coco_instances_results.json

/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python /tmp/r40_oracle_miss_atlas.py \
  --ann magformer_datasets/pseudo_real_512/annotations/instances_target_unlabeled.json \
  --pred output/diagnostics/r15_topk200_maxdets200_20260515/coco_instances_results.json \
  --out-dir output/diagnostics/r40_oracle_miss_atlas_20260516
```

The script completed with `gt=11750`, `predictions=13806`, RGB/depth readable `200/200`, and all three contact sheets generated.
