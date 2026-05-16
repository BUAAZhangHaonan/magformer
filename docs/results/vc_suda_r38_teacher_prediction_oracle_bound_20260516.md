# VC-SUDA R38 Teacher Prediction Oracle Bound - 2026-05-16

## Conclusion

The current R12/R15 teacher predictions do not have enough mask coverage quality on `target_unlabeled200` to plausibly reach `0.61` segm AP through score thresholding, calibration, or NMS.

The best-case score oracle, where prediction scores are replaced by GT-derived mask IoU, reaches only `0.3614` segm AP. The raw masks themselves cover only `34.78%` of GT instances at IoU `>=0.75`, and only `3.69%` at IoU `>=0.90`. Stop the calibration/NMS route for this prediction set; the bottleneck is mask/localization coverage, not ranking.

## Scope

- No training was run.
- No repo code was changed.
- Diagnostics were computed by a temporary script at `/tmp/r38_oracle_bound.py` on `4029`.
- Diagnostic outputs were written under `output/diagnostics/r38_oracle_score_upper_bound_20260516/` and are intentionally not committed.

## Inputs

GT:

- `magformer_datasets/pseudo_real_512/annotations/instances_target_unlabeled.json`
- Images: `200`
- GT instances: `11,750`

Prediction files:

| Label | File | Predictions | Note |
|---|---|---:|---|
| R15 topk200 | `output/diagnostics/r15_topk200_maxdets200_20260515/coco_instances_results.json` | `13,806` | Historical R15 topk/maxDets=200 output |
| R33 R12 ckpt499 | `output/diagnostics/r33_r12_ckpt499_sanity_unlabeled200_20260516/coco_instances_results.json` | `13,806` | Current-code reproduction of the same R12 ckpt499 baseline |

The two prediction JSON files are byte-identical by `cmp -s`, so all metrics below apply to both R15 and R33/R12 `ckpt499`.

## Oracle Recall

For each GT mask, R38 finds the maximum mask IoU against all predictions in the same image.

| Metric | Value | Count |
|---|---:|---:|
| Oracle recall@50 | `0.684340` | `8,041 / 11,750` |
| Oracle recall@75 | `0.347830` | `4,087 / 11,750` |
| Oracle recall@90 | `0.036936` | `434 / 11,750` |
| Max-IoU p10 / p50 / p90 | `0.1706 / 0.6608 / 0.8579` | - |

One-to-one maximum bipartite matching gives almost the same readout, so the recall is not inflated by one prediction covering many GT masks:

| Metric | Value | Count |
|---|---:|---:|
| One-to-one recall@50 | `0.684000` | `8,037 / 11,750` |
| One-to-one recall@75 | `0.347745` | `4,086 / 11,750` |

## Density Buckets

| GT density bucket | GT | Oracle R@50 | Oracle R@75 | Oracle R@90 | One-to-one R@50 | One-to-one R@75 | Max-IoU p50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `25-30` | `550` | `0.972727` | `0.778182` | `0.223636` | `0.972727` | `0.778182` | `0.847431` |
| `46-60` | `6,547` | `0.791049` | `0.436230` | `0.044142` | `0.790744` | `0.436078` | `0.723810` |
| `>90` | `4,653` | `0.500107` | `0.172577` | `0.004728` | `0.499678` | `0.172577` | `0.500000` |

The dense `>90` images are the main blocker. Even with oracle selection, only `17.26%` of dense GT masks have a prediction at IoU `>=0.75`.

## GT Area Buckets

| GT area bucket | GT | Oracle R@50 | Oracle R@75 | Oracle R@90 | One-to-one R@50 | One-to-one R@75 | Max-IoU p50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `<=256` | `2,934` | `0.154056` | `0.009884` | `0.000000` | `0.154056` | `0.009884` | `0.240273` |
| `257-452` | `2,943` | `0.732246` | `0.204553` | `0.006116` | `0.731906` | `0.204553` | `0.620690` |
| `453-579` | `2,945` | `0.920543` | `0.519864` | `0.046180` | `0.919864` | `0.519525` | `0.755793` |
| `>579` | `2,928` | `0.929986` | `0.657445` | `0.095628` | `0.929645` | `0.657445` | `0.800245` |

Small masks are not recoverable by ranking. For `<=256` px GT masks, oracle recall@75 is only `0.99%`.

## Prediction Statistics

| Statistic | Value |
|---|---:|
| Predictions/image mean | `69.03` |
| Predictions/image min / p10 / p50 / p90 / max | `25 / 45.8 / 62.5 / 110.0 / 127` |
| Score p10 / p50 / p90 | `0.511687 / 0.935378 / 0.964177` |
| Pred mask area / image area p10 / p50 / p90 | `0.000671 / 0.001835 / 0.002750` |
| Pred mask area / best-GT area p10 / p50 / p90 | `0.483218 / 1.090341 / 1.704041` |
| Same-image pred-pred pairs checked | `526,934` |
| Duplicate pairs with pred-pred mask IoU `>0.70` | `127` |
| Duplicate pair ratio | `0.000241` |
| Images with at least one duplicate pair | `73 / 200` |

Duplicate predictions are not the main AP blocker. The duplicate-pair ratio is only `0.0241%`, so NMS can clean some local clutter but cannot create missing high-IoU masks.

## Oracle Score AP Upper Bound

R38 also ran a score oracle: keep the original predictions and masks, but replace each prediction score with either its max mask IoU to any same-image GT or its one-to-one matched IoU. COCOeval was run with `maxDets=200`; AP was read directly from the COCO precision tensor because pycocotools' printed `AP` slot is hardcoded to `maxDets=100`.

| Oracle score | segm AP | segm AP50 | segm AP75 | segm AR200 | bbox AP | bbox AP50 | bbox AP75 | bbox AR200 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Max IoU score | `0.361035` | `0.681884` | `0.346416` | `0.359574` | `0.422177` | `0.758454` | `0.422772` | `0.449302` |
| Matched IoU score | `0.361386` | `0.683168` | `0.346535` | `0.359574` | `0.423042` | `0.761191` | `0.423157` | `0.449311` |

This is the key upper-bound result. Perfect score ordering of the existing prediction masks only moves segm AP from the R15/R33 baseline `0.320048` to about `0.3614`. That is far below `0.61`, so score calibration, threshold search, or NMS cannot close the gap.

## Decision

Stop the calibration/NMS/threshold route for the current R12/R15 teacher predictions.

The next useful work should change the prediction mask set itself: model quality, mask generation, dense/small-object coverage, data path, or a different pseudo-teacher source. Pure post-processing of these masks does not have the needed upper bound.

## Validation

Commands and checks:

```bash
cmp -s \
  output/diagnostics/r15_topk200_maxdets200_20260515/coco_instances_results.json \
  output/diagnostics/r33_r12_ckpt499_sanity_unlabeled200_20260516/coco_instances_results.json

/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python /tmp/r38_oracle_bound.py \
  --ann magformer_datasets/pseudo_real_512/annotations/instances_target_unlabeled.json \
  --pred r15_topk200 output/diagnostics/r15_topk200_maxdets200_20260515/coco_instances_results.json \
  --pred r33_r12_ckpt499 output/diagnostics/r33_r12_ckpt499_sanity_unlabeled200_20260516/coco_instances_results.json \
  --out-dir output/diagnostics/r38_oracle_score_upper_bound_20260516
```

Output summaries:

- `output/diagnostics/r38_oracle_score_upper_bound_20260516/r15_topk200_oracle_bound_summary.json`
- `output/diagnostics/r38_oracle_score_upper_bound_20260516/r33_r12_ckpt499_oracle_bound_summary.json`
- `output/diagnostics/r38_oracle_score_upper_bound_20260516/combined_oracle_bound_summary.json`

No `output/` files are part of the commit.
