# R120 MagFormer RGB-D Error Atlas

Date: 2026-05-18
Branch: `feature/vc-suda-sim2real`
Mode: no training, no new inference. This diagnosis reads existing COCO GT and prediction JSON files only.

## Scope

R120 checks where MagFormer is still failing on target dense, tiny, and AP75-like masks after the first50 evaluation fix.

Outputs:

- `output/diagnostics/r120_error_atlas_20260518/summary.json`
- `output/diagnostics/r120_error_atlas_20260518/summary.md`
- `output/diagnostics/r120_error_atlas_20260518/instance_match.csv`
- `output/diagnostics/r120_error_atlas_20260518/prediction_fp.csv`
- `output/diagnostics/r120_error_atlas_20260518/bucket_compare.csv`
- `output/diagnostics/r120_error_atlas_20260518/delta_atlas.csv`
- `output/diagnostics/r120_error_atlas_20260518/model_delta_summary.md`

The script is `tools/diagnose_r120_error_atlas.py`.

## Protocol Boundary

Held-out rows:

- R114 val28: formal held-out anchor.
- R114 remaining75: formal held-out anchor.
- R115 val28 oracle-model eval: held-out eval from a model trained with target200 GT, so it is diagnostic only.

Reference row:

- R114 full200: reference continuity row on target200. It includes promoted training images and is not a clean held-out number.

Oracle row:

- R115 train200: oracle row trained on target200 GT. It is a capacity/recipe bound only.

## Headline Atlas Metrics

These are one-to-one mask matches from existing predictions, not official COCO AP. They expose the per-instance failure mode behind AP75.

| run | role | images | GT | pred | P50 | R50 | P75 | R75 | GT best IoU p50 | boundary F mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R114 val28 | held-out | 28 | 1892 | 3272 | 0.387225 | 0.669662 | 0.200183 | 0.346195 | 0.662690 | 0.586143 |
| R114 remaining75 | held-out | 75 | 4364 | 7278 | 0.438719 | 0.731668 | 0.268068 | 0.447067 | 0.718459 | 0.629956 |
| R114 full200 | reference | 200 | 11750 | 17432 | 0.562816 | 0.834979 | 0.418713 | 0.621191 | 0.805680 | 0.746161 |
| R115 train200 | oracle | 200 | 11750 | 15523 | 0.682213 | 0.901277 | 0.557560 | 0.736596 | 0.841463 | 0.823638 |
| R115 val28 | held-out eval from oracle model | 28 | 1892 | 3149 | 0.411877 | 0.685518 | 0.224516 | 0.373679 | 0.677827 | 0.601629 |

## Hard Buckets

R114 remaining75 held-out:

| bucket | GT | P50 | R50 | P75 | R75 | best IoU mean | boundary F mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| high100 density | 1691 | 0.330320 | 0.561798 | 0.144993 | 0.246600 | 0.504393 | 0.532020 |
| tiny area <=256 | 1070 | 0.118173 | 0.222430 | 0.012413 | 0.023364 | 0.307333 | 0.460147 |

R114 full200 reference vs R115 train200 oracle:

| bucket | run | GT | P50 | R50 | P75 | R75 | best IoU mean | boundary F mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| high100 density | R114 full200 | 4653 | 0.425565 | 0.704062 | 0.255911 | 0.423383 | 0.598893 | 0.659221 |
| high100 density | R115 train200 | 4653 | 0.499121 | 0.792822 | 0.339602 | 0.539437 | 0.659537 | 0.742040 |
| tiny area <=256 | R114 full200 | 2934 | 0.227774 | 0.464554 | 0.056985 | 0.116224 | 0.428464 | 0.596709 |
| tiny area <=256 | R115 train200 | 2934 | 0.306018 | 0.613497 | 0.089085 | 0.178596 | 0.504470 | 0.682728 |

## Model Delta

Delta compares R114 full200 reference with R115 train200 oracle on the same target200 GT.

| delta label | GT |
| --- | ---: |
| both_good | 7255 |
| both_low | 1310 |
| both_miss | 1141 |
| good_to_worse | 44 |
| hit_to_miss | 18 |
| low_to_good | 1184 |
| miss_to_hit | 798 |

Mean delta IoU is `+0.053901`. Mean delta boundary F is `+0.077477`.

By density:

| bucket | GT | R114 R50 | R115 R50 | R114 R75 | R115 R75 | mean dIoU | miss_to_hit | low_to_good | both_miss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| high100 | 4653 | 0.704062 | 0.792822 | 0.423383 | 0.539437 | 0.060644 | 427 | 478 | 950 |
| mid50 | 6547 | 0.915534 | 0.970215 | 0.735604 | 0.855812 | 0.051415 | 363 | 681 | 190 |
| low25 | 550 | 0.983636 | 0.998182 | 0.932727 | 0.985455 | 0.026453 | 8 | 25 | 1 |

By area:

| bucket | GT | R114 R50 | R115 R50 | R114 R75 | R115 R75 | mean dIoU | miss_to_hit | low_to_good | both_miss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tiny <=256 | 2934 | 0.464554 | 0.613497 | 0.116224 | 0.178596 | 0.076007 | 455 | 163 | 1116 |
| small 257-1024 | 8816 | 0.958258 | 0.997051 | 0.789247 | 0.922300 | 0.046544 | 343 | 1021 | 25 |

## Conclusion

R120 supports a single RGB-D depth-boundary and geometry-consistency next step.

Reasons:

1. The held-out blocker is tiny AP75 quality. R114 remaining75 tiny R75 is only `0.023364`, with P75 only `0.012413`.
2. Dense images remain hard even with many predictions. R114 remaining75 high100 R75 is `0.246600`, and R115 oracle only raises target200 high100 R75 to `0.539437`.
3. Oracle training mostly turns low-quality masks into good masks on non-tiny objects, but tiny objects still have `1116` both-miss cases and only `0.178596` oracle R75.

The next experiment should not be another broad Stage C run. It should test one faithful RGB-D geometry idea that directly targets small-mask boundary placement and dense-object separation.

## Validation

Commands run:

```bash
conda activate magformer
python tools/diagnose_r120_error_atlas.py --self-test
python -m py_compile tools/diagnose_r120_error_atlas.py
python tools/diagnose_r120_error_atlas.py
git diff --check
```
