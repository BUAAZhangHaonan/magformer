# VC-SUDA R60 R59 failure diagnosis - 2026-05-16

## Conclusion

R59 failed because its target-domain predictions are too large, not because the eval protocol or annotations are broken.

The run also shows that a 250-iter conservative Stage B step from the original Teacher cannot by itself adapt to 512 pseudo-real target scale while retaining original-source behavior. The next work should stay diagnostic: low-cost scale visualization and overlay checks first, then a minimal training contrast only after the scale conflict is explicit.

## R59 Failure Summary

R59 was the retention-min Stage B run from Teacher8499. It fails both hard gates:

| Eval | segm AP | Gate | Result |
|---|---:|---:|---|
| original first50 source | `0.4731567886` | `>= 0.55` | fail |
| target_unlabeled200 | `0.0020426278` | `>= 0.30` | fail |

The original first50 source failure means source retention is not enough. The target_unlabeled200 failure is much worse: target AP is almost zero even though inference produced predictions on every target image.

## Not An Eval Or Annotation Error

The target failure is not explained by protocol drift, checkpoint loading, empty prediction export, wrong category ids, or invalid boxes.

- Protocol checker passed for both R59 external evals.
- Strict checkpoint load matched `774/774` keys, with missing/unexpected/shape mismatch `0/0/0`.
- R59 target export has `24,196` predictions.
- `200/200` target_unlabeled images have predictions.
- All predicted categories are category `1`.
- Exported bboxes have no out-of-image violations.

Primary evidence paths:

- `output/diagnostics/r59_retention_min_ckpt0250_target_unlabeled200_20260516`
- `output/diagnostics/r59_retention_min_ckpt0250_original_first50_20260516`

## Core Evidence: Target Predictions Are Too Large

The target-domain GT objects are small. Teacher direct target predictions and R59 predictions are much larger than the target GT distribution.

| Source | bbox area p50 | mask area p50 | bbox w/h p50 | segm AP |
|---|---:|---:|---:|---:|
| target_unlabeled GT | `744` | `452` | `27/29` | n/a |
| Teacher direct target pred | `6561` | n/a | `71/91` | `0.0000259` |
| R59 pred | `3960` | `2204` | `56/70` | `0.002043` |
| R46 pred | `~756` | `~485` | n/a | `~0.323` |
| R52 pred | `~759` | `~489` | n/a | `~0.323` |

This is the clearest separator. R46/R52 predictions sit near the target GT scale and reach about `0.323` segm AP. R59 is still several times too large in bbox and mask area, so it misses the target objects even though it predicts many instances.

Reference paths:

- Teacher direct target: `output/experiments/teacher_pseudoreal_target_unlabeled200_1024_backmap_recheck_20260515`
- R46 target-scale reference: `output/diagnostics/r46_pseudoreal_source_true_resume_ckpt0750_target_unlabeled200_1024_backmap_20260516`
- R52 target-scale reference: `output/diagnostics/r52_target_sampling_ckpt0750_target_unlabeled200_1024_backmap_20260516`

## Data Split Is Clean

The split audit does not point to leakage, category mismatch, broken RGB/depth pairing, or invalid annotations.

- `source`, `target_labeled`, and `target_unlabeled` use the same category set.
- `orphan=0` for the checked splits.
- `bbox_outside=0` for the checked splits.
- RGB/depth basenames are aligned.
- Basename overlap across the three splits is `0`.
- The three split annotation scales are close enough that the failure is not caused by a malformed target_unlabeled annotation scale.

This matters because R59's target predictions are too large even though the data split itself is clean and target GT scale is stable.

## Training Log Readout

R59 was a Stage B source + target_labeled run:

- `target_unlabeled=0`, so there was no unlabeled pseudo branch in this run.
- `target_labeled_weight=0.05` was active.
- Source loss mean was about `10.48`.
- Target_labeled raw loss mean was about `91.63`.
- Weighted target_labeled loss mean was about `4.58`.

So target_labeled still provided a visible training signal. It was not ignored. But the small weighted target signal over only 250 iterations was not enough to pull the Teacher-scale target predictions back to the true target scale.

Training log path:

- `output/vc_suda/vc_suda_stage_b_r59_teacher8499_retention_min_1024/metrics_log.csv`

## Diagnosis

R59 shows a direct conflict between target-scale adaptation and original-source retention.

The original Teacher is healthy on original-source sanity, but its direct target predictions are far too large. A conservative 250-iter Stage B update preserves too much of that target-scale mismatch while still hurting source retention. This means the next solution cannot rely only on a tiny Stage B adjustment from original Teacher into 512 pseudo-real target data.

The target-scale adaptation must be handled explicitly, and the design must also protect original-source retention. Treating those as one small low-LR update is the failure mode seen in R59.

## Next Step

Do not start long training from this result.

The next low-cost work should be scale visualization and overlay diagnosis only:

- overlay target GT boxes/masks with Teacher direct predictions and R59 predictions on the same target images;
- show bbox width/height and mask area histograms for target GT, Teacher direct target, R59, R46, and R52;
- inspect a small fixed set of easy, dense, and small-object target images.

After that, design the smallest training contrast that directly tests target-scale adaptation against original-source retention.
