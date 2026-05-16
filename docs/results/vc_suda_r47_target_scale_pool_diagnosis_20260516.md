# VC-SUDA R47 Target-Scale Source Pool Diagnosis, 2026-05-16

R47 passes the no-train construction gate. A target-scale source pool larger than 4K images is feasible if the ordinary 32K source annotation is resized by a linear scale around `0.24-0.25`.

No training, model eval, or GPU job was run.

## Inputs and Outputs

Tool:

```bash
/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python tools/diagnose_target_scale_source_pool.py \
  --source-ann magformer_datasets/20260318_1K_32254/annotations/instances_train.json \
  --target-ann magformer_datasets/pseudo_real_512/annotations/instances_target_unlabeled.json \
  --existing-pool-ann magformer_datasets/pseudo_real_512/annotations/instances_source.json \
  --forbidden-ann target_labeled=magformer_datasets/pseudo_real_512/annotations/instances_target_labeled.json \
  --forbidden-ann target_unlabeled=magformer_datasets/pseudo_real_512/annotations/instances_target_unlabeled.json \
  --forbidden-ann val=magformer_datasets/pseudo_real_512/annotations/instances_val.json \
  --scale-factors 0.20 0.24 0.25 0.30 0.35 0.40 \
  --summary-json output/diagnostics/r47_target_scale_pool_diagnosis_20260516/summary.json
```

Generated files:

- `tools/diagnose_target_scale_source_pool.py`
- `tests/test_diagnose_target_scale_source_pool.py`
- `output/diagnostics/r47_target_scale_pool_diagnosis_20260516/summary.json`
- `output/diagnostics/r47_target_scale_pool_diagnosis_20260516/run.stdout`

## Gate Definition

The prior gate uses:

- images `>=4000`
- mask area p50 in `339-678`, derived from target p50 `452` with `0.75x-1.5x`
- `area<=256` ratio within absolute `0.05` of target `0.249702`
- basename overlap with `target_labeled`, `target_unlabeled`, and `val` equal to `0`

## Baselines

| pool | images | instances | inst/img p50 | inst/img p90 | mask area p50 | area<=256 | basename overlap | gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| target_unlabeled200 | 200 | 11,750 | 50 | 100 | 452 | 0.249702 | self-overlap only | fail images |
| ordinary 32K source | 25,654 | 1,398,531 | 50 | 99 | 7,572 | 0.011850 | 0 | fail p50/small |
| pseudo_real source | 1,008 | 61,652 | 50 | 100 | 438 | 0.266431 | 0 | fail images |

The pseudo_real source is already target-scale, but it is only `1,008` images. It cannot meet the `>=4K` pool-size gate alone.

## Candidate Results

Projected source means annotation-only projection with `area_projected = area_original * scale^2`. Mixed pool means `pseudo_real source + projected ordinary 32K source`.

| scale | projected images | projected instances | projected p50 | projected area<=256 | projected inst/img p50/p90 | projected gate | mixed images | mixed instances | mixed p50 | mixed area<=256 | mixed inst/img p50/p90 | mixed gate |
|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---|
| 0.20 | 25,654 | 1,398,531 | 302.88 | 0.389409 | 50 / 99 | fail | 26,662 | 1,460,183 | 306.20 | 0.384217 | 50 / 99 | fail |
| 0.24 | 25,654 | 1,398,531 | 436.15 | 0.227160 | 50 / 99 | pass | 26,662 | 1,460,183 | 436.20 | 0.228819 | 50 / 99 | pass |
| 0.25 | 25,654 | 1,398,531 | 473.25 | 0.202906 | 50 / 99 | pass | 26,662 | 1,460,183 | 471.50 | 0.205588 | 50 / 99 | pass |
| 0.30 | 25,654 | 1,398,531 | 681.48 | 0.128762 | 50 / 99 | fail | 26,662 | 1,460,183 | 664.02 | 0.134575 | 50 / 99 | fail |
| 0.35 | 25,654 | 1,398,531 | 927.57 | 0.091106 | 50 / 99 | fail | 26,662 | 1,460,183 | 899.15 | 0.098508 | 50 / 99 | fail |
| 0.40 | 25,654 | 1,398,531 | 1211.52 | 0.068286 | 50 / 99 | fail | 26,662 | 1,460,183 | 1174.40 | 0.076652 | 50 / 99 | fail |

All projected and mixed candidates have basename overlap `0` against `target_labeled`, `target_unlabeled`, and `val`.

## Answers

There is a `>=4K` target-scale source pool. The feasible pool is the ordinary 32K source after real resize, optionally mixed with the existing pseudo_real source. It gives `25,654` projected images alone or `26,662` mixed images.

The recommended linear resize scale is `0.24`. It best matches the target p50 and small-area ratio together: projected p50 `436.15` versus target `452`, and `area<=256` `0.227160` versus target `0.249702`. Scale `0.25` also passes, but its small-area ratio is farther from target.

Annotation-only resize is enough for this diagnosis, but it is not enough for training. Training needs RGB, depth, instance-map, and depth-cache tensors whose geometry matches the resized annotation. The existing `scripts/analysis/build_multires_dataset.py` pattern is the right path for materializing a real resized dataset.

The next step is allowed: build a real resized dataset at scale `0.24`, or the nearest explicit square resolution that reproduces this annotation scale. Do not train from annotation-only files.
