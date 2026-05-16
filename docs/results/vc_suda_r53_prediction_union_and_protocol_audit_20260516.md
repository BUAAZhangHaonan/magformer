# VC-SUDA R53 Prediction Union and Protocol Audit - 2026-05-16

## Conclusion

R53 finds only modest run-to-run mask complementarity. The R46/R50/R51/R52 union reaches oracle R@75 `0.365277` and matched oracle-score segm AP `0.375248`, not a `0.45+` pool. This stays in the same `0.36-0.38` oracle-score band as the single-run diagnostics.

Do not continue training by trial and error from these runs. The useful next work is data/protocol work or a real change to the prediction mask source, not another short continuation.

## Scope

- No training was run.
- No model eval was run.
- R53 read existing prediction JSON, metrics JSON, and GT annotations only.
- New committed tool: `tools/diagnose_prediction_union_oracle.py`.
- Diagnostic output: `output/diagnostics/r53_prediction_union_oracle_20260516/r53_prediction_union_oracle_summary.json`.

## Inputs

GT:

- `magformer_datasets/pseudo_real_512/annotations/instances_target_unlabeled.json`
- Images: `200`
- GT instances: `11,750`

Predictions:

| Run | Prediction JSON | Predictions |
|---|---|---:|
| R46 | `output/diagnostics/r46_pseudoreal_source_true_resume_ckpt0750_target_unlabeled200_1024_backmap_20260516/coco_instances_results.json` | `13,758` |
| R50 | `output/diagnostics/r50_scale024_source_true_resume_ckpt0750_target_unlabeled200_1024_backmap_20260516/coco_instances_results.json` | `13,633` |
| R51 | `output/diagnostics/r51_scale024_source1k_ckpt0750_target_unlabeled200_1024_backmap_20260516/coco_instances_results.json` | `13,691` |
| R52 | `output/diagnostics/r52_target_sampling_ckpt0750_target_unlabeled200_1024_backmap_20260516/coco_instances_results.json` | `13,735` |
| Union | Concatenated R46/R50/R51/R52 predictions | `54,817` |

## Union Oracle

| Pool | Oracle R@50 | Oracle R@75 | Oracle R@90 | Dense `>90` R@75 | Small `<=256` R@75 | Matched oracle-score segm AP/AP50/AP75 |
|---|---:|---:|---:|---:|---:|---:|
| R46 | `0.685787` | `0.348511` | `0.039489` | `0.170213` | `0.010566` | `0.362376 / 0.683168 / 0.346535` |
| R50 | `0.685532` | `0.344085` | `0.037191` | `0.166129` | `0.009543` | `0.361386 / 0.683168 / 0.346535` |
| R51 | `0.685191` | `0.344511` | `0.036681` | `0.165485` | `0.009543` | `0.359406 / 0.683168 / 0.346535` |
| R52 | `0.686723` | `0.348255` | `0.038894` | `0.169568` | `0.011929` | `0.362376 / 0.683168 / 0.346535` |
| Union | `0.705532` | `0.365277` | `0.042553` | `0.181818` | `0.013292` | `0.375248 / 0.702970 / 0.366337` |

The union improves R@75 by `+0.016766` over R46 and matched oracle-score AP by `+0.012871`. This is real but small. It does not approach a `0.45+` oracle pool, and it cannot explain a path to `0.61+` AP through score ordering or run ensembling.

## Complementarity

Coverage uses best same-image mask IoU `>=0.75` per GT instance.

| Metric | Count |
|---|---:|
| GT covered by any run | `4,292 / 11,750` |
| GT covered by all four runs | `3,852 / 11,750` |
| Shared misses | `7,458 / 11,750` |
| R46-only covered GT | `37` |
| R50-only covered GT | `40` |
| R51-only covered GT | `33` |
| R52-only covered GT | `39` |

The runs mostly cover the same masks. Only `149` GT instances are covered by exactly one run, while `7,458` are missed by all runs at IoU `0.75`. There is no large hidden complementary mask pool.

## Protocol Audit

| Run | Protocol | Split | segm AP/AP50/AP75 | Meaning |
|---|---|---|---:|---|
| R44 | original 1.5K first50 Teacher sanity | original `instances_all.json`, first 50 | `0.625293 / 0.867953 / 0.724339` | Source-domain sanity only |
| R46 | pseudo_real target hidden eval | `target_unlabeled200` | `0.323252 / 0.649728 / 0.287485` | Current best target-domain gate |
| R52 | pseudo_real target hidden eval | `target_unlabeled200` | `0.322596 / 0.649458 / 0.286915` | Below R46 |

The `61+` AP number belongs to the corrected original 1.5K first50 Teacher protocol from R44. It does not describe pseudo_real `target_unlabeled200`. On the target-domain protocol, the current best is still R46 segm AP `0.323252`; R52 does not pass it.

## Gate Decision

Stop training trial-and-error for R46/R50/R51/R52 continuations. R53 shows:

1. The union oracle stays at matched oracle-score AP `0.375248`, not near `0.45+`.
2. Dense scenes and tiny masks remain weak: union dense `>90` R@75 is `0.181818`, and small `<=256` R@75 is `0.013292`.
3. Run-only coverage is tiny compared with shared misses.

The next valid direction is data/protocol or a different mask-generation source. Ensembling these prediction pools or continuing the same training family is not supported by the oracle bound.

## Validation

Commands run:

```bash
/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python tools/diagnose_prediction_union_oracle.py --help
/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python -m pytest tests/test_diagnose_prediction_union_oracle.py -q
/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python -m py_compile tools/diagnose_prediction_union_oracle.py tests/test_diagnose_prediction_union_oracle.py
/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python tools/diagnose_prediction_union_oracle.py \
  --ann magformer_datasets/pseudo_real_512/annotations/instances_target_unlabeled.json \
  --pred R46 output/diagnostics/r46_pseudoreal_source_true_resume_ckpt0750_target_unlabeled200_1024_backmap_20260516/coco_instances_results.json \
  --pred R50 output/diagnostics/r50_scale024_source_true_resume_ckpt0750_target_unlabeled200_1024_backmap_20260516/coco_instances_results.json \
  --pred R51 output/diagnostics/r51_scale024_source1k_ckpt0750_target_unlabeled200_1024_backmap_20260516/coco_instances_results.json \
  --pred R52 output/diagnostics/r52_target_sampling_ckpt0750_target_unlabeled200_1024_backmap_20260516/coco_instances_results.json \
  --out-dir output/diagnostics/r53_prediction_union_oracle_20260516
git diff --check
```

The actual summary JSON is readable at:

- `output/diagnostics/r53_prediction_union_oracle_20260516/r53_prediction_union_oracle_summary.json`
