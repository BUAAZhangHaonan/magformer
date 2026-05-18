# R118 VC-SUDA Stage C Small-Step Validation

Date: 2026-05-18
Branch: `feature/vc-suda-sim2real`
Mode: small-step Stage C validation only, not a long training run.

## Scope

- Start from R114 formal teacher weights:
  `output/baseline/r114_magformer_r113warm_target150_balanced_2000/checkpoint_iter_0002000.pth`
- Use the R117 passing pseudo bank, materialized for R118 without new inference:
  `output/diagnostics/r118_r117_pseudo_bank_remaining75_20260518/pseudo_bank_score0967_fill059_area100_nms095.json`
- Train for 300 iterations only.
- Use low unsupervised weight: `vc_suda.unsupervised_weight=0.05`.
- Keep R114 recipe controls: no RGB photo augmentation, no depth noise, image size 1024, dice weight 10, mask weight 5, and point importance sampling 0.

## Bank Gate

R118 must reproduce the R117 best pseudo-bank gate before training can start.

| item | expected |
| --- | ---: |
| images | 75 |
| kept annotations | 2544 |
| P50 | 0.947327 |
| P75 | 0.682783 |
| R50 | 0.552246 |
| R75 | 0.398029 |
| dense/high R50 | 0.353046 |
| dense/high R75 | 0.208161 |
| empty pseudo images | 0 |

## Training Guard

Training is allowed only if the bank gate passes.

Stop conditions:

- From iter 20 onward, stop if pseudo loss ratio is greater than 12% for two consecutive logged rows.
- Stop immediately if any pseudo loss ratio is greater than 15%.
- Stop at iter 100 if remaining75 AP is below 0.392.
- Stop if val28 or original first50 source sanity clearly drops.

## Evaluation Plan

- Built-in val28 eval at iter 100, 200, and 300.
- At iter 100 and best checkpoint, evaluate:
  - remaining75
  - full200 reference
  - original first50 source sanity

## Preflight Notes

- `validate_config(strict=True)` passes.
- Loader smoke passes with 150 source images, 150 zero-weight target-labeled images, 75 target-unlabeled images, and 2544 offline pseudo instances.
- `tools/verify_vc_suda_stage.py` fails on its old warm-start whitelist because R118 starts from the R114 formal-teacher checkpoint. No training-code change was made for this.
- The R118 bank annotations store fill ratio under `pseudo_fill_ratio`, so the config uses `offline_pseudo.fill_ratio_key=pseudo_fill_ratio`.

## Success Criteria

| gate | required |
| --- | ---: |
| val28 | >= 0.322 |
| remaining75 | >= 0.397 |
| full200 reference | >= 0.523 |
| original first50 source sanity | >= 0.495 |

## Status

Pending training and eval.
