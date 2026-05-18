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

Stopped at the iter100 gate. Do not continue Stage C from this run.

## Training Result

- Session: `r118_vc_suda_smallstep`
- Config: `configs/baseline_vc_suda_r118_magformer_r114warm_pseudo300.yaml`
- Output: `output/vc_suda/r118_magformer_r114warm_pseudo300`
- Warm-start: R114 formal-teacher checkpoint loaded with 0 missing keys and 0 unexpected keys.
- Checkpoint evaluated: `output/vc_suda/r118_magformer_r114warm_pseudo300/checkpoint_iter_0000099.pth`
- The run was stopped after the iter100 gate check. No 300-iter completion was attempted after the gate failed.

Pseudo weighted-loss contribution stayed below the guard:

| iter | pseudo contribution / total loss |
| ---: | ---: |
| 20 | 0.022473 |
| 40 | 0.004152 |
| 60 | 0.012822 |
| 80 | 0.008278 |
| 100 | 0.008565 |
| 120 | 0.021527 |

No logged row exceeded 12%, and no row exceeded 15%.

## Eval Result

External eval used the 1024 backmap protocol. Target-domain eval used topk200/maxDets200. Original first50 source sanity used the guarded original-first50 wrapper with topk100/maxDets100.

| checkpoint | split | bbox AP | bbox AP50 | bbox AP75 | segm AP | segm AP50 | segm AP75 | gate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| iter0099 | val28 | 0.357882 | 0.706915 | 0.322983 | 0.321894 | 0.636520 | 0.293118 | fail, below 0.322 |
| iter0099 | remaining75 | 0.430418 | 0.755625 | 0.441692 | 0.392568 | 0.706452 | 0.395474 | fail, below 0.397 |
| iter0099 | full200 reference | 0.521945 | 0.825188 | 0.589285 | 0.520202 | 0.820947 | 0.587653 | fail, below 0.523 |
| iter0099 | original first50 source sanity | 0.424621 | 0.735228 | 0.451832 | 0.502724 | 0.790991 | 0.555318 | pass |

Eval artifacts:

- `output/diagnostics/r118_iter0099_val28_1024_backmap_topk200_20260518`
- `output/diagnostics/r118_iter0099_remaining75_1024_backmap_topk200_20260518`
- `output/diagnostics/r118_iter0099_full200_reference_1024_backmap_topk200_20260518`
- `output/diagnostics/r118_iter0099_original_first50_1024_backmap_20260518`

## Decision

R118 does not allow continuing Stage C.

Reasons:

1. Val28 misses the required `0.322` by a small margin: `0.321894`.
2. Remaining75 is above the hard early-stop floor `0.392`, but below the success gate `0.397`.
3. Full200 reference is below the required `0.523`.

Source first50 stays healthy, so the failure is target-side quality rather than source collapse.
