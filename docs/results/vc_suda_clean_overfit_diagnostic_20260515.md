# VC-SUDA Clean Overfit Diagnostic - 2026-05-15

## Conclusion

The clean target_labeled overfit run passed the supervised-chain sanity check on the same 25-image labeled target set, but it did not prove final target-domain generalization.

Use external `tools/evaluate_1024_backmap.py` results for model selection from now on. The training-time quick eval is 512 direct inference, so it severely underestimates this setup and must not choose `model_best`.

## Code And Config

- Contrastive fix commit: `796768b7811ddaf174ea8553c495fcd80a0531c3`.
- Fix: `runtime.contrastive_enabled=false` is now honored.
- Validation for the fix: `42 passed, 1 skipped`; `ruff` passed.
- Clean overfit config commit: `eb0982151547e07f1575dc3df120b579ebd93211`.
- Config: `configs/debug_target_labeled_clean_overfit_1024_teacher8499.yaml`.

Clean overfit setup:

- Dataset: `target_labeled25`, evaluated on the same set for the sanity check.
- No VC-SUDA.
- No contrastive loss.
- No strong augmentation.
- Fixed scale.
- Warmup: `0`.
- Weight decay: `0.01`.

## Eval Protocol Finding

The training-time quick eval uses 512 direct inference. It is useful only as a rough training signal here.

The valid reporting and model-selection protocol is external 1024 backmap:

```bash
tools/evaluate_1024_backmap.py \
  --image-size 1024 \
  --batch-size 1
```

The mismatch is large:

| Checkpoint | Eval protocol | bbox AP | segm AP |
| --- | --- | ---: | ---: |
| ckpt499 | training quick eval, 512 direct | 0.0705 | - |
| ckpt499 | external 1024 backmap | 0.5510 | 0.4950 |
| ckpt999 | external 1024 backmap | 0.6319 | 0.5804 |
| ckpt1499 | external 1024 backmap, same target_labeled25 | 0.6669 | 0.6222 |

ckpt1499 reaches the overfit sanity target: same-set segm AP is above 61.

## Generalization Check

ckpt1499 does not solve the final generalization target.

| Eval set | bbox AP | segm AP |
| --- | ---: | ---: |
| target_labeled25 same-set | 0.6669 | 0.6222 |
| val28 | 0.3225 | 0.2604 |
| target_unlabeled200 | 0.3784 | 0.3084 |

Reference on `target_unlabeled200`:

| Run | segm AP |
| --- | ---: |
| Stage B | 0.2980 |
| R3-A10 | 0.3068 |
| clean overfit ckpt1499 | 0.3084 |

The clean overfit ckpt1499 is only `+0.0016` segm AP above R3-A10 on `target_unlabeled200`.

## Readout

The objective is not complete.

What this proves:

- The supervised target-labeled training path can overfit clean same-set target labels under the correct 1024 backmap protocol.
- The earlier poor quick-eval numbers were mostly an eval-protocol artifact.
- Same-set overfit does not transfer enough to val28 or target_unlabeled200.

Next step:

- Make external 1024 backmap the model-selection path.
- Stop using training-time 512 quick eval to choose best checkpoints for this target.
- Move into true generalization and semi-supervised experiments.

## Related

- [Target unlabeled diagnostic](vc_suda_target_unlabeled_diagnostic_20260515.md)
- [Target hidden-GT upper-bound diagnosis](target_hidden_gt_upper_bound_plan_20260515.md)
