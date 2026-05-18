# VC-SUDA Baseline Gap Protocol - 2026-05-19

## Research Question

The current question is not whether VC-SUDA reaches Teacher-10AP alone. The useful comparison is:

1. How large is the source-to-pseudo-real sim2real gap under the same 32K synthetic setup?
2. Does VC-SUDA improve over fair labeled-only baselines on the same pseudo-real target splits?

Teacher-10AP can stay as a historical sanity target, but it should not be the only target-domain success criterion.

## Fair Protocol

Use a fixed source and target protocol:

- Source setup: train on the 32K synthetic source cache at 1024 input resolution.
- Pseudo-real target setup: use the deterministic target150 train split, val28 anchor split, and remaining75 non-leakage split.
- Target evaluation: run 1024 backmap eval with `topk=200` and `maxDets=200` for val28 and remaining75.
- Report source-val AP and target AP separately, then compute the practical sim2real gap from source-only target eval and labeled-only target eval.

Leakage risks:

- `full200` includes promoted target training images after target150, so it is only a continuity reference.
- `train200` is an oracle fit check, not a fair generalization number.
- Pseudo-bank tuning on remaining75 can leak validation pressure into the unsupervised target pool. Treat it as diagnosis unless the bank rule is frozen before the final comparison.

## Current MagFormer Facts

| Run | Role | Split / protocol | segm AP | Status |
| --- | --- | --- | ---: | --- |
| R98 | MagFormer source-only | 32K source val subset, first 300 val images | `0.810410` | Strong source-domain learning signal. |
| R98 | MagFormer source-only | pseudo-real target_unlabeled200, 1024 backmap | `0.0000009269` | Near-zero target transfer under the documented source-only check. |
| R114 | MagFormer labeled-only target150 | val28, 1024 backmap | `0.321831` | Current fair labeled-only anchor. |
| R114 | MagFormer labeled-only target150 | remaining75, 1024 backmap | `0.392173` | Current non-leakage labeled-only target score. |
| R136 | Official RGB Mask2Former labeled-only target150 | val28 / remaining75, official built-in maxDets100 | `0.178559 / 0.240292` | Current RGB diagnostic baseline; below R114 by `0.143272 / 0.151881` segm AP. |
| R118 | VC-SUDA from R114 | val28 / remaining75 / full200 at iter0099 | `0.321894 / 0.392568 / 0.520202` | Approximately neutral and failed its gates. |
| R122 | VC-SUDA depth-boundary from R114 | val28 / remaining75 at iter0099 | `0.322904 / 0.392410` | AP is approximately neutral; bucket go/no-go failed. |

Interpretation: R114 already gives a strong labeled-only target baseline. The early VC-SUDA variants have not shown a clear target-domain gain over that baseline.

## Non-MagFormer Baseline Status

R136 official RGB Mask2Former is the current target150 diagnostic baseline, but it is still official built-in `maxDets=100` eval rather than the final topk200/maxDets200 fair wrapper.

Historical R97 facts before the R134-R136 chain:

- It used official RGB-only Mask2Former on the same 32K source cache.
- It ran only 1500 iterations.
- Its source val segm AP reached `36.2643` at iter1500.
- Its bbox AP was unusable in that export because all serialized prediction boxes were zero, while segmentation AP was non-zero.

Needed before treating R136 as a fair non-MagFormer baseline:

- Run the fair target eval wrapper with topk200/maxDets200 on val28 and remaining75.
- Keep the current R136 remaining75 number as official built-in `maxDets=100` diagnosis only.
- Do not launch R136 continuation to 2000 from these metrics alone; first close the eval-protocol mismatch.

## Next Execution Matrix

| Step | Run target | Purpose | Output metric |
| ---: | --- | --- | --- |
| 1 | Continue R97 source on 32K / 1024 | Get a fairer official RGB Mask2Former source baseline. | Source val segm AP. |
| 2 | R97 labeled-only finetune on pseudo-real train150 | Match the MagFormer R114 target-supervised setup. | Train150 fit plus val28 guard. |
| 3 | Evaluate R97 finetune on val28 and remaining75 | Compare non-MagFormer labeled-only against R114 and VC-SUDA. | val28 and remaining75 segm AP with 1024 backmap. |

Decision rule: VC-SUDA should be claimed as useful only if it beats the frozen labeled-only target baseline on val28 and remaining75 without relying on full200/train200 or post-hoc pseudo-bank tuning.
