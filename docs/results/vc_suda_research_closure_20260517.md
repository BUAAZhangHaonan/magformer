# VC-SUDA research closure, 2026-05-17

This note closes the R88-R91 research loop. The original goal was to bring target segmentation AP to within 10 AP of the Teacher, which means about `61+` main segm AP. That goal is not reached.

## Current Verified Best

- Best verified target result remains R80 on `target_unlabeled200`: segm AP `33.64`, bbox AP `38.99`.
- R88 `32K_1024` cache source-scale did not improve the target result: segm AP `33.31`, bbox AP `38.90`.
- R88 target segm AP50 was `65.08`, but AP50 is not the main AP gate and does not mean the `61+` target was reached.
- Teacher source first50 anchor is segm AP `62.53`; R91 zero-step anchor reproduced the same line with segm AP `62.53`.

## Paths Excluded

- R84 offline TTA bank with lower pseudo weight finished but stayed below R80: target segm AP `33.51`.
- R86 `32K_512` source-scale finished but stayed below R80: target segm AP `33.47`.
- R88 faithful `32K_1024` VC-SUDA source-scale finished but stayed below R80: target segm AP `33.31`.
- R90 supervised-only `32K_1024` smoke at LR `1.0e-5` quickly dropped source first50 segm AP to `52.72` after 150 iterations.
- R91 supervised-only `32K_1024` smoke at LR `1.0e-6` improved over R90 but still dropped source first50 segm AP to `55.97` after 150 iterations, below the `58` smoke gate.

## Common Conclusion

Pseudo loss scaling, a high-score offline bank, more source data, and low-LR supervised fine-tuning did not open target main AP. The source first50 sanity also falls quickly after `32K` fine-tuning, even in supervised-only smoke runs.

The shared failure mode is not a single missing weight knob. The current MagFormer VC-SUDA path remains around `33-34` target main segm AP while the Teacher source anchor is `62.53` on first50. Continuing blind trials on the same adaptation modules is not justified.

## Decision

- Do not start R90 or R91 long training.
- Close this VC-SUDA research iteration.
- Do not mark the `61+` target segm AP goal as achieved.
- Do not treat R88 AP50 `65.08` as passing evidence, because the goal is main segm AP.

## If Work Continues

The next serious path should revisit data generation, target split, and annotation protocol, or run an independent RGB-only official Mask2Former baseline. It should not keep stacking MagFormer adaptation modules without first proving that the data and protocol can support the target-domain AP goal.
