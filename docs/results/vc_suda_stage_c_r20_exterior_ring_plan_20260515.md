# VC-SUDA Stage C R20 Exterior Ring Loss Plan - 2026-05-15

## Conclusion

Run one cautious short R20 continuation from R12 `ckpt499` with only one new
variable: matched pseudo-positive exterior ring probability loss on the
target-unlabeled pseudo branch.

## Design

- Config: `configs/vc_suda_stage_c_r20_exterior_ring_r12_ckpt499_1024_teacher8499.yaml`.
- Init checkpoint: `output/vc_suda/stage_c_r12_32k_source_r8b_ckpt999_continue_1024_teacher8499/checkpoint_iter_0000499.pth`.
- Radius: `2`.
- Weight: `0.05`.
- Scope: only matched pseudo-positive masks after Hungarian matching.
- Loss: mean predicted foreground probability on the binary exterior ring.
- Default behavior: disabled by schema default.

The weight uses the R19 dry-run signal directly. The matched ring probability
mean was `0.111685`, so a raw `0.05` ring weight gives about `0.0056` main-head
loss before the existing Stage C `unsupervised_weight: 0.02`. That is small
beside pseudo mask/dice weights `5.0`, but large enough to show up in dry-run
loss diagnostics. This keeps R20 cautious without making the new term vanish.

## Guardrails

- No source, LR, pseudo threshold, mask/dice, top-k, or data changes.
- No unmatched-query background pressure.
- `checkpoint_max_keep: null`.
- No sweep.
- GPU launch only on `4,5,6,7`.

## Validation Plan

1. Unit tests:
   - default-off does not change pseudo loss;
   - enabled ring loss is positive;
   - changing only the target interior does not change ring loss;
   - zero-ring cases return zero loss and record counts.
2. Preflight:
   - `tools/verify_vc_suda_stage.py --config configs/vc_suda_stage_c_r20_exterior_ring_r12_ckpt499_1024_teacher8499.yaml --require-stage C`.
3. Dry-run:
   - run R19 signal diagnostics on R20 config and R12 `ckpt499`;
   - confirm `pseudo_exterior_ring_loss`, matched count, ring pixel count, and exterior ring probability are logged.
4. First checkpoint external eval:
   - evaluate `checkpoint_iter_0000249.pth`;
   - dataset `target_unlabeled200`;
   - 1024 backmap;
   - `--inference-topk 200`;
   - `--max-dets 200`.

## First Checkpoint Stop Rule

Hard stop R20 at `ckpt249` if either:

- segm AP `< 0.3195`;
- AP75 `< 0.284144`.

Even if AP is flat, stop if low-IoU false positives do not drop by at least 2%.
