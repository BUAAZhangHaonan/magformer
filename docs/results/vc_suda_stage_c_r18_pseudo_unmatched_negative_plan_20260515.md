# VC-SUDA Stage C R18 Pseudo Unmatched Negative Plan - 2026-05-15

## Conclusion

Run one short R18 continuation from R12 `ckpt499` with only one new variable:
target-unlabeled pseudo-branch high-score unmatched query background CE.

## Design

- Config: `configs/vc_suda_stage_c_r18_pseudo_unmatched_neg_r12_ckpt499_1024_teacher8499.yaml`.
- Init checkpoint: `output/vc_suda/stage_c_r12_32k_source_r8b_ckpt999_continue_1024_teacher8499/checkpoint_iter_0000499.pth`.
- Threshold: `pseudo_unmatched_negative_score_thresh: 0.9`.
- Weight: `pseudo_unmatched_negative_weight: 0.05`.
- Scope: target-unlabeled pseudo branch only, after Hungarian pseudo-positive matching.
- Default behavior: disabled by schema default.

The weight is deliberately below the existing pseudo CE scale. `pseudo_loss_ce`
uses weight `2.0`; R18 adds `0.05`, or 2.5% of that CE scale, and it is still
multiplied by the existing Stage C `unsupervised_weight: 0.02`. The threshold
starts at `0.9` because R17 found many unmatched queries above that score while
their max IoU to kept pseudo masks stayed low.

## Guardrails

- No source, LR, pseudo threshold, mask/dice, top-k, or data changes.
- `checkpoint_max_keep: null`.
- No sweep.
- GPU launch only on `4,5,6,7`.

## Validation Plan

1. Unit tests:
   - default-off loss keys are unchanged;
   - enabled loss penalizes only high-score unmatched foreground queries;
   - low-score unmatched and matched queries are not penalized;
   - config fields have strict types.
2. Preflight:
   - `tools/verify_vc_suda_stage.py --config configs/vc_suda_stage_c_r18_pseudo_unmatched_neg_r12_ckpt499_1024_teacher8499.yaml --require-stage C`.
3. Dry-run:
   - run R17 signal diagnostics on R18 config and R12 `ckpt499`;
   - confirm `pseudo_unmatched_high_score_count` and `pseudo_unmatched_negative_loss` appear in train loss diagnostics.
4. First checkpoint external eval:
   - evaluate first numbered checkpoint with external 1024 backmap on `target_unlabeled200`, bbox+segm.
   - hard stop if segm AP is below R7/R8B stopline; current R8B ckpt999 reference is `0.319162`, and current best R15/R12 topk200 is `0.320048`.
