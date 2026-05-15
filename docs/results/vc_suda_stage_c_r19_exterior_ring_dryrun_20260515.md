# VC-SUDA Stage C R19 Exterior Ring Dry-Run - 2026-05-15

## Conclusion

The dry-run supports one cautious short train check with a matched-mask exterior ring penalty, but not a sweep.

Reasons:

- Ring foreground probability is clearly above far background: exterior ring mean/p50/p90 is `0.111685/0.091848/0.217193`, while far-background mean is `0.000129`.
- Interior probability stays high: mean/p50 is `0.833223/0.898117`, so the signal is about spillover around matched positives, not a missing-positive problem.
- Area ratio is only moderately inflated on most matches: mean/p90 is `1.085093/1.269596`, but the max `4.833333` still shows tail cases worth checking.

## Implementation

R19 adds dry-run instrumentation only.

- Criterion hook: `VCSUDACriterion.pseudo_label_diagnostics`.
- Diagnose script: `tools/diagnose_vc_suda_signals.py`.
- Default ring radius: `--exterior-ring-radius 2`.
- Training loss: unchanged by default. No exterior ring loss is added to `pseudo_label_loss` unless diagnostics are explicitly requested.
- Ring construction: threshold the matched pseudo mask at `0.5`, dilate by radius `2`, use `dilated - interior` as exterior ring, and treat pixels outside dilation as far background.
- GT density bucket: unsupported in this dry-run path because the target-unlabeled pseudo diagnostic batch does not expose image id or GT count.

## Dry-Run Command

No optimizer step was run.

```bash
CUDA_VISIBLE_DEVICES=0 python tools/diagnose_vc_suda_signals.py \
  --config configs/vc_suda_stage_c_r12_32k_source_r8b_ckpt999_continue_1024_teacher8499.yaml \
  --weights output/vc_suda/stage_c_r12_32k_source_r8b_ckpt999_continue_1024_teacher8499/checkpoint_iter_0000499.pth \
  --max-batches 2 \
  --batch-size 1 \
  --num-workers 0 \
  --device cuda:0 \
  --exterior-ring-radius 2 \
  --output-json output/diagnostics/r19_exterior_ring_20260515/r12_ckpt499_max_batches2.json
```

The output JSON stays uncommitted under `output/diagnostics/r19_exterior_ring_20260515/`.

## Dry-Run Metrics

| Metric | Value |
|---|---:|
| matched_count | 79 |
| exterior_ring_prob mean | 0.111685 |
| exterior_ring_prob p50 | 0.091848 |
| exterior_ring_prob p90 | 0.217193 |
| interior_prob mean | 0.833223 |
| interior_prob p50 | 0.898117 |
| background_far_prob mean | 0.000129 |
| pred/target area ratio mean | 1.085093 |
| pred/target area ratio p90 | 1.269596 |
| pred/target area ratio p95 | 1.460224 |
| pred/target area ratio max | 4.833333 |
| ring_pixel_count mean | 130.139236 |
| ring_pixel_count p50 | 130.0 |
| ring_pixel_count p90 | 149.599991 |

R17 counts are unchanged in the same run shape: `79/79` kept/matched pseudo instances, `37` unmatched queries above score `0.7`, and `29` above `0.9`.

## Recommendation

Recommend one short R19 train check only.

- Radius: keep `2`.
- Initial loss weight if implemented: `0.05` on a matched-ring mean foreground-prob penalty, normalized over valid matched rings and multiplied by the existing unsupervised branch weight.
- Do not use unmatched negatives. R18 showed that path can push recall down and increase FN.
