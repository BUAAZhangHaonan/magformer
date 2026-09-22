# Corrected v317 next-stage screening

This stage isolates point-sampling changes and the DCCG confidence switch while
keeping the effective v317 model path fixed.

For fusion, these runs diagnose the bidirectional cross-attention path with and
without learned depth confidence. They do not treat depth edge, valid-hole
priors, `residual_alpha`, or noise-mask weighting as active mechanisms.
Every arm therefore sets:

```yaml
priors: []
prior:
  enabled: false
```

Keys that the current DCCG forward does not consume, including
`residual_alpha`, `noise_mask_weight`, the legacy fusion `hidden_dim`,
static `feature_dims`, and robust prior-normalization options, are omitted.
Temperature, clamp, confidence-estimator width, and entropy weight remain
explicit because the confidence-enabled DCCG path consumes them.

| Arm | Matcher points | Loss points | Oversample | Confidence |
| --- | ---: | ---: | ---: | --- |
| C0 | 12544 | 12544 | 3.0 | on |
| P1 | 12544 | 25088 | 3.0 | on |
| P2 | 12544 | 12544 | 6.0 | on |
| D0 | 12544 | 12544 | 3.0 | off |
| P3 | 12544 | 50176 | 3.0 | on |

P3 is conditional and must not launch with the initial C0/P1/P2/D0 screen.
See [PROMOTION.md](PROMOTION.md) for the seed-43 and seed-44 derivation
contract.

## AMP scale diagnostic

The fixed runtime values come from the paired smoke diagnostic under:

```text
output/smoke/20260712_062711_4512c35_scale_diag/R64_summary.json
output/smoke/20260712_062711_4512c35_scale_diag/R1_summary.json
```
output/smoke/20260712_062711_4512c35_scale_diag/scale_comparison.json

- R64: PASS; 4 micro-steps, 1 optimizer step, 0 AMP skips, finite gradients,
  scale 64.0 -> 64.0, and one-image validation mAP 0.6818156121.
- R1: PASS; 4 micro-steps, 1 optimizer step, 0 AMP skips, finite gradients,
  scale 1.0 -> 1.0, and one-image validation mAP 0.6817464737.

Both R64 and R1 pass. The screening matrix fixes `amp_init_scale: 64.0`
because it preserves more FP16 underflow headroom than scale 1 while the
diagnostic shows no overflow skip at 64. The consecutive-overflow guard is
also explicit as `max_consecutive_amp_skips: 16` in every arm.
