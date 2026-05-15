# VC-SUDA Stage C R17 Signal Diagnostics - 2026-05-15

R17 adds no-step instrumentation for the VC-SUDA training signal. It does not
change the loss, matcher, pseudo-label filter, or training schedule.

## Method

- Script: `tools/diagnose_vc_suda_signals.py`.
- Criterion hook: `VCSUDACriterion.pseudo_label_diagnostics`, default off.
- Config: `configs/vc_suda_stage_c_r12_32k_source_r8b_ckpt999_continue_1024_teacher8499.yaml`.
- Dry-run: `max_batches=2`, `batch_size=1`, `num_workers=0`, `CUDA_VISIBLE_DEVICES=0`.
- No optimizer was built. No backward pass or `optimizer.step()` was run.
- Output JSON files were written under `output/diagnostics/r17_vc_suda_signal_diagnostics_20260515/` and are intentionally not committed.

Commands:

```bash
CUDA_VISIBLE_DEVICES=0 /home/hdd3/zhanghaonan/anaconda3/bin/conda run -n magformer \
  python tools/diagnose_vc_suda_signals.py \
  --config configs/vc_suda_stage_c_r12_32k_source_r8b_ckpt999_continue_1024_teacher8499.yaml \
  --weights output/vc_suda/stage_c_r12_32k_source_r8b_ckpt999_continue_1024_teacher8499/checkpoint_iter_0000499.pth \
  --max-batches 2 \
  --batch-size 1 \
  --num-workers 0 \
  --device cuda:0 \
  --output-json output/diagnostics/r17_vc_suda_signal_diagnostics_20260515/r12_ckpt499_max_batches2.json

CUDA_VISIBLE_DEVICES=0 /home/hdd3/zhanghaonan/anaconda3/bin/conda run -n magformer \
  python tools/diagnose_vc_suda_signals.py \
  --config configs/vc_suda_stage_c_r12_32k_source_r8b_ckpt999_continue_1024_teacher8499.yaml \
  --weights output/vc_suda/stage_c_r8b_lsj10_low_lr_continue_1024_teacher8499/checkpoint_iter_0000999.pth \
  --max-batches 2 \
  --batch-size 1 \
  --num-workers 0 \
  --device cuda:0 \
  --output-json output/diagnostics/r17_vc_suda_signal_diagnostics_20260515/r8b_ckpt999_with_r12_config_max_batches2.json
```

## Key Numbers

| Weights | source total loss mean | target_labeled raw loss mean | pseudo raw loss mean | pseudo weighted mean | kept / matched pseudo | unmatched queries | high-score unmatched >=0.7 | high-score unmatched >=0.9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R12 `ckpt499` | 9.254741 | 13.665007 | 1.821150 | 1.26e-12 | 79 / 79 | 321 | 37 | 29 |
| R8B `ckpt999` with R12 config | 108.480019 | 13.461350 | 1.841105 | 1.20e-12 | 73 / 73 | 327 | 41 | 37 |

The tiny pseudo weighted mean is expected at the first dry-run batches. R12 uses
the same quadratic unsupervised ramp as training, so `unsupervised_weight` is
near zero at the start of the 32K-source epoch.

## Unmatched High-Score Distribution

R12 `ckpt499`, threshold `0.7`:

- count `37`
- score mean `0.955288`, median `0.996371`, p90 `0.999563`, max `0.999763`
- max-IoU-to-pseudo mean `0.082664`, median `0.050466`, p90 `0.197189`, max `0.349340`

R12 `ckpt499`, threshold `0.9`:

- count `29`
- score mean `0.995536`, median `0.998118`, p90 `0.999587`, max `0.999763`
- max-IoU-to-pseudo mean `0.075329`, median `0.041823`, p90 `0.140286`, max `0.322466`

R8B `ckpt999`, threshold `0.7`:

- count `41`
- score mean `0.976218`, median `0.999393`, p90 `0.999836`, max `0.999934`
- max-IoU-to-pseudo mean `0.073029`, median `0.034445`, p90 `0.117278`, max `0.456971`

R8B `ckpt999`, threshold `0.9`:

- count `37`
- score mean `0.996275`, median `0.999541`, p90 `0.999838`, max `0.999934`
- max-IoU-to-pseudo mean `0.059255`, median `0.034445`, p90 `0.116785`, max `0.308103`

## Interpretation

The pseudo branch confirms the R16 loss-audit concern. The current
`pseudo_label_loss` supervises only Hungarian-matched pseudo positives. Unmatched
queries are not added as background negatives, and high-score unmatched queries
therefore have no direct pseudo negative gradient.

The two-batch signal is not enough to estimate AP impact, but it is enough to
show the missing training signal is common: `29-37` very high-score unmatched
queries at threshold `0.9` across only two target-unlabeled images. Most of them
have low max IoU to kept pseudo masks, so a direct negative term should be gated
carefully by query score and pseudo-mask IoU rather than applied to every
unmatched query.

## Next Step

Recommend designing a diagnostic-only pseudo unmatched negative candidate first:
foreground-high unmatched queries with low max IoU to any kept pseudo mask get a
background CE target, with the threshold and IoU gate logged before training.
Alternative: prototype a geometry-aware mask separation loss for matched queries
in dense scenes. Do not start a sweep until the proposed loss adds a measurable
signal on this same R17 dry-run.
