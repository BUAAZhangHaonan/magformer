# VC-SUDA R66 R65 failure analysis - 2026-05-17

## Conclusion

R65 is not mainly a ranking problem.

The multi-source run fixed the coarse target scale error, but it did not pass the source or target AP gates. The target failure is still mask quality: masks remain too large, high-IoU matches are weak, and tiny objects are still missed.

Recommend R67: run `R65 + freeze rgb_backbone/depth_backbone/fusion/agpe` for `500` iterations first. This keeps the Teacher source representation fixed and lets only the pixel decoder / transformer decoder learn the target mask scale. Do not add a new loss in R67.

## Inputs

- Training log: `output/vc_suda/vc_suda_stage_b_r65_multisource_retention_1024/metrics_log.jsonl`
- R65 target eval: `output/diagnostics/r65_multisource_retention_20260516/ckpt0000999_target_unlabeled200/`
- R46 target baseline eval: `output/diagnostics/r46_pseudoreal_source_true_resume_ckpt0750_target_unlabeled200_1024_backmap_20260516/`
- R65 result recap: `docs/results/vc_suda_r65_multisource_retention_20260516.md`

## R65 Recap

| Checkpoint | Source first50 segm AP | Target unlabeled200 segm AP | Target bbox area ratio |
|---|---:|---:|---:|
| `ckpt499` | `0.501789` | `0.186416` | `1.619x` |
| `ckpt999` | `0.501977` | `0.196696` | `1.590x` |

Readout:

- Multi-source mixing corrected the coarse target scale. The area ratio moved below `2x`.
- It still did not pass the AP gate. Source stayed near `0.502`, below the `0.55` source floor, and target stayed near `0.197`, far below the `0.30` target floor.
- The `500 -> 1000` extension had weak return: target AP improved only `+0.010280`, while source was almost unchanged at `+0.000188`.

## Training Log Readout

The source `1:1` config path is correct. The R65 smoke in the R65 document showed alternating `original, pseudo, original, pseudo` source samples. The training JSONL itself does not log per-batch source counts, so it cannot prove every training batch retained exact source composition after collation.

The loss balance shows target labeled supervision dominated the run:

| Signal | Mean | Median |
|---|---:|---:|
| `train/source_total_loss` | `11.815` | `8.083` |
| `train/tl_total_loss_weighted` | `37.051` | `28.571` |
| `target_labeled / source` | `4.670` | `3.348` |

This explains the observed trajectory. R65 learned target scale early, but by `1000` iterations the target AP gain was only about `+0.010` and source AP was flat. The current configuration is close to a local plateau, not an unfinished climb.

## Target Error Diagnosis

R65 is not primarily score ordering.

Key target-error facts:

- R65_999 oracle R@75 is only about `0.156`, while R46 reaches about `0.349`.
- R65 prediction mask p50 / GT mask p50 is about `1.75x`; even matched masks are still about `1.53x` GT.
- Tiny-object recall remains poor: R50 tiny R@50 is about `0.013`, while R46 tiny R@50 is about `0.157`.

Interpretation:

- Mask scale improved compared with the earlier huge-mask failures, but masks are still too large for high-IoU AP.
- The oracle R@75 gap says re-ranking scores cannot recover enough target AP from the current mask pool.
- The tiny-object gap says the model still misses many small instances instead of only assigning them low scores.

Main cause: masks remain oversized, high-IoU quality is bad, and small targets are missed.

## R67 Decision

Recommend R67:

```text
R65 + freeze rgb_backbone/depth_backbone/fusion/agpe
```

Run only `500` iterations first.

Reasons:

1. It protects the Teacher source representation, which R65 did not retain well enough.
2. It still leaves the pixel decoder and transformer decoder trainable, so the target-labeled signal can adjust mask scale and decoder behavior.
3. It does not add a new loss, so the result tests whether parameter isolation fixes the conflict before adding more machinery.

Hard short-run indicators:

- Source first50 segm AP must be `>=0.55`.
- Target bbox area ratio must be `<=2.0`.
- Target unlabeled200 segm AP should be `>=0.22` in the short run.

If R67 misses these indicators at `500` iterations, do not extend it.

## Alternative

Alternative: `R65 + target_labeled_weight 0.5`.

Do not prioritize it. R62 already showed that reducing target-labeled pressure can preserve neither the source gate nor target AP under a nearby setup. R65 shows target-labeled pressure is useful for coarse scale, so cutting it first risks losing the only part that improved.

Teacher distillation and L2-SP are valid later options. Do not add them in R67. R67 should stay code-free and test the freeze-only hypothesis first.
