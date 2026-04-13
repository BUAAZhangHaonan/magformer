# 2026-03-13 F5 Prior-Guided Cross-Attention Final Summary

## Scope

- Dataset / protocol: `0831_1K / 1024 / 20 epochs`
- F5 output root:
  - `output/experiments/0831_1k_20ep_1024_lightdepth_stage_b_f5`
- F5 design:
  - `mode=prior_guided_cross_attn`
  - explicit prior-conditioned K/V
  - single-scale `res3`
  - no dense attention bias

## F5 candidate results

| Model | Best segm AP | Last segm AP | Peak memory (MB) | Wall time (sec) | Infer FPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| `magformer_lightdepth_convnextlite_priorguidedcrossattn_edge_validhole_variance` | **71.3505** | **71.2551** | 29,643.28 | 7,298 | 1.7011 |
| `magformer_lightdepth_mobilenetv3_priorguidedcrossattn_edge_validhole_variance` | 71.1718 | 70.7227 | **26,256.05** | 7,321 | 1.7082 |
| `magformer_lightdepth_convnextlite_priorguidedcrossattn_edge_validhole` | 70.9603 | 70.6731 | 29,403.07 | 7,329 | **1.8777** |
| `magformer_lightdepth_mobilenetv3_priorguidedcrossattn_edge` | 70.9880 | 70.5855 | 27,120.09 | 7,348 | 1.6964 |
| `magformer_lightdepth_convnextlite_priorguidedcrossattn_edge` | 70.8623 | 70.7561 | 29,487.66 | 7,351 | 1.6934 |
| `magformer_lightdepth_mobilenetv3_priorguidedcrossattn_edge_validhole` | 70.8416 | 70.8416 | 27,054.78 | 7,408 | 1.6987 |

## Comparison

- Best F5 candidate:
  - `magformer_lightdepth_convnextlite_priorguidedcrossattn_edge_validhole_variance`
  - `best segm AP = 71.3505`
  - `last segm AP = 71.2551`
- Gain over best plain `cross_attn` candidate:
  - `71.3505 - 71.1444 = 0.2061 AP`
- Gain over RGB-only reference:
  - `71.3505 - 70.4891 = 0.8614 AP`
- Gap to best lightweight Stage A model:
  - `75.1968 - 71.3505 = 3.8463 AP`
- Gap to heavy RGB-D upper bound:
  - `77.4998 - 71.3505 = 6.1493 AP`

## Interpretation

- F5 completed the originally open implementation item:
  - independent fusion mode
  - independent Stage B successor suite
  - full 20-epoch results
  - inference benchmark and overlay outputs
- F5 is a modest improvement over the old `cross_attn(res3)` line.
- F5 does **not** beat the promoted Stage A low-cost fusion family.
- The best F5 result remains in the same overall accuracy band as prior Stage B attention experiments and should be kept as:
  - a completed negative-to-mixed diagnostic result
  - not the new default lightweight recommendation

## Final recommendation

- Keep `magformer_lightdepth_convnextlite_spatialgate_edge_validhole` as the best-accuracy lightweight candidate.
- Keep `magformer_lightdepth_mobilenetv3_spatialgate_edge_validhole` as the lower-memory practical alternative.
- Record `magformer_lightdepth_convnextlite_priorguidedcrossattn_edge_validhole_variance` as:
  - best F5 candidate
  - slight improvement over plain `cross_attn`
  - not promoted over Stage A winners
