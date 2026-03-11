# 2026-03-12 LightDepth v2 Final Report

## Scope

- Dataset / protocol: `0831_1K / 1024 / 20 epochs`
- Stage A output root:
  - `output/experiments/0831_1k_20ep_1024_lightdepth_stage_a`
- Stage B output root:
  - `output/experiments/0831_1k_20ep_1024_lightdepth_stage_b`

## Reference baseline

| Model | Best segm AP | Last segm AP |
| --- | ---: | ---: |
| `magformer_nodpth_ref` | 70.4891 | 70.3317 |

## Stage A — Lowest-Cost Feasibility

| Model | Best segm AP | Last segm AP | Peak memory (MB) | Wall time (sec) |
| --- | ---: | ---: | ---: | ---: |
| `magformer_lightdepth_mobilenetv3_directadd_edge` | **74.7425** | 74.7000 | **26,654.61** | **7,307** |
| `magformer_lightdepth_mobilenetv3_gatedadd_edge` | 74.7000 | 74.5697 | 27,675.91 | 7,456 |
| `magformer_lightdepth_mobilenetv3_film_edge_validhole` | 74.4047 | 74.2559 | 27,551.66 | 7,403 |
| `magformer_lightdepth_resnet18_gatedadd_edge` | 74.2844 | 73.7688 | 28,183.59 | 7,316 |

### Stage A decision

- All Stage A light-depth candidates beat `magformer_nodpth_ref`.
- Top 2 by `best segm AP`:
  1. `magformer_lightdepth_mobilenetv3_directadd_edge`
  2. `magformer_lightdepth_mobilenetv3_gatedadd_edge`
- Selected best encoder family for Stage B:
  - `MobileNetV3-Small`

## Stage B — Cross-Attention Validation

| Model | Best segm AP | Last segm AP | Peak memory (MB) | Wall time (sec) |
| --- | ---: | ---: | ---: | ---: |
| `magformer_lightdepth_mobilenetv3_crossattn_edge` | 70.6976 | 70.3714 | 27,068.74 | 7,513 |
| `magformer_lightdepth_mobilenetv3_crossattn_edge_validhole` | 70.5774 | 70.5464 | 27,027.91 | 7,713 |
| `magformer_lightdepth_mobilenetv3_crossattn_edge_validhole_variance` | 71.0372 | 70.9806 | 26,884.93 | 8,098 |

### Stage B decision

- No Stage B candidate beat the best Stage A model.
- Best Stage B candidate:
  - `magformer_lightdepth_mobilenetv3_crossattn_edge_validhole_variance`
  - `best segm AP = 71.0372`
- Gap to best Stage A:
  - `74.7425 - 71.0372 = 3.7053 AP`
- Conclusion:
  - `cross_attn(res3)` is not promoted.

## Final recommendation

### Primary candidate

- `magformer_lightdepth_mobilenetv3_directadd_edge`
- Why:
  - highest `best segm AP`
  - lowest peak memory among Stage A finalists
  - lowest wall time among Stage A finalists

### Secondary candidate

- `magformer_lightdepth_mobilenetv3_gatedadd_edge`
- Why:
  - nearly tied with the primary candidate on AP
  - serves as the strongest gated-fusion reference

## Takeaways

- The lightweight depth path is clearly useful on this dataset.
- `MobileNetV3-Small` dominates `ResNet18` under the current budget.
- Simple additive fusion outperformed both gated and cross-attention variants in this line.
- Adding Stage B cross-attention increased complexity without producing a meaningful AP gain.

## Recommended next action

- Use `magformer_lightdepth_mobilenetv3_directadd_edge` as the default lightweight RGB-D candidate for future comparisons.
- Keep `magformer_lightdepth_mobilenetv3_gatedadd_edge` as the ablation/reference alternative.
