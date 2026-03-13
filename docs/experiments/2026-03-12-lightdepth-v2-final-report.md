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
| `magformer_lightdepth_convnextlite_spatialgate_edge_validhole` | **75.1968** | **75.1351** | 29,751.82 | 7,612 |
| `magformer_lightdepth_mobilenetv3_sagate_edge_validhole` | **74.8344** | 74.6657 | 26,674.87 | 8,063 |
| `magformer_lightdepth_mobilenetv3_spatialgate_edge_validhole` | **74.8259** | 74.8259 | **26,494.71** | 7,798 |
| `magformer_lightdepth_mobilenetv3_channelattn_edge` | 74.8212 | 74.8212 | 26,553.98 | 7,788 |
| `magformer_lightdepth_mobilenetv3_esanetctx_edge_validhole` | 73.9342 | 73.5803 | 26,923.31 | 7,631 |
| `magformer_lightdepth_mobilenetv3_directadd_edge` | **74.7425** | 74.7000 | **26,654.61** | **7,307** |
| `magformer_lightdepth_mobilenetv3_gatedadd_edge` | 74.7000 | 74.5697 | 27,675.91 | 7,456 |
| `magformer_lightdepth_mobilenetv3_film_edge_validhole` | 74.4047 | 74.2559 | 27,551.66 | 7,403 |
| `magformer_lightdepth_resnet18_gatedadd_edge` | 74.2844 | 73.7688 | 28,183.59 | 7,316 |

### Stage A decision

- All Stage A light-depth candidates beat `magformer_nodpth_ref`.
- Top 2 by `best segm AP`:
  1. `magformer_lightdepth_convnextlite_spatialgate_edge_validhole`
  2. `magformer_lightdepth_mobilenetv3_sagate_edge_validhole`
- Selected best encoder family for Stage B:
  - `ConvNeXt-lite`

## Stage B — Cross-Attention Validation

| Model | Best segm AP | Last segm AP | Peak memory (MB) | Wall time (sec) |
| --- | ---: | ---: | ---: | ---: |
| `magformer_lightdepth_convnextlite_crossattn_edge_validhole_variance` | **71.1444** | **71.0442** | 29,795.48 | 7,650 |
| `magformer_lightdepth_convnextlite_crossattn_edge_validhole` | 71.1047 | 70.9047 | 29,402.16 | 7,690 |
| `magformer_lightdepth_convnextlite_crossattn_edge` | 70.8097 | 70.7701 | 29,835.68 | 7,564 |
| `magformer_lightdepth_mobilenetv3_crossattn_edge` | 70.6976 | 70.3714 | 27,068.74 | 7,513 |
| `magformer_lightdepth_mobilenetv3_crossattn_edge_validhole` | 70.5774 | 70.5464 | 27,027.91 | 7,713 |
| `magformer_lightdepth_mobilenetv3_crossattn_edge_validhole_variance` | 71.0372 | 70.9806 | 26,884.93 | 8,098 |

### Stage B decision

- No Stage B candidate beat the best Stage A model.
- Best Stage B candidate:
  - `magformer_lightdepth_convnextlite_crossattn_edge_validhole_variance`
  - `best segm AP = 71.1444`
- Gap to best Stage A:
  - `75.1968 - 71.1444 = 4.0524 AP`
- Conclusion:
  - `cross_attn(res3)` is not promoted.

## Final recommendation

### Primary candidate

- `magformer_lightdepth_convnextlite_spatialgate_edge_validhole`
- Why:
  - highest stable `last segm AP`
  - highest `best segm AP`
  - still far cheaper than the original heavy dual-backbone line
  - keeps the depth prior path explicit via edge + valid-hole gating

### Secondary candidate

- `magformer_lightdepth_mobilenetv3_spatialgate_edge_validhole`
- Why:
  - strongest lightweight MobileNetV3 candidate
  - lower memory than the ConvNeXt-lite winner

### Accuracy ceiling candidate

- `magformer_lightdepth_mobilenetv3_sagate_edge_validhole`
- Why:
  - highest `best segm AP`
  - still lightweight, but less stable than `spatial_gate`

### Speed-oriented candidate

- `magformer_lightdepth_mobilenetv3_esanetctx_edge_validhole`
- Why:
  - fastest among the new literature-inspired follow-up methods
  - keeps AP in the same overall lightweight band

## Takeaways

- The lightweight depth path is clearly useful on this dataset.
- `MobileNetV3-Small` dominates `ResNet18` under the current budget.
- Simple additive fusion was competitive, but low-cost spatial/channel attention and shared-specific gating now edge it out.
- Adding Stage B cross-attention increased complexity without producing a meaningful AP gain.

## Recommended next action

- Use `magformer_lightdepth_convnextlite_spatialgate_edge_validhole` as the default best-accuracy lightweight RGB-D candidate.
- Keep `magformer_lightdepth_mobilenetv3_spatialgate_edge_validhole` as the lower-memory lightweight alternative.
- Track `magformer_lightdepth_mobilenetv3_sagate_edge_validhole` as the best-peak-AP but lower-stability candidate.
