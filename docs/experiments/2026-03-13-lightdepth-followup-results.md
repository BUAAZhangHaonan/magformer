# 2026-03-13 LightDepth Follow-up Results

## Scope

- New literature-inspired fusion candidates added on top of the existing light-depth Stage A line:
  - `magformer_lightdepth_mobilenetv3_channelattn_edge`
  - `magformer_lightdepth_mobilenetv3_spatialgate_edge_validhole`
  - `magformer_lightdepth_mobilenetv3_sagate_edge_validhole`
  - `magformer_lightdepth_mobilenetv3_esanetctx_edge_validhole`
- Output root:
  - `output/experiments/0831_1k_20ep_1024_lightdepth_stage_a`

## New candidates

| Model | Best segm AP | Last segm AP | Peak memory (MB) | Wall time (sec) | Infer ms | Infer FPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `magformer_lightdepth_mobilenetv3_sagate_edge_validhole` | **74.8344** | 74.6657 | 26,674.87 | 8,063 | 583.7467 | 1.7131 |
| `magformer_lightdepth_mobilenetv3_spatialgate_edge_validhole` | 74.8259 | **74.8259** | **26,494.71** | 7,798 | 564.9292 | 1.7701 |
| `magformer_lightdepth_mobilenetv3_channelattn_edge` | 74.8212 | 74.8212 | 26,553.98 | 7,788 | 561.3976 | 1.7813 |
| `magformer_lightdepth_mobilenetv3_esanetctx_edge_validhole` | 73.9342 | 73.5803 | 26,923.31 | 7,631 | **519.2059** | **1.9260** |

## Updated ranking among lightweight candidates

| Rank | Model | Best segm AP |
| --- | --- | ---: |
| 1 | `magformer_lightdepth_mobilenetv3_sagate_edge_validhole` | 74.8344 |
| 2 | `magformer_lightdepth_mobilenetv3_spatialgate_edge_validhole` | 74.8259 |
| 3 | `magformer_lightdepth_mobilenetv3_channelattn_edge` | 74.8212 |
| 4 | `magformer_lightdepth_mobilenetv3_directadd_edge` | 74.7425 |
| 5 | `magformer_lightdepth_mobilenetv3_gatedadd_edge` | 74.7000 |
| 6 | `magformer_lightdepth_mobilenetv3_film_edge_validhole` | 74.4047 |
| 7 | `magformer_lightdepth_resnet18_gatedadd_edge` | 74.2844 |
| 8 | `magformer_lightdepth_mobilenetv3_esanetctx_edge_validhole` | 73.9342 |

## Interpretation

- The new lightweight attention/gating methods slightly improve AP over the original `direct_add` winner.
- The gain is small:
  - `74.8259 - 74.7425 = 0.0833 AP`
- `sagate_edge_validhole` now has the highest `best AP`.
- `direct_add_edge` still remains the simpler and faster reference among the top group.
- `spatialgate_edge_validhole` remains the most stable top candidate by `last AP`.
- `esanetctx_edge_validhole` is not the best AP candidate, but it is the fastest among the new follow-up methods.

## Current recommendation

- Best accuracy candidate:
  - `magformer_lightdepth_mobilenetv3_spatialgate_edge_validhole`
- Best simplicity / throughput candidate:
  - `magformer_lightdepth_mobilenetv3_directadd_edge`
- Highest peak AP candidate:
  - `magformer_lightdepth_mobilenetv3_sagate_edge_validhole`

## Remaining gap

- Gap to `magformer_depthnorm_on`:
  - `77.4998 - 74.8344 = 2.6654 AP`
- This is slightly better than the previous best lightweight gap, but still above the `<= 2.0 AP` strong criterion.
