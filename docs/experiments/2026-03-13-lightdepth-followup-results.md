# 2026-03-13 LightDepth Follow-up Results

## Scope

- New literature-inspired fusion candidates added on top of the existing light-depth Stage A line:
  - `magformer_lightdepth_mobilenetv3_channelattn_edge`
  - `magformer_lightdepth_mobilenetv3_spatialgate_edge_validhole`
- Output root:
  - `output/experiments/0831_1k_20ep_1024_lightdepth_stage_a`

## New candidates

| Model | Best segm AP | Last segm AP | Peak memory (MB) | Wall time (sec) | Infer ms | Infer FPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `magformer_lightdepth_mobilenetv3_channelattn_edge` | 74.8212 | 74.8212 | 26,553.98 | 7,788 | 624.4580 | 1.6014 |
| `magformer_lightdepth_mobilenetv3_spatialgate_edge_validhole` | **74.8259** | **74.8259** | **26,497.73** | 7,798 | 633.2295 | 1.5792 |

## Updated ranking among lightweight candidates

| Rank | Model | Best segm AP |
| --- | --- | ---: |
| 1 | `magformer_lightdepth_mobilenetv3_spatialgate_edge_validhole` | 74.8259 |
| 2 | `magformer_lightdepth_mobilenetv3_channelattn_edge` | 74.8212 |
| 3 | `magformer_lightdepth_mobilenetv3_directadd_edge` | 74.7425 |
| 4 | `magformer_lightdepth_mobilenetv3_gatedadd_edge` | 74.7000 |
| 5 | `magformer_lightdepth_mobilenetv3_film_edge_validhole` | 74.4047 |
| 6 | `magformer_lightdepth_resnet18_gatedadd_edge` | 74.2844 |

## Interpretation

- The new lightweight attention/gating methods slightly improve AP over the original `direct_add` winner.
- The gain is small:
  - `74.8259 - 74.7425 = 0.0833 AP`
- `spatialgate_edge_validhole` is now the best-accuracy lightweight candidate.
- `direct_add_edge` still remains the simpler and faster reference among the top group.

## Current recommendation

- Best accuracy candidate:
  - `magformer_lightdepth_mobilenetv3_spatialgate_edge_validhole`
- Best simplicity / throughput candidate:
  - `magformer_lightdepth_mobilenetv3_directadd_edge`

## Remaining gap

- Gap to `magformer_depthnorm_on`:
  - `77.4998 - 74.8259 = 2.6739 AP`
- This is slightly better than the previous best lightweight gap, but still above the `<= 2.0 AP` strong criterion.
