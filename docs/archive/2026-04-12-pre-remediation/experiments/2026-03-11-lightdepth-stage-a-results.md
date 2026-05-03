# 2026-03-11 LightDepth Stage A Results

## Scope

- Protocol: `0831_1K / 1024 / 20 epoch`
- Output root: `output/experiments/0831_1k_20ep_1024_lightdepth_stage_a`
- RGB-only control: reused existing `magformer_nodpth_ref`

## Stage A candidates

1. `magformer_nodpth_ref`
2. `magformer_lightdepth_mobilenetv3_directadd_edge`
3. `magformer_lightdepth_mobilenetv3_gatedadd_edge`
4. `magformer_lightdepth_resnet18_gatedadd_edge`
5. `magformer_lightdepth_mobilenetv3_film_edge_validhole`

## Results

| Model ID | Best segm AP | Last segm AP | Peak memory (MB) | Wall time (sec) | Trainable params |
|---|---:|---:|---:|---:|---:|
| `magformer_nodpth_ref` | 70.49 | 70.33 | - | - | 79,628,990 |
| `magformer_lightdepth_mobilenetv3_directadd_edge` | 74.74 | 74.70 | 26,654.61 | 7,307 | 49,912,142 |
| `magformer_lightdepth_mobilenetv3_gatedadd_edge` | 74.70 | 74.57 | 27,675.91 | 7,456 | 52,034,512 |
| `magformer_lightdepth_mobilenetv3_film_edge_validhole` | 74.40 | 74.26 | 27,551.66 | 7,403 | 49,747,238 |
| `magformer_lightdepth_resnet18_gatedadd_edge` | 74.28 | 73.77 | 28,183.59 | 7,316 | 51,820,968 |

## Decision

All four lightweight RGB-D candidates beat the RGB-only control.

Top 2 by AP:

1. `magformer_lightdepth_mobilenetv3_directadd_edge`
2. `magformer_lightdepth_mobilenetv3_gatedadd_edge`

Selected depth encoder for Stage B:

- `MobileNetV3-Small`

## Notes

- The strongest Stage A model is `MobileNetV3-Small + Direct Add + [edge]`.
- `Gated Add` is extremely close to `Direct Add`, so both are worth keeping in view while introducing cross-attention.
- `FiLM` remains competitive, but not enough to displace the top-2 AP results.
- `ResNet18` did not beat the best MobileNetV3 candidate, so it should not be the first encoder promoted into Stage B.

## Artifacts

- Summary: `output/experiments/0831_1k_20ep_1024_lightdepth_stage_a/summary_0831_1k_20ep_1024_lightdepth_stage_a.json`
- Overlays: `output/experiments/0831_1k_20ep_1024_lightdepth_stage_a/visualizations`
