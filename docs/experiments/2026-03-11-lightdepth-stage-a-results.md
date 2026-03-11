# 2026-03-11 LightDepth Stage A Results

## Scope

- Protocol: `0831_1K / 1024 / 20 epochs`
- Output root: `output/experiments/0831_1k_20ep_1024_lightdepth_stage_a`
- Baseline reused via symlink:
  - `magformer_nodpth_ref`
  - source: `output/experiments/0831_1k_20ep_1024_depth_revisit/magformer_nodpth_ref`

## Candidates

1. `magformer_nodpth_ref`
2. `magformer_lightdepth_mobilenetv3_directadd_edge`
3. `magformer_lightdepth_mobilenetv3_gatedadd_edge`
4. `magformer_lightdepth_resnet18_gatedadd_edge`
5. `magformer_lightdepth_mobilenetv3_film_edge_validhole`

## Metrics

| Model | Best segm AP | Last segm AP | Peak memory (MB) | Wall time (sec) | Trainable params |
| --- | ---: | ---: | ---: | ---: | ---: |
| `magformer_nodpth_ref` | 70.4891 | 70.3317 | - | - | 79,628,990 |
| `magformer_lightdepth_mobilenetv3_directadd_edge` | 74.7425 | 74.7000 | 26,654.61 | 7,307 | 49,912,142 |
| `magformer_lightdepth_mobilenetv3_gatedadd_edge` | 74.7000 | 74.5697 | 27,675.91 | 7,456 | 52,034,512 |
| `magformer_lightdepth_resnet18_gatedadd_edge` | 74.2844 | 73.7688 | 28,183.59 | 7,316 | 51,820,968 |
| `magformer_lightdepth_mobilenetv3_film_edge_validhole` | 74.4047 | 74.2559 | 27,551.66 | 7,403 | 49,747,238 |

## Stage A decision

- All 4 light-depth candidates beat `magformer_nodpth_ref`.
- Top 2 by `best segm AP`:
  1. `magformer_lightdepth_mobilenetv3_directadd_edge`
  2. `magformer_lightdepth_mobilenetv3_gatedadd_edge`
- Best encoder family for Stage B:
  - `MobileNetV3-Small`

## Notes

- `direct_add` slightly edges out `gated_add` on AP while also using less peak memory.
- `ResNet18` did not justify its extra memory over the MobileNetV3-Small runs.
- `FiLM` remained competitive but still trailed the top two MobileNetV3-Small variants.
- Baseline `magformer_nodpth_ref` was reused from the existing 1024 revisit suite, so `peak_memory_mb` and `wall_time_sec` are unavailable in this Stage A summary.

## Artifacts

- Suite summary:
  - `output/experiments/0831_1k_20ep_1024_lightdepth_stage_a/summary_0831_1k_20ep_1024_lightdepth_stage_a.json`
- Suite overlays:
  - `output/experiments/0831_1k_20ep_1024_lightdepth_stage_a/visualizations/`

## Next step

- Implement and evaluate Stage B `cross_attn(res3)` using `MobileNetV3-Small` as the depth encoder.
