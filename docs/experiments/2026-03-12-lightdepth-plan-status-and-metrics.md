# 2026-03-12 LightDepth Plan Status And Metrics

## Unified Metrics Table

Protocol focus: `0831_1K / 1024 / 20 epochs`

| Model | Best segm AP | Last segm AP | Peak memory (MB) | Wall time (sec) | Trainable params | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `magformer_depthnorm_on` | 77.4998 | 77.3394 | - | - | 79,696,062 | fixed baseline |
| `mgm_mask2former_depthnorm_on` | 77.3932 | 77.3465 | - | - | 79,696,062 | fixed baseline |
| `magformer_lightdepth_mobilenetv3_directadd_edge` | 74.7425 | 74.7000 | 26,654.61 | 7,307 | 49,912,142 | Stage A best |
| `magformer_lightdepth_mobilenetv3_gatedadd_edge` | 74.7000 | 74.5697 | 27,675.91 | 7,456 | 52,034,512 | Stage A top-2 |
| `magformer_lightdepth_mobilenetv3_film_edge_validhole` | 74.4047 | 74.2559 | 27,551.66 | 7,403 | 49,747,238 | Stage A |
| `magformer_lightdepth_resnet18_gatedadd_edge` | 74.2844 | 73.7688 | 28,183.59 | 7,316 | 51,820,968 | Stage A |
| `official_mask2former_pretrained` | 73.0499 | 73.0499 | - | 7,708 | 44,056,196 | baseline |
| `magformer_lightdepth_mobilenetv3_crossattn_edge_validhole_variance` | 71.0372 | 70.9806 | 26,884.93 | 8,098 | 50,061,710 | Stage B best |
| `magformer_lightdepth_mobilenetv3_crossattn_edge` | 70.6976 | 70.3714 | 27,068.74 | 7,513 | 50,061,134 | Stage B |
| `magformer_lightdepth_mobilenetv3_crossattn_edge_validhole` | 70.5774 | 70.5464 | 27,027.91 | 7,713 | 50,061,518 | Stage B |
| `magformer_nodpth_ref` | 70.4891 | 70.3317 | - | - | 79,628,990 | RGB-only reference |
| `maskrcnn_pretrained` | 62.1665 | 62.1643 | - | 1,706 | 44,024,278 | baseline |
| `mgm_mask2former_nodpth_ref` | 54.6420 | 54.4859 | - | - | 79,628,990 | diagnostic ref |
| `yolov8_seg_pretrained` | 51.6830 | 49.8496 | - | 463 | 3,263,811 | baseline |
| `maskrcnn_scratch` | 48.4830 | 48.4406 | - | 1,816 | 44,024,278 | baseline |
| `yolov8_seg_scratch` | 35.3430 | 32.2247 | - | 468 | 3,263,811 | baseline |
| `official_mask2former_scratch` | 28.7189 | 28.4250 | - | 7,253 | 44,056,196 | baseline |
| `uoais_scratch` | 16.0916 | 15.8524 | - | 4,748 | 81,686,167 | baseline |
| `unet_semantic_inst` | 11.0193 | 9.0530 | - | 1,495 | 1,942,594 | baseline |
| `unet_boundary_inst` | 7.6994 | 2.9490 | - | 5,685 | 1,942,594 | baseline |
| `unetpp_boundary_inst` | 6.6712 | 5.9291 | - | 5,005 | 529,026 | baseline |
| `ucn_scratch` | 0.0032 | 0.0032 | - | 5,527 | 42,635,008 | low-value baseline |

## LightDepth Outcome

- Best new model overall: `magformer_lightdepth_mobilenetv3_directadd_edge`
- Best `cross_attn` model: `magformer_lightdepth_mobilenetv3_crossattn_edge_validhole_variance`
- Gap from best new model to `magformer_depthnorm_on`: `77.4998 - 74.7425 = 2.7573 AP`
- Current final recommendation:
  - primary: `magformer_lightdepth_mobilenetv3_directadd_edge`
  - secondary: `magformer_lightdepth_mobilenetv3_gatedadd_edge`

## Plan Status

### Implemented

- `arch.py` config plumbing for `depth_mode`, lightweight depth backbone construction, and `res3`-only depth flow
- `fusion.py` support for `direct_add`, `gated_add`, `film`, and `cross_attn`
- `prior.compute_on = res3` default for the light-depth configs
- Stage A full 20-epoch experiments and top-2 selection
- Stage B full 20-epoch cross-attention experiments
- Separate Stage A / Stage B summaries, overlay generation, checkpoint pruning, and final docs
- Unified backbone structure for `ConvNeXtDepth`, `MobileNetV3Depth`, and `ResNetDepth`

### Partially Implemented

- `magformer/models/common/pixel_decoder_msdeformattn.py` decoupling:
  - Light-depth runs avoid the old DPE coupling by disabling DPE in configs
  - The file itself still depends on `confidence_maps` when DPE is enabled
- Stage C:
  - Final candidates were selected
  - There is no separate dedicated `stage_c` output root or rerun package

### Not Implemented

- Dedicated config file:
  - `configs/magformer_0831_1k_20ep_1024_lightdepth_crossattn.yaml`
  - Stage B currently uses runner overrides instead of a standalone YAML
- Full candidate coverage from the broader component menu:
  - `MobileNetV3-Large`
  - explicit `ConvNeXt-lite` experiment configs/runs
  - separate `F5 prior-guided cross-attention` mode as a distinct experiment track
- Complete resource-baseline closure:
  - older baseline artifacts still lack standardized `peak_memory_mb`
  - strict `1.5x` budget verification versus `magformer_nodpth_ref` is therefore not fully closed from recorded artifacts alone

## Acceptance Criteria Check

- Success criterion `best segm AP > magformer_nodpth_ref`: satisfied by all new Stage A and Stage B candidates
- Success criterion `peak memory within budget`: not fully auditable against the old RGB-only reference because the historical reference lacks standardized peak-memory artifacts
- Success criterion `wall time within budget`: same limitation as above for strict relative comparison
- Strong criterion `AP gap <= 2.0` versus `magformer_depthnorm_on`: not satisfied
- Dominant criterion `match/exceed upper bound while <= 60% memory`: not satisfied
