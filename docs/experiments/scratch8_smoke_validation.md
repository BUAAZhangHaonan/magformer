# Scratch8 Smoke Validation (0831_1K)

- Generated: 2026-02-20 21:07:20
- Output root: `output/experiments/0831_1k_5k_scratch8_smoke`
- Artifact contract checked: `run.log`, `wall_time_sec.txt`, `params_trainable.txt`, `coco_instances_results.json`, `metrics.cocoeval.json`

| Model | Status | segm/AP (COCOeval) | bbox/AP (COCOeval) | Params | Wall Time (s) |
|---|---:|---:|---:|---:|---:|
| `magformer_scratch` | ok | 0.0 | 0.0 | 79715570 | 429.0 |
| `mgm_mask2former_scratch` | ok | 0.0 | 0.0 | 79747262 | 261.0 |
| `msmformer_scratch` | ok | 0.0 | 0.0 | 50745922 | 209.0 |
| `uoais_scratch` | ok | 0.0 | 0.0 | 81686167 | 48.0 |
| `ucn_scratch` | ok | 0.0 | 0.0 | 42635008 | 109.0 |
| `official_mask2former_scratch` | ok | 0.0 | 0.0 | 44056196 | 179.0 |
| `maskrcnn_scratch` | ok | 0.3386106239879943 | 2.1214124227396813 | 44024278 | 110.0 |
| `yolov8_seg_scratch` | ok | 10.132276199942531 | 12.764927426822878 | 3263811 | 91.0 |

## Notes

- This smoke run uses reduced budget (`MAX_ITER=20` or `epochs=1`) and is only for end-to-end contract validation.
- Full benchmark numbers must come from `output/experiments/0831_1k_5k_scratch8/` after full run.
