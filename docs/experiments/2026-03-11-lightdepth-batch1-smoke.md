# 2026-03-11 LightDepth Batch1 Smoke

## Scope

- Branch: `codex/light-magformer-v2`
- Goal: validate the new light-depth backbone + single-scale fusion plumbing can build and train on the 0831_1K dataset.
- Dataset: `magformer_datasets/0831_1K`

## Committed implementation nodes

1. `b4882a7` — `magformer: add light-depth backbone and fusion core`
2. `dbc3a41` — `experiments: add 1024 light-depth configs and drop ucn`
3. `9d20a02` — `visualization: unify inference overlay with yolo style`

## Validation completed before smoke

- Targeted unit/integration checks passed in the `magformer` conda env:

```bash
conda run --no-capture-output -n magformer pytest -q \
  tests/test_light_depth_backbone_spec.py \
  tests/test_light_fusion_modes.py \
  tests/test_lightdepth_configs.py \
  tests/test_yolo_dataset_cache_location.py \
  tests/test_inference_uses_yolo_overlay.py \
  tests/test_config_wiring.py \
  tests/test_magformer_ablation_switches.py \
  tests/test_optimizer_groups.py
```

- CPU dummy forward passed for:
  - `magformer_0831_1k_20ep_1024_lightdepth_mobilenetv3`
  - `magformer_0831_1k_20ep_1024_lightdepth_resnet18`
  - `magformer_0831_1k_20ep_1024_lightdepth_film`

## Smoke runs

### 1) MobileNetV3 + gated_add

- Config base: `configs/magformer_0831_1k_20ep_1024_lightdepth_mobilenetv3.yaml`
- Smoke config: derived temporary config with:
  - `max_iter=10`
  - `ims_per_batch=1`
  - `eval_period=10`
  - `checkpoint_period=10`
  - pretrained loading disabled to avoid network downloads
- Result: training + validation completed successfully
- Output: `output/smoke_runs/lightdepth_mobilenetv3_smoke10`
- Notes:
  - model params: `49,871,164`
  - smoke val AP remained `0.0`, which is acceptable for this random-init plumbing check

### 2) ResNet18 + gated_add

- Config base: `configs/magformer_0831_1k_20ep_1024_lightdepth_resnet18.yaml`
- Smoke config: derived temporary config with:
  - `max_iter=2`
  - `ims_per_batch=1`
  - `eval/checkpoint` disabled for speed
  - pretrained loading disabled
- Result: 2 training iterations completed successfully
- Output: `output/smoke_runs/lightdepth_resnet18_smoke10`
- Notes:
  - model params: `49,641,212`

### 3) MobileNetV3 + FiLM

- Config base: `configs/magformer_0831_1k_20ep_1024_lightdepth_film.yaml`
- Smoke config: derived temporary config with:
  - `max_iter=2`
  - `ims_per_batch=1`
  - `eval/checkpoint` disabled for speed
  - pretrained loading disabled
- Result: 2 training iterations completed successfully
- Output: `output/smoke_runs/lightdepth_film_smoke10`
- Notes:
  - model params: `49,716,476`

## Current conclusion

- Batch 1 code path is runnable for:
  - light depth backbone construction
  - `res3`-only fusion
  - `gated_add`
  - `film`
- 1024 revisit suite no longer schedules UCN.
- Inference visualization is now on the same YOLO-style overlay path used elsewhere in the repo.

## Next recommended step

- Add a dedicated light-depth experiment runner and run Stage A candidates with pretrained warm-start enabled under the fixed 20-epoch protocol.
