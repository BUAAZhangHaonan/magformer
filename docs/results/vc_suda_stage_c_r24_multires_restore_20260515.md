# VC-SUDA Stage C R24 Multires Restore - 2026-05-15

Conclusion: the 512/multires dataset conversion tool has been restored for fast validation. No training was started, and no 32K full conversion was run.

## Scope

- Restored `scripts/analysis/build_multires_dataset.py` from `02b69530b1384d980d934a2e98a9a2cb49b964ad:scripts/analysis/build_multires_dataset.py`.
- Verified restored script sha256: `c37227d3211ea9502c6f75b21a9ee2bec0cd06cd3f89b37d3e63d57d592ba6dd`.
- Included the required stats helpers because `build_multires_dataset.py` imports `ensure_dataset_stats.py`, and that module imports `compute_depth_stats_0831_1k.py` and `compute_rgb_stats_coco.py`.
- Left existing `output/` directories untracked.

## Validation

Focused pytest:

```bash
/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python -m pytest tests/test_multires_dataset_builder.py -q
```

Result: `1 passed`.

The test covers compressed COCO RLE, uncompressed RLE, polygon input, RGB/depth resizing, image width/height rewrite, mask RLE size, bbox/area legality, and non-empty decoded masks.

Small script-level validation used one real train sample from `magformer_datasets/20260318_1K_1566` and temporary directories under `/tmp`:

```bash
/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python scripts/analysis/build_multires_dataset.py \
  --source-root /tmp/r24_multires_*/source_1sample \
  --target-root /tmp/r24_multires_*/target_512 \
  --image-size 512 \
  --cache-root /tmp/r24_multires_*/stats_cache \
  --force
```

Checked output facts:

- Source image: `7448262510_XXL_50_scene_000012_000001_v0.png`.
- Input annotations: 50.
- Output annotations: 50.
- Non-empty decoded masks: 50.
- RGB size: `512x512`.
- Depth shape: `512x512`.
- Instance map max id: 50.
- Every output annotation had RLE size `[512, 512]`, positive area, and bbox inside the 512 frame.

## Why 32K Was Not Run

A full `20260318_1K_32254` conversion is an expensive materialization step and was outside this restore task. The goal here was to prove the restored tool works and is safe to use for the next 512 fast check without creating large output artifacts or committing generated data.

## Safe 32K_512 Generation Later

Run this only after choosing the final output location and confirming disk budget:

```bash
/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python scripts/analysis/build_multires_dataset.py \
  --source-root magformer_datasets/20260318_1K_32254 \
  --target-root magformer_datasets/20260318_1K_32254_512 \
  --image-size 512 \
  --cache-root output/cache/dataset_stats \
  --force
```

Before committing after that run, check `git status --short` and keep generated dataset/output artifacts untracked unless a later task explicitly asks to version them.
