# VC-SUDA Stage C R25 32K->512 Shard Validation - 2026-05-15

Conclusion: the 32K->512 conversion is correct on a real 50-image shard. It is safe to run the full `20260318_1K_32254` -> 512 materialization after reserving about 200GB free disk and keeping the generated dataset/output untracked. No 32K full conversion was run, and no training was started.

## Scope

- Remote host: `ssh 4029`.
- Repo: `/home/hdd3/zhanghaonan/magformer` at `9874c64`.
- Initial git state: only `output/diagnostics/` and `output/upper_bound/` were untracked.
- Source dataset: `magformer_datasets/20260318_1K_32254`.
- Shard source: first 50 train images, hardlinked under `output/diagnostics/r25_32k_512_shard_20260515/source_32k_shard50`.
- Derived target: `output/diagnostics/r25_32k_512_shard_20260515/derived_512_shard50`.

## Validation

Shard extraction:

- Images: `50`.
- Annotations: `2,095`.
- Parsed annotations before stopping: `2,096`, because train annotations are ordered by `image_id`.
- Extraction wall time: `61.97s`.
- Peak RSS: `57MB`.

512 conversion command:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OPENCV_FOR_THREADS_NUM=1 \
/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python scripts/analysis/build_multires_dataset.py \
  --source-root output/diagnostics/r25_32k_512_shard_20260515/source_32k_shard50 \
  --target-root output/diagnostics/r25_32k_512_shard_20260515/derived_512_shard50 \
  --image-size 512 \
  --cache-root output/diagnostics/r25_32k_512_shard_20260515/stats_cache \
  --force
```

Conversion result:

- Wall time: `14.68s` for 50 images.
- Average: `0.294s/image`.
- Peak RSS: `140MB`.
- CPU: single-thread run used about one full core; server-level RAM stayed far below 90%.

Artifact checks passed:

- RGB files: `50`; depth files: `50`; COCO images: `50`.
- RGB/depth shape: all `512x512`.
- Output annotations: `2,095`.
- Non-empty decoded masks: `2,095/2,095`.
- `segmentation.size`: all `[512, 512]`.
- `bbox` and `area`: exact match with decoded masks, max diff `0.0`.
- Categories: only `component`, `id=1`.
- Manifests and stats exist: `cache/derived_dataset_manifest.json`, `cache/preprocess_manifest.json`, `cache/splits/train_images.json`, RGB stats, depth stats, stats manifest.

Read-only loader sanity passed:

- `register_ecc_coco_rgbd("r25_32k_512_shard", derived_root)` loaded `50` train records with `depth_file_name` present.
- `ECCUCNDataset` + `torch.utils.data.DataLoader` loaded 3 batches, batch size 2.
- Batch tensor shapes: image `[2,3,512,512]`, depth `[2,3,512,512]`, label `[2,1,512,512]`.
- Label range in checked batches: `-1..24`.

## Full 32K Estimate

Current source split counts:

- Train: `25,654` images/depth files.
- Val: `3,276` images/depth files.
- Test: `3,324` images/depth files.
- Total: `32,254` images.

Estimate from the 50-image shard:

- Runtime: about `2.6h` at `0.294s/image`.
- Derived 512 disk: about `170GB` from `264MB / 50 images`.
- Source dataset size: `173GB`.
- Current filesystem free space: about `7.7TB` on `/home/hdd3`.

## Decision

Full `32K_512` generation is allowed from a correctness and disk-budget view. Use the single-thread environment variables above if the resource cap is strict, monitor system CPU/RAM during the full run, and do not commit generated output.

Residual risk: this shard covered only the first 50 train images. It validates the real COCO/RLE path and loader path, but it does not prove every later image file or depth file in val/test exists. A full run should still fail fast on missing files if any exist.
