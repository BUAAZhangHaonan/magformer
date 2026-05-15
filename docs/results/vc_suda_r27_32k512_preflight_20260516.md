# VC-SUDA R27 32K_512 Preflight - 2026-05-16

## Purpose

Validate the newly materialized `20260318_1K_32254_512` dataset and prove that a low-cost 512 training/eval chain can run before launching any long experiment.

## Data Gate

Command:

```bash
python tools/validate_derived_dataset.py \
  --dataset-root magformer_datasets/20260318_1K_32254_512 \
  --sample-images-per-split 10 \
  --sample-anns-per-split 120
```

Result: pass.

Summary:

- train: 25,654 images, 1,397,990 annotations
- val: 3,276 images, 181,634 annotations
- test: 3,324 images, 176,328 annotations
- `cache/preprocess_manifest.json` exists and points to the 512 RGB/depth stats.
- RGB/depth/depth_xyz/instance_map shape sampling passed.
- decoded mask, bbox, and area sampling passed.

## MSDeformAttn Finding

The default CUDA backend is currently not usable in this environment. A synthetic call to `MSDeformAttn` fails before any dataset/model-specific logic:

```text
error in ms_deformable_im2col_cuda: an illegal memory access was encountered
RuntimeError: CUDA error: CUBLAS_STATUS_EXECUTION_FAILED ...
```

This reproduces on multiple GPUs and also on the old `pseudo_real_512` data path. Rebuilding the tracked extension with CUDA 12.3 did not fix it. The failure is therefore not caused by R26 data, target labels, or the 512 smoke config.

The code now fails fast for invalid backend settings and only uses the PyTorch implementation when explicitly requested:

```bash
MAGFORMER_MS_DEFORM_ATTN_BACKEND=pytorch
```

This is an explicit backend selection, not a silent fallback.

## Smoke Config

Config:

```text
configs/vc_suda_stage_a_r26_32k_512_smoke.yaml
```

Key properties:

- source dataset: `magformer_datasets/20260318_1K_32254_512`
- source annotation: `annotations/instances_train.json`
- image size: `512`
- RGB stats: generated from R26 train split
- depth clip: p1/p99 from R26 depth stats
- warm-start: matching Stage A 512 checkpoint `checkpoint_iter_0002527.pth`
- `max_iter: 2`
- `eval_iou_types: [bbox]`
- `eval_max_images: 16`
- single GPU smoke: `gpus: [4]`, `ddp_enabled: false`

## Smoke Result

Command ran in tmux session:

```text
r27_stage_a_32k512_smoke_pytorch_msda_20260516_031235
```

Command environment:

```bash
MAGFORMER_MS_DEFORM_ATTN_BACKEND=pytorch
```

Result: pass, `EXIT_CODE:0`.

Observed chain:

- config validation: pass
- dataset load: pass
- warm-start load: 729/729 keys, no missing/unexpected keys
- two train iterations completed
- bbox-only eval on 16 val images completed
- checkpoint/eval artifacts written under `output/vc_suda/stage_a_r26_32k_512_smoke`

The 16-image bbox AP is near zero and is not an experiment metric. This smoke only proves the 512 R26 data and training/eval plumbing can execute when the MSDeformAttn backend is explicitly set to PyTorch.

## Next Gate

Before any long 512 run, either:

1. repair the CUDA MSDeformAttn extension and pass the synthetic CUDA op test, or
2. accept the explicit PyTorch backend cost and run only a short bounded ablation first.

Do not launch a long 32K_512 experiment with the default CUDA backend in the current environment.
