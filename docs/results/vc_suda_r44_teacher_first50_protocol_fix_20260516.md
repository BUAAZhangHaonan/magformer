# VC-SUDA R44 Teacher first50 protocol fix - 2026-05-16

## Conclusion

The original 1.5K first50 Teacher eval must use `configs/finetune_1k_full_1024.yaml`. The temporary low-AP run mixed the original 1.5K data with the pseudo-real Stage B base config, so it used the wrong depth normalization path.

## Root Cause

- Bad temporary run: `/tmp/magformer_taskB_backmap_original_50`.
- Wrong base config: `configs/vc_suda_stage_b_1024_teacher8499.yaml`.
- Bad reported segm AP: `0.0199377460`.
- Correct base config: `configs/finetune_1k_full_1024.yaml`.

The Stage B config is a pseudo-real config. It points at `magformer_datasets/pseudo_real_512`, `annotations/instances_val.json`, split `val`, and has `data.depth.clip_min: 0.0`. The original Teacher first50 protocol points at `magformer_datasets/20260318_1K_1566`, `annotations/instances_all.json`, split `all`, and uses `data.depth.clip_min: 1.0015300512313843`, `data.depth.clip_max: 2.095623016357422`.

## Fix

Added `tools/evaluate_teacher_first50_1024_backmap.py` as the stable wrapper for this exact protocol. It does not change the core evaluator. It fails before eval if the base config, dataset root, annotation, split, image size, max image count, or depth clip do not match the original 1.5K first50 Teacher protocol.

Rules enforced by the wrapper:

- `--base-config` must be `configs/finetune_1k_full_1024.yaml`.
- `--dataset-root` must be `magformer_datasets/20260318_1K_1566`.
- `--ann` must be `annotations/instances_all.json`.
- `--split` must be `all`.
- `--image-size` must be `1024`.
- `--max-images` must be `50`.
- `--iou-types` must be `bbox,segm`.
- Depth clip must be `1.0015300512313843/2.095623016357422`.
- The wrapper always passes `--force-pytorch-msda` and runs the child eval with `MAGFORMER_MS_DEFORM_ATTN_BACKEND=pytorch`.

## Corrected Run

```bash
CUDA_VISIBLE_DEVICES=4 \
OMP_NUM_THREADS=4 \
MKL_NUM_THREADS=4 \
NUMEXPR_NUM_THREADS=4 \
MAGFORMER_MS_DEFORM_ATTN_BACKEND=pytorch \
python tools/evaluate_teacher_first50_1024_backmap.py \
  --weights output/experiments/20260510_1k_finetune_full_1024_v13/checkpoint_iter_0008499.pth \
  --output-dir output/diagnostics/r44_teacher_first50_protocol_fix_20260516 \
  --batch-size 4
```

Output files:

- `output/diagnostics/r44_teacher_first50_protocol_fix_20260516/eval.log`
- `output/diagnostics/r44_teacher_first50_protocol_fix_20260516/eval_1024_runtime.yaml`
- `output/diagnostics/r44_teacher_first50_protocol_fix_20260516/metrics.cocoeval.json`
- `output/diagnostics/r44_teacher_first50_protocol_fix_20260516/coco_instances_results.json`

## Corrected Metrics

| Metric | Value |
| --- | ---: |
| bbox AP | `0.6519530084` |
| bbox AP50 | `0.8531437401` |
| bbox AP75 | `0.7175465372` |
| segm AP | `0.6252930576` |
| segm AP50 | `0.8679527609` |
| segm AP75 | `0.7243386328` |

This reproduces the known-good first50 Teacher line. It is far above the bad temporary segm AP `0.0199377460`.

## Do Not Mix

Do not evaluate original 1.5K first50 Teacher with `configs/vc_suda_stage_b_1024_teacher8499.yaml` or any pseudo-real base config. Do not copy commands from pseudo-real diagnostics and only replace the dataset arguments. The base config owns the depth normalization, so the base config must match the data protocol.
