# 0831_1K 20ep A/B Audit (MagFormer Export Path)

## Context
- Goal: validate whether local, uncommitted export/inference patches can introduce pathological COCO outputs.
- A/B setup:
  - A: current local patch (`magformer/models/magformer/arch.py` + `magformer/engine/coco_export.py`)
  - B: clean `HEAD` versions of the same files
  - Budget: `max_iter=20` smoke with identical config and seed
  - Dataset: `magformer_datasets/0831_1K`

## Smoke Results (20 iter)
- Local patch summary: `docs/experiments/0831_1k_20ep_ab_local_patch_smoke.json`
- HEAD baseline summary: `docs/experiments/0831_1k_20ep_ab_head_baseline_smoke.json`

| Variant | pred_count | segm/AP | bbox/AP | bbox_area_ratio p50 | mask_area std |
| --- | ---: | ---: | ---: | ---: | ---: |
| local patch smoke | 100 | 0.0 | 0.0 | 2.2888e-05 | 0.1990 |
| head baseline smoke | 100 | 0.0 | 0.0 | 2.2888e-05 | 0.0995 |

Both smoke runs are equally underfit (AP=0), so 20 iter is insufficient to separate quality.

## Historical Pathology Evidence (5k scratch run)
- File: `docs/experiments/0831_1k_5k_scratch8_magformer_pathology.json`
- Observed:
  - `pred_count=8197`
  - `bbox_area_ratio p50=1.0` (boxes nearly full-frame)
  - `mask_area std=0.0` with constant area `10485`
  - `segm/AP=0.0`

This is a canonical pathological distribution and is incompatible with healthy instance predictions.

## Decision
- Remove implicit/always-on empty-mask fallback from default export path.
- Keep fallback only as explicit debug opt-in parameter (default disabled).
- Revert MagFormer inference score fallback to standard Mask2Former-style scoring path.
- Add diagnostics in eval logs for score quantiles, mask non-empty ratio, and bbox area ratio.
