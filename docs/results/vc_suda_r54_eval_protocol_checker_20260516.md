# VC-SUDA R54 eval protocol checker - 2026-05-16

## Conclusion

Added `tools/check_eval_protocol.py` as a no-train, no-model-eval static gate for the two eval protocols that were easy to mix up:

- `original_first50_teacher`
- `pseudo_real_target_unlabeled200`

The checker locks base config, dataset root, annotation, split, image size, max images, thresholds, IoU types, inference top-k, maxDets, weights existence, depth norm/clip, and annotation image count before any eval command is allowed to run.

## Why This Exists

R44 showed that original 1.5K first50 Teacher eval can be badly corrupted by using a pseudo-real base config. That wrong mix used the pseudo-real dataset/depth path and produced the temporary low segm AP line near `0.01994`.

R53 clarified the protocol boundary:

- R44 `0.625293` segm AP is original 1.5K first50 Teacher sanity only.
- R46 `0.323252` segm AP is the current best pseudo_real `target_unlabeled200` target-domain gate.
- The `61+` AP line must not be read as a pseudo_real `target_unlabeled200` result.

From this point, run the checker before running `tools/evaluate_1024_backmap.py` or any wrapper copied from a previous eval.

## Locked Protocols

### `original_first50_teacher`

- Base config: `configs/finetune_1k_full_1024.yaml`
- Dataset root: `magformer_datasets/20260318_1K_1566`
- Annotation: `annotations/instances_all.json`
- Split: `all`
- Image size: `1024`
- Max images: `50`
- Score/mask thresholds: `0.05/0.5`
- IoU types: `bbox,segm`
- Inference top-k/maxDets: `100/100`
- Default weights: `output/experiments/20260510_1k_finetune_full_1024_v13/checkpoint_iter_0008499.pth`
- Depth config: clip `1.0015300512313843/2.095623016357422`, norm `minmax`, per-sample norm `true`
- Annotation image count: `1566`

The default requires the Teacher checkpoint exactly. Use `--allow-nondefault-weights` only for an intentional original first50 non-Teacher check.

### `pseudo_real_target_unlabeled200`

- Base config: `configs/eval_vc_suda_stage_b_1024_teacher8499_segm.yaml`
- Dataset root: `magformer_datasets/pseudo_real_512`
- Annotation: `annotations/instances_target_unlabeled.json`
- Split: `train`
- Image size: `1024`
- Max images: `null` / unset
- Score/mask thresholds: `0.05/0.5`
- IoU types: `bbox,segm`
- Inference top-k/maxDets: `200/200`
- Weights: any existing checkpoint path
- Depth config: clip `0.0/2.095623016357422`, norm `minmax`, per-sample norm `true`
- Annotation image count: `200`

## Actual Static Checks

Output directory:

- `output/diagnostics/r54_eval_protocol_checker_20260516`

Positive checks:

```bash
python tools/check_eval_protocol.py original_first50_teacher \
  --base-config configs/finetune_1k_full_1024.yaml \
  --dataset-root magformer_datasets/20260318_1K_1566 \
  --ann annotations/instances_all.json \
  --split all \
  --weights output/experiments/20260510_1k_finetune_full_1024_v13/checkpoint_iter_0008499.pth \
  --image-size 1024 \
  --max-images 50 \
  --score-threshold 0.05 \
  --mask-threshold 0.5 \
  --iou-types bbox,segm \
  --inference-topk 100 \
  --max-dets 100 \
  --summary-json output/diagnostics/r54_eval_protocol_checker_20260516/original_first50_teacher.summary.json
```

Result: pass. The summary reports `1566` images, the original depth clip, and the exact Teacher checkpoint.

```bash
python tools/check_eval_protocol.py pseudo_real_target_unlabeled200 \
  --base-config configs/eval_vc_suda_stage_b_1024_teacher8499_segm.yaml \
  --dataset-root magformer_datasets/pseudo_real_512 \
  --ann annotations/instances_target_unlabeled.json \
  --split train \
  --weights output/vc_suda/stage_c_r46_pseudoreal_source_true_resume_1024_teacher8499/checkpoint_iter_0000750.pth \
  --image-size 1024 \
  --score-threshold 0.05 \
  --mask-threshold 0.5 \
  --iou-types bbox,segm \
  --inference-topk 200 \
  --max-dets 200 \
  --summary-json output/diagnostics/r54_eval_protocol_checker_20260516/pseudo_real_target_unlabeled200.summary.json
```

Result: pass. The summary reports `200` images, pseudo-real depth clip, and existing R46 `ckpt0750` weights.

Negative check:

```bash
python tools/check_eval_protocol.py original_first50_teacher \
  --base-config configs/eval_vc_suda_stage_b_1024_teacher8499_segm.yaml \
  --dataset-root magformer_datasets/20260318_1K_1566 \
  --ann annotations/instances_all.json \
  --split all \
  --weights output/experiments/20260510_1k_finetune_full_1024_v13/checkpoint_iter_0008499.pth \
  --image-size 1024 \
  --max-images 50 \
  --score-threshold 0.05 \
  --mask-threshold 0.5 \
  --iou-types bbox,segm \
  --inference-topk 100 \
  --max-dets 100
```

Result: expected fail with exit code `2`. The log is `output/diagnostics/r54_eval_protocol_checker_20260516/negative_original_with_pseudoreal_base_config.log`, and it names the `base_config`, `base_config data.dataset_root`, and `depth config` mismatches.

## Validation

Commands run:

```bash
python tools/check_eval_protocol.py --help
python -m pytest tests/test_check_eval_protocol.py -q
python -m py_compile tools/check_eval_protocol.py tests/test_check_eval_protocol.py
git diff --check
```

No training was started. No model eval was run.
