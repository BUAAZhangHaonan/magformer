# R98 MagFormer 32K Source Short - 2026-05-18

Conclusion: stopped at the iter 500 gate. R98 MagFormer RGB-D 32K cache supervised-only starts cleanly from Teacher8499, but original first50 source sanity at `checkpoint_iter_0000499.pth` is below the `0.55` gate: segm AP `0.526974`.

## Config

- Config: `configs/baseline_supervised_r98_magformer_32k1024_cache_teacher8499_short.yaml`
- Data root: `magformer_datasets/20260318_1K_32254`
- Train cache: `cache/coco_loader/instances_train.sqlite`
- Warm start: `output/experiments/20260510_1k_finetune_full_1024_v13/checkpoint_iter_0008499.pth`
- VC-SUDA, EMA, pseudo, offline bank, contrastive, and unsupervised losses: disabled.

## Validation

Completed before launch.

- Static config: `MAGFORMER_COCO_LOADER_CACHE_VERIFY_SOURCE_HASH=0 python - <<'PY' ... load_config + validate_config ... PY` passed.
- Loader smoke: `MAGFORMER_COCO_LOADER_CACHE_VERIFY_SOURCE_HASH=0 python - <<'PY' ... build_datasets/build_data_loaders ... PY` passed.
- Cache test: `python -m pytest tests/test_coco_loader_cache.py -q` passed, `6 passed`.
- Train loader evidence: `CocoLoaderCache`, `25654` train images, batch RGB `(4, 3, 1024, 1024)`, depth `(4, 1, 1024, 1024)`, first target masks `(25, 1024, 1024)`.
- Runtime cache setting: use `MAGFORMER_COCO_LOADER_CACHE_VERIFY_SOURCE_HASH=0` so 32K train opens SQLite without re-reading the 9.89GB source JSON.
- GPU preflight: GPUs 4-7 were idle at `15 MiB` used and `0%` util before launch.

## Training

- tmux session: `r98_magformer_32k_source_short`.
- Launch command shape: `MAGFORMER_COCO_LOADER_CACHE_VERIFY_SOURCE_HASH=0 torchrun --nproc_per_node=4 tools/train.py --config configs/baseline_supervised_r98_magformer_32k1024_cache_teacher8499_short.yaml`.
- Warm-start load: `missing=0`, `unexpected=0`.
- Saved checkpoint: `output/baseline/r98_magformer_32k_source_short/checkpoint_iter_0000499.pth`.
- Built-in eval at iter 499: bbox AP `0.8006`, segm AP `0.8362` on the config val subset. This is not the gate result because the required gate is the fixed original first50 external protocol.
- Stop point: training was interrupted after the external gate failed; the latest logged train iter was `627`.

## Source First50 Metrics

| checkpoint | bbox AP | segm AP | decision | eval dir |
| --- | ---: | ---: | --- | --- |
| iter 500 (`checkpoint_iter_0000499.pth`) | `0.582708` | `0.526974` | fail, stopped | `output/diagnostics/r98_magformer_32k_source_short_ckpt0499_original_first50_20260518` |
| iter 1000 | not run | not run | skipped after gate fail | n/a |
| iter 1500 | not run | not run | skipped after gate fail | n/a |

## 32K Val / Subset Diagnostic

Conclusion: the original first50 gate was probably too strict for R98. It failed the old source-first50 check, but the same `checkpoint_iter_0000499.pth` has a strong learning signal on the matching 32K val protocol. This is not a final full-val number because only the first `300 / 3276` val images were evaluated.

Command:

```bash
CUDA_VISIBLE_DEVICES=4 MAGFORMER_MS_DEFORM_ATTN_BACKEND=pytorch \
/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python tools/evaluate_1024_backmap.py \
  --base-config configs/baseline_supervised_r98_magformer_32k1024_cache_teacher8499_short.yaml \
  --dataset-root magformer_datasets/20260318_1K_32254 \
  --ann annotations/instances_val.json \
  --split val \
  --weights output/baseline/r98_magformer_32k_source_short/checkpoint_iter_0000499.pth \
  --output-dir output/diagnostics/r98_magformer_32k_val_subset_or_full_20260518 \
  --image-size 1024 \
  --batch-size 4 \
  --num-workers 0 \
  --score-threshold 0.05 \
  --mask-threshold 0.5 \
  --iou-types bbox,segm \
  --inference-topk 100 \
  --max-dets 100 \
  --max-images 300 \
  --dump-inference-stats output/diagnostics/r98_magformer_32k_val_subset_or_full_20260518/inference_stats.json \
  --force-pytorch-msda
```

Metrics:

| eval | images | bbox AP | bbox AP50 | bbox AP75 | segm AP | segm AP50 | segm AP75 | speed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 32K val subset | 300 | `0.770478` | `0.920889` | `0.805208` | `0.810410` | `0.958792` | `0.885158` | about `211s` wall, `1.42 img/s` end-to-end |

Protocol notes:

- Eval entry point: `tools/evaluate_1024_backmap.py`.
- Dataset: `magformer_datasets/20260318_1K_32254/annotations/instances_val.json`, `3276` val images.
- Geometry: 1024 input with backmap to GT size.
- Thresholds: score `0.05`, mask `0.5`.
- Top-k/maxDets: `100/100`, matching the original first50 source protocol and COCO-style `maxDets=100`.
- `tools/check_eval_protocol.py` does not currently define a fixed 32K val contract, so there is no protocol-checker pass/fail for this 32K val subset. The target_unlabeled200 checker below did pass.
- Inference stats show `17699` exported predictions and `300/300` images hitting the raw top-k cap. This does not block the diagnostic result, but a final full 32K val run can reasonably repeat with `topk/maxDets=200` if the gate wants uncapped dense-source coverage.

Interpretation:

- The first50 source gate said segm AP `0.526974`, below the old `0.55` stop line.
- The comparable 32K val subset says segm AP `0.810410`, far above a "no learning signal" result.
- R97 official RGB-only was judged on the 32K val protocol, not old original first50. By that standard, stopping R98 only from original first50 was probably a false negative.

## Target Unlabeled200 Secondary Check

This was run only after the 32K val subset finished. It confirms that strong source-val learning does not transfer to target_unlabeled200 yet.

Protocol checker:

```bash
/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python tools/check_eval_protocol.py pseudo_real_target_unlabeled200 \
  --base-config configs/eval_vc_suda_stage_b_1024_teacher8499_segm.yaml \
  --dataset-root magformer_datasets/pseudo_real_512 \
  --ann annotations/instances_target_unlabeled.json \
  --split train \
  --weights output/baseline/r98_magformer_32k_source_short/checkpoint_iter_0000499.pth \
  --image-size 1024 \
  --score-threshold 0.05 \
  --mask-threshold 0.5 \
  --iou-types bbox,segm \
  --inference-topk 200 \
  --max-dets 200 \
  --summary-json output/diagnostics/r98_magformer_32k_target_unlabeled200_20260518/protocol_check.json
```

Checker status: pass.

Eval command:

```bash
CUDA_VISIBLE_DEVICES=4 MAGFORMER_MS_DEFORM_ATTN_BACKEND=pytorch \
/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python tools/evaluate_1024_backmap.py \
  --base-config configs/eval_vc_suda_stage_b_1024_teacher8499_segm.yaml \
  --dataset-root magformer_datasets/pseudo_real_512 \
  --ann annotations/instances_target_unlabeled.json \
  --split train \
  --weights output/baseline/r98_magformer_32k_source_short/checkpoint_iter_0000499.pth \
  --output-dir output/diagnostics/r98_magformer_32k_target_unlabeled200_20260518 \
  --image-size 1024 \
  --batch-size 4 \
  --num-workers 0 \
  --score-threshold 0.05 \
  --mask-threshold 0.5 \
  --iou-types bbox,segm \
  --inference-topk 200 \
  --max-dets 200 \
  --dump-inference-stats output/diagnostics/r98_magformer_32k_target_unlabeled200_20260518/inference_stats.json \
  --force-pytorch-msda
```

Metrics:

| eval | images | bbox AP | bbox AP50 | bbox AP75 | segm AP | segm AP50 | segm AP75 | speed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| target_unlabeled200 | 200 | `0.0000007723` | `0.0000019308` | `0.0000000000` | `0.0000009269` | `0.0000050722` | `0.0000000000` | `82.5s`, `2.42 img/s` inference |

Target note: top-k/maxDets `200/200` had `0/200` top-k-truncated images, so the near-zero AP is not from the cap.

## Notes

- Gate: stop after iter 500 if source first50 segm AP is below `0.55` and clearly below the R91 line.
- R98 failed this gate with source first50 segm AP `0.526974`, so it was not run to 1000/1500.
- External eval protocol check passed with `--allow-nondefault-weights` for the R98 checkpoint.
- The new 32K val subset diagnostic shows source-domain learning is present at iter 499. The old first50 gate is not a reliable stop rule for this R98/R97 comparison.
- Output, checkpoints, logs, and diagnostics are not meant to be committed.
