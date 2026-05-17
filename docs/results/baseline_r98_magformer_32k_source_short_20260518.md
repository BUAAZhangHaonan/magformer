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

## Notes

- Gate: stop after iter 500 if source first50 segm AP is below `0.55` and clearly below the R91 line.
- R98 failed this gate with source first50 segm AP `0.526974`, so it was not run to 1000/1500.
- External eval protocol check passed with `--allow-nondefault-weights` for the R98 checkpoint.
- Output, checkpoints, logs, and diagnostics are not meant to be committed.
