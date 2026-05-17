# R98 MagFormer 32K Source Short - 2026-05-18

Conclusion: pending. Training is launched from Teacher8499 with MagFormer RGB-D, 32K SQLite cache, supervised-only, and 1024 input size.

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

Launch pending at first config commit. Intended tmux session: `r98_magformer_32k_source_short`.

## Source First50 Metrics

| checkpoint | bbox AP | segm AP | decision | eval dir |
| --- | ---: | ---: | --- | --- |
| iter 500 | pending | pending | pending | pending |
| iter 1000 | pending | pending | pending | pending |
| iter 1500 | pending | pending | pending | pending |

## Notes

- Gate: stop after iter 500 if source first50 segm AP is below `0.55` and clearly below the R91 line.
- Output, checkpoints, logs, and diagnostics are not meant to be committed.
