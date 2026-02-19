# Patches (Local)

This directory vendors MSMFormer baseline code under `baselines/icra_2026/msmformer/`.

The upstream project mixes model code, dataset pipelines, and external checkpoints.
In this workspace we keep **model-only** code and apply small patches to make it runnable
for ECC `0831_1K` scratch baselines.

## 2026-02-19

### 1) Make `meanshiftformer` importable without vendored dataset modules

- File: `baselines/icra_2026/msmformer/MSMFormer/meanshiftformer/__init__.py`
- Change: remove eager imports of `.data` and upstream dataset mappers; keep only modeling/meta-arch/config exports.
- Why: the vendored checkout does not include upstream dataset code, but Detectron2 still needs registry side effects.

### 2) Allow scratch backbone init (no external UCN checkpoints)

- File: `baselines/icra_2026/msmformer/lib/fcn/get_network_crop.py`
- Changes:
  - remove an unused import of `.test_utils` that pulled in non-vendored `utils.*` modules at import time
  - `get_backbone()` no longer `sys.exit()` when checkpoint is missing/empty; it falls back to random init
  - `get_backbone_crop()` now checks file existence before `torch.load`

### 3) ECC runner wrapper

- File: `baselines/run_msmformer_0831_1k.py`
- Adds:
  - ECC RGBD COCO registration (adds `.npy` `depth_file_name`)
  - minimal RGBD DatasetMapper compatible with the UCN-style backbone
  - writes `params_trainable.txt` in the output dir for suite summarization

