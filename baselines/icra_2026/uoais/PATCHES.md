# Patches (Local)

This directory vendors UOAIS/AdelaiDet baseline code under `baselines/icra_2026/uoais/`.

To make it usable as a baseline in this workspace (ECC `0831_1K` COCO modal RGBD), we apply a few **minimal** patches.
These are intentionally small and isolated to data loading / import ergonomics.

## 2026-02-19

### 1) Allow importing `adet` utilities without compiled ops

- File: `baselines/icra_2026/uoais/adet/__init__.py`
- Change: wrap `from adet import modeling` in `try/except`
- Why:
  - `adet.modeling` imports custom ops (`adet._C`) that are only available after
    `pip install -e baselines/icra_2026/uoais` builds them.
  - Unit tests and lightweight utilities (dataset mappers, mask conversion) should not hard-fail
    when ops are not built yet.

### 2) Avoid importing dataset mappers at `adet.data` import time

- File: `baselines/icra_2026/uoais/adet/data/__init__.py`
- Change: remove eager imports; re-export `DatasetMapperWithBasis` only if import succeeds.
- Why: keeps `adet.data.detection_utils` importable without optional deps.

### 3) Optional dependency: `pyfastnoisesimd`

- File: `baselines/icra_2026/uoais/adet/data/augmentation.py`
- Change: make `pyfastnoisesimd` optional; raise a clear error only when `PerlinDistortion` is used.
- Why: ECC baselines disable `cfg.INPUT.PERLIN_DISTORTION`, so requiring this extra compiled dep
  adds fragility for no benefit.

### 4) ECC depth `.npy` support in dataset mapper

- File: `baselines/icra_2026/uoais/adet/data/dataset_mapper.py`
- Change: if `depth_file_name` ends with `.npy`, load via `np.load(...).astype(np.float32)`.

### 5) COCO modal mask compatibility (no amodal metadata)

- File: `baselines/icra_2026/uoais/adet/data/detection_utils.py`
- Changes:
  - Default `occluded_rate` to `0.0` if missing.
  - When `amodal=False` and `visible_mask` is missing, fall back to `segmentation`.
  - Fix a bug where polygon masks referenced an undefined `image_size` variable.

