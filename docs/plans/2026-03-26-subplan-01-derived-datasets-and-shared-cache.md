# Derived Dataset And Shared Cache Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add deterministic offline builders for `20260318_1K_1566_512` and `20260318_1K_1566_256`, including rewritten annotations and reusable per-dataset cache artifacts.

**Architecture:** Build one shared Python entrypoint that reads the original dataset root, writes a derived root with resized RGB/depth/annotation assets, and then invokes existing stat cache utilities plus new reusable cache builders. Keep the public dataset structure unchanged so current dataset registration code keeps working.

**Tech Stack:** Python, OpenCV, NumPy, COCO JSON, pathlib, pytest.

---

### Task 1: Builder tests

**Files:**
- Create: `tests/test_multires_dataset_builder.py`
- Test: `tests/test_custom_dataset_stats_cache.py`

**Step 1: Write the failing test**

Cover:

- derived root is created with expected folders
- output annotations have resized width and height
- `ensure_dataset_stats` produces a distinct cache for the derived root
- shared cache manifest files are written

**Step 2: Run test to verify it fails**

Run:
`conda run -n magformer pytest -q tests/test_multires_dataset_builder.py`

Expected:
- fail because the builder module does not exist yet

**Step 3: Write minimal implementation**

Create a builder script and helper module for:

- target size parsing
- RGB resize with linear interpolation
- depth resize with nearest-neighbor or area-safe rule as chosen in code review
- COCO annotation rewrite with consistent image dimensions
- derived dataset and cache manifest writing

**Step 4: Run test to verify it passes**

Run:
`conda run -n magformer pytest -q tests/test_multires_dataset_builder.py`

Expected:
- pass

### Task 2: Cache builder wiring

**Files:**
- Modify: `scripts/analysis/ensure_dataset_stats.py`
- Create: `scripts/analysis/build_multires_dataset.py`
- Create: `scripts/analysis/ensure_dataset_preprocess_cache.py`
- Test: `tests/test_multires_dataset_builder.py`

**Step 1: Write the failing test**

Extend coverage so the builder also verifies:

- `cache/derived_dataset_manifest.json`
- `cache/preprocess_manifest.json`
- split manifest files
- offline YOLO cache directory when requested

**Step 2: Run test to verify it fails**

Run:
`conda run -n magformer pytest -q tests/test_multires_dataset_builder.py -k cache`

Expected:
- fail due to missing cache writer behavior

**Step 3: Write minimal implementation**

Add shared cache generation for:

- split image manifests
- stats manifest passthrough
- optional YOLO segmentation label export

Keep the first iteration framework-neutral where possible.

**Step 4: Run test to verify it passes**

Run:
`conda run -n magformer pytest -q tests/test_multires_dataset_builder.py`

Expected:
- pass

### Task 3: CLI smoke run on the real dataset

**Files:**
- Runtime only: `magformer_datasets/20260318_1K_1566_512`
- Runtime only: `magformer_datasets/20260318_1K_1566_256`

**Step 1: Execute the builder for 512**

Run:
`conda run -n magformer python scripts/analysis/build_multires_dataset.py --source-root /home/team/zhanghaonan/magformer/magformer_datasets/20260318_1K_1566 --target-root /home/team/zhanghaonan/magformer/magformer_datasets/20260318_1K_1566_512 --image-size 512`

Expected:
- derived root exists
- stats manifest exists

**Step 2: Execute the builder for 256**

Run:
`conda run -n magformer python scripts/analysis/build_multires_dataset.py --source-root /home/team/zhanghaonan/magformer/magformer_datasets/20260318_1K_1566 --target-root /home/team/zhanghaonan/magformer/magformer_datasets/20260318_1K_1566_256 --image-size 256`

Expected:
- derived root exists
- stats manifest exists

**Step 3: Verify cache manifests**

Run:
`conda run -n magformer python scripts/analysis/ensure_dataset_stats.py --dataset-root /home/team/zhanghaonan/magformer/magformer_datasets/20260318_1K_1566_512`

Expected:
- returns cache payload for the derived root, not the original root
