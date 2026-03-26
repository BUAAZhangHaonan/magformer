# Full19 Multiresolution Baselines Design

**Date:** 2026-03-26

**Goal:** Extend scheme 1 from the repaired `ucn` and `msmformer` baselines to the full 19-model ECC suite by building offline `512x512` and `256x256` dataset variants, recomputing dataset-specific normalization/depth statistics for each variant, and precomputing reusable cache artifacts so all baselines can train and infer efficiently without relying on stale 1024-only assumptions.

## Problem

The current `full_20260318_1k_1566` suite assumes a single canonical dataset root:

- dataset root: `magformer_datasets/20260318_1K_1566`
- image geometry: `1024x1024`
- cache/stat assumptions: partly generic, partly hard-coded per register or runner

That is enough for the ongoing 1024 formal runs, but not enough for the next phase:

1. We need fair multiresolution experiments at `512x512` and `256x256`.
2. The new datasets must use their own RGB mean/std and depth clip statistics.
3. Preprocessing and postprocessing work that can be reused across runs should be prepared offline, not repeated inside every training loop.
4. The solution must cover the entire `full19` suite, not only `ucn` and `msmformer`.

## Constraints

- Do not interrupt the active 1024 `msmformer` formal run on GPU 1.
- Continue working inside the isolated worktree `feature/ucn-msmformer-repair`.
- Keep every paper-facing baseline as an instance-segmentation model.
- Preserve the existing ECC dataset contract:
  - `images/{train,val,test}`
  - `depth/depth_npy/{train,val,test}`
  - `annotations/instances_{split}.json`
- Avoid stale shared stats. Each derived dataset root must produce and use its own cache manifest.

## Baseline Scope

The target scope is the roster in `configs/experiments/full_20260318_1k_1566_roster.json`:

- `magformer_*`
- `mgm_mask2former_*`
- `mask2former`
- `maskrcnn`
- `yolov8_seg_{n,s,m,l,x}`
- `msmformer`
- `ucn`
- `uoais`
- `unet_semantic_inst`
- `unet_boundary_inst`
- `unetpp_boundary_inst`

## Approaches

### Approach A: Offline derived datasets plus shared reusable caches

Create new dataset roots:

- `magformer_datasets/20260318_1K_1566_512`
- `magformer_datasets/20260318_1K_1566_256`

Each derived dataset contains resized RGB, resized depth, rewritten COCO annotations, and a local manifest describing its source root and geometry. Then run a shared cache builder that prepares:

- dataset RGB mean/std
- dataset depth clip stats
- reusable split manifests and dataset metadata
- offline YOLO label conversion for segmentation models
- optional framework-neutral evaluation metadata needed repeatedly by reporting/inference utilities

Pros:

- Removes repeated runtime resize cost from the dataset side
- Makes normalization/depth stats unambiguous per resolution
- Keeps all baselines aligned to the same geometry for fair comparisons
- Lets runner scripts stay simple: point to a new dataset root and native image size

Cons:

- Requires one-time storage and preprocessing time
- Needs careful annotation rewrite and validation

### Approach B: Keep one dataset root and resize on the fly per model

Keep `20260318_1K_1566` as the only root and pass smaller `--image-size` values to each baseline.

Pros:

- Minimal new storage
- Fewer new files

Cons:

- Repeats resize cost in each data loader
- Some runners still assume 1024 or register-specific stats
- Harder to guarantee that every baseline uses the same derived geometry and stats

### Approach C: Per-framework custom caches only

Leave the dataset root untouched and add one cache mechanism per framework.

Pros:

- Can optimize each framework deeply

Cons:

- High maintenance burden
- Hard to keep comparable across all 19 models
- Easy to introduce inconsistent preprocessing between baselines

## Recommendation

Use **Approach A**.

It best matches the user requirement to maximize reuse and efficiency, and it gives us a clean experimental contract:

- one dataset root per resolution
- one stats manifest per dataset root
- one shared cache bundle per dataset root
- runner scripts consume dataset-native geometry instead of hiding resize work in training loops

## Derived Dataset Contract

Each derived dataset root will keep the same public structure as the original root:

- `images/{train,val,test}`
- `depth/depth_npy/{train,val,test}`
- `annotations/instances_{train,val,test}.json`
- `dataset_info.json`
- `build_stats.json`
- `alignment_report.json`
- `mask_parity_report.json`

Additional derived-only metadata will be added under a non-invasive folder:

- `cache/derived_dataset_manifest.json`
- `cache/preprocess_manifest.json`
- `cache/splits/{train,val,test}_images.json`
- `cache/yolo_seg/{train,val,test}/...`

The repo-global stats cache in `output/cache/dataset_stats/...` remains the source of truth for computed stats, but the derived dataset root will carry enough metadata to rebuild or verify its cache deterministically.

## Shared Cache Design

The offline cache layer should stay framework-neutral where possible.

### Required shared cache artifacts

- Derived dataset manifest
  - source dataset root
  - target size
  - interpolation rules
  - generation timestamp
- Cached stats manifest
  - RGB mean/std
  - depth min/max/percentiles
  - paths to cache files
- Split manifests
  - file lists and image ids for train/val/test
- Optional evaluation metadata
  - resolved val annotation path
  - resolved val image list for benchmarking/visualization tools

### Baseline-specific reusable artifacts

- YOLO segmentation labels converted offline from COCO annotations
- Any dataset-native references that are expensive and deterministic to build repeatedly

The first phase should not over-engineer cross-framework binary tensor caches unless profiling proves they matter more than the offline resized dataset itself.

## Runner Integration Rules

All `full19` runners must follow these rules:

1. Accept explicit `--register`, `--dataset-root`, and `--image-size` when applicable.
2. Use dataset-root-based stat lookup for custom roots instead of fixed `0831` or `20260318_1k_1566` shortcuts.
3. Default image size should match the dataset root's native geometry for multires runs.
4. Avoid extra on-the-fly resize when the input sample is already at the requested size.
5. Reuse derived dataset caches where available instead of regenerating them silently on each run.

## Verification Strategy

Verification should happen in four layers.

### 1. Derived dataset correctness

- image/depth files exist at expected sizes
- rewritten COCO annotations report the target width and height
- segmentation areas stay non-zero after resize

### 2. Cache correctness

- each derived dataset gets a distinct stats cache key
- RGB/depth stats are recomputed for the derived root
- cache manifests point to existing files

### 3. Runner wiring

- full19 roster can render commands for 1024, 512, and 256 tracks
- representative runners dry-run with the derived dataset roots
- known hard-coded 1024 U-Net and runner paths are removed or parameterized

### 4. Experimental sanity

- GPU 0 can build a derived dataset and launch at least one multires training job without disturbing GPU 1
- summary/reporting tools still produce unified output tables

## Risks

- Annotation resize must preserve instance masks faithfully enough for COCOeval comparability.
- Some runners may still perform internal resize/crop augmentations even when the dataset is already native-sized.
- YOLO and U-Net families may need extra cache plumbing because they do more bespoke dataset preparation than Detectron2-based baselines.
- Storage growth may be non-trivial, especially for duplicated RGB/depth assets.

## Deliverables

- `512x512` and `256x256` derived ECC dataset roots
- per-derived-dataset stats and cache manifests
- full19 runner/roster support for multires tracks
- updated tests for cache generation and runner wiring
- initial multires training launches on GPU 0 while the 1024 `msmformer` run continues on GPU 1
