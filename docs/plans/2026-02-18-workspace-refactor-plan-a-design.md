# Workspace Refactor Plan A (Hygiene + Output Canonicalization + Baselines Consolidation)

Date: 2026-02-18

## Summary

This document defines **Plan A** for making the `magformer/` repo production-grade and easy to operate without destabilizing ongoing experiments.

It delivers:

1. Workspace hygiene:
   - Move workspace-root `./scripts/` and `./output/` (MagFormer-related only) into `magformer/`.
   - After migration, the workspace root should not have ad-hoc `scripts/` or `output/` directories.
2. Output canonicalization:
   - Standardize `magformer/output/` into a readable, indexable structure.
   - Archive legacy / messy outputs under `magformer/output/_legacy/` (default: **do not delete**).
3. Baselines consolidation:
   - Move `icra_2026_experiment/` (3 additional baselines) under `magformer/baselines/` as external baseline packages.
   - Do not block the existing 5-model `0831_1k_5k` suite; extra baselines are organized first, integrated later.

## Hard Constraints (Safety)

1. **While the 5-model `output/experiments/0831_1k_5k/` suite is running:**
   - Do **not** move, rename, or delete any output directory (workspace-root `./output/`, `magformer/output/`, etc.).
   - Only code/doc changes are allowed.
2. Do not modify `mask2former/` implementation code. It is a reference runtime only.
3. Do not touch unrelated workspace projects/directories (e.g. `GaussianRender/`, `ecc-dataset/`, etc.).

## Target Layout (End State)

Workspace root (expected):

```text
electronic-components-grasp-and-segment/
  magformer/                 # primary, versioned, reproducible repo
  mask2former/               # reference implementation (no refactor here)
  magformer_datasets/        # datasets root
  scripts/                   # should NOT exist after migration
  output/                    # should NOT exist after migration
  icra_2026_experiment/      # should NOT exist after migration (moved into magformer/)
```

Inside `magformer/` (expected):

```text
magformer/
  baselines/
    detectron2/              # official repo (submodule)
    Mask2Former/             # official repo (submodule)
    ultralytics/             # official repo (submodule)
    ultralytics_tools/       # our data conversion / helper scripts
    icra_2026_experiment/    # external baseline packages (not guaranteed runnable)
      UnseenObjectClustering/
      UnseenObjectsWithMeanShift/
      uoais/
    VERSIONS.lock            # pinned commits for official baselines
    VERSIONS.local.lock      # provenance for icra_2026_experiment baselines
  output/
    experiments/             # canonical experiment outputs
    baselines/               # canonical baseline artifacts (e.g. yolo conversion cache)
    _legacy/                 # archived non-canonical outputs (default: do not delete)
    README.md                # output policy & conventions (tracked)
    INDEX.md                 # generated output index (tracked)
  scripts/
    experiments/             # runnable experiment suites
    analysis/                # reporting / indexing / param and speed tools
    visualization/           # triptych / overlay utilities
    dev/                     # lint/format scripts
  docs/
    plans/
      2026-02-18-workspace-refactor-plan-a-design.md
```

## Public Interface Changes

1. Workspace-root:
   - `./scripts/*` will migrate to `magformer/scripts/analysis/*`.
   - `./output/*` will migrate to `magformer/output/_legacy/root_output_<timestamp>/`.
2. `magformer/output/` will be standardized:
   - Top-level allowed: `experiments/`, `baselines/`, `_legacy/`, `README.md`, `INDEX.md`.
   - Anything else at `magformer/output/*` (top-level) gets archived into `_legacy/`.
3. `icra_2026_experiment/` will migrate to `magformer/baselines/icra_2026_experiment/`.

Compatibility policy:
 - No guarantee of preserving old workspace-root paths. The canonical entrypoint becomes `magformer/`.
 - During `0831_1k_5k`, keep all existing output paths unchanged.

## Milestones (Each Must git add/commit/push to origin/master)

### R0 — Wait for `0831_1k_5k` Suite Completion + Freeze Evidence

Completion conditions:
- `scripts/experiments/run_0831_1k_5k_all.sh` prints `done`
- `output/experiments/0831_1k_5k/summary_0831_1k_5k.json` exists

Deliverable:
- Fill `docs/experiments/baselines_0831_1k_5k_report.md` with real metrics/time/paths.

### R1 — Commit This Design Doc

Deliverable:
- `docs/plans/2026-02-18-workspace-refactor-plan-a-design.md` (this file)

### R2 — Migrate Workspace-root `scripts/` Into `magformer/scripts/analysis/`

Actions:
- Move:
  - `scripts/compare_model_params.py`
  - `scripts/compare_predictions.py`
  - `scripts/comparison_report.md`
  into `magformer/scripts/analysis/` (rename only if needed, keep intent clear).
- Record migration mapping in `docs/refactor/migration_log.md`.

### R3 — Archive Workspace-root `output/` Under `magformer/output/_legacy/`

Action:
- Move workspace-root `output/` into `magformer/output/_legacy/root_output_<timestamp>/`.

### R4 — Canonicalize `magformer/output/` Top-level (Archive Only)

Action:
- Move any `magformer/output/<non-canonical>` into `magformer/output/_legacy/pre_refactor_<timestamp>/`.
- Add a generator script `scripts/analysis/index_outputs.py` that writes:
  - `output/INDEX.md`
  - (optional) `output/index.json` for machines

NOTE: `output/` is typically git-ignored; track `output/README.md` and `output/INDEX.md` via explicit allowlist rules in `.gitignore`.

### R5 — Move `icra_2026_experiment/` Under `magformer/baselines/` (Non-blocking)

Actions:
- Move workspace-root `icra_2026_experiment/` to `magformer/baselines/icra_2026_experiment/`.
- Do not commit large artifacts (checkpoints/data/build/output). Add `.gitignore` inside the moved folder.
- Add `baselines/VERSIONS.local.lock` describing provenance, size, prerequisites, and why it is not part of the 0831_1K COCO AP suite yet.

### R6 — Minimal Production Tooling

Add:
- `pyproject.toml` (ruff/black/isort/pytest config)
- `.pre-commit-config.yaml`
- `scripts/dev/lint.sh` and `scripts/dev/format.sh`

### R7 — Code Hygiene Rules (Incremental Refactor Policy)

Add:
- `docs/refactor/CODE_HYGIENE_RULES.md`

Rules include:
- Never write outputs under package tree (e.g. `magformer/magformer/output`); outputs must go to `magformer/output/`.
- Any deletion/merge must include:
  - `rg` proof of references
  - replacement location
  - tests passing

## Verification

- Baseline suite runner still works:
  - `bash scripts/experiments/run_0831_1k_5k_all.sh --dry-run`
- Unit tests:
  - `pytest -q`
- `output/INDEX.md` can be regenerated by script (after R4).

