# Code Hygiene Rules (MagFormer)

This document defines non-negotiable rules for keeping the repo maintainable during ongoing refactors.

## 1) Outputs Must Not Live Inside the Package Tree

- Forbidden: writing runtime outputs under `magformer/magformer/output/` (or any path under the importable Python package).
- Required: all experiment/baseline artifacts go under `magformer/output/`.

Rationale:
- Prevents accidentally packaging large artifacts.
- Avoids mixing code and runtime state.
- Makes it possible to archive/clean outputs without touching source.

## 2) Canonical Output Structure

Only these directories are considered canonical under `magformer/output/`:

- `magformer/output/experiments/<exp_id>/<model_id>/...`
- `magformer/output/experiments/<experiment_id>/_shared/yolo_<dataset_id>/...`
- `magformer/output/_legacy/...` (archived, non-canonical)

Anything else under `magformer/output/*` is legacy and must be moved into `_legacy/`.

## 3) Vendored / Third-party Code Rules

- All third-party repos live under `magformer/baselines/`.
- Do not modify vendored baseline source code unless it is required to run the baseline on our dataset.
  - If we must patch: keep a minimal patch set and document it.

## 4) Scripts Must Be Versioned and Named Intentionally

- All runnable scripts must live under `magformer/scripts/`.
- Naming conventions:
  - Experiment runners: `scripts/experiments/run_<dataset>_<budget>_<model>.sh`
  - Analysis utilities: `scripts/analysis/<verb>_<noun>.py`
  - Visualization: `scripts/visualization/<verb>_<noun>.py`

No ad-hoc scripts should remain in the workspace root.

## 5) Deleting / Merging Code Requires Proof

Any deletion or consolidation must include:

1. Proof it is unused:
   - `rg` results showing no references, or the exact replacement call sites.
2. Replacement location:
   - Where the functionality moved to.
3. Verification:
   - `pytest -q` passes.
   - Any relevant dry-run command(s) run successfully.

## 6) Git Discipline

- Each milestone must be committed and pushed to `origin/master` immediately.
- Keep commits small and single-purpose.
- Commit messages must be meaningful (e.g. `chore(output): ...`, `refactor(scripts): ...`).

## 7) Safety During Long Runs

While a long-running experiment suite is executing:

- Do not move/rename/delete any output directory.
- Avoid editing model/training code that the running process imports.
- Prefer working in a git worktree.
