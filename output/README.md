# Output Layout (Canonical)

This repository writes all runtime artifacts under `output/`.

## Canonical Top-Level

Only these top-level entries are considered canonical:

- `output/experiments/`: experiment outputs (checkpoints, logs, metrics, visualizations)
- `output/baselines/`: baseline data conversion caches and intermediate artifacts
- `output/_legacy/`: archived historical outputs (never deleted by default)
- `output/README.md`: this file
- `output/INDEX.md`: optional auto-generated index (if present)

Any other top-level directories under `output/` should be archived under `_legacy/`.

## Deletion Policy

By default, we do **not** delete any historical outputs. We only **move** them into
`output/_legacy/<tag>_<timestamp>/` for organization.

