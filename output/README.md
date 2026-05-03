# Output Layout (Canonical)

This repository writes all runtime artifacts under `output/`.

## Canonical Top-Level

Only these top-level entries are considered canonical:

- `output/experiments/`: experiment outputs (checkpoints, logs, metrics, visualizations)
- `output/pretrained/`: managed pretrained weights and shared checkpoints
- `output/_legacy/`: archived historical outputs (never deleted by default)
- `output/README.md`: this file
- `output/INDEX.md`: optional auto-generated index (if present)

Any other top-level directories under `output/` should be archived under `_legacy/`.

## Deletion Policy

By default, we do **not** delete any historical outputs. We only **move** them into
`output/_legacy/<tag>_<timestamp>/` for organization.

## Direct Cleanup Mode (Temporary Artifacts)

For large scratch runs, we allow explicit direct deletion of temporary artifacts when
requested by the experiment owner.

Use:

```bash
cd magformer
python scripts/analysis/cleanup_temp_artifacts.py --dry-run
python scripts/analysis/cleanup_temp_artifacts.py --write
```

This script only removes repo-local temporary paths (scratch outputs, tmp legacy dirs,
cache folders like `__pycache__` / `.pytest_cache`, and `baselines/*/runs`).
