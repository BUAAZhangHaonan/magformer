# Checkpoint Cleanup Plan - 2026-05-19

This plan converts checkpoint cleanup into an auditable dry run. No files were deleted.

## Current Dry-Run Estimate

- Manifest: `docs/results/checkpoint_cleanup_manifest_20260519.md`
- Tool: `tools/plan_checkpoint_cleanup.py`
- Checkpoints scanned: 502
- Delete candidates: 425
- Estimated reclaim if every listed candidate is manually approved: 233.9 GiB
- First recommended batch: R139/R141 intermediate checkpoints only
- First batch estimated reclaim: 109.5 GiB

## Required Manual Gate

Deletion must not run from this commit. The manifest is only a review list. A human must inspect the candidate paths, confirm that no candidate is needed for reproduction, and approve a separate deletion step before any file is removed.

## First Batch Recommendation

Start with only the R139/R141 intermediate checkpoints listed as candidates in the manifest. Do not remove these gate checkpoints:

- `output/baseline/r139_magformer_rgb_only_r98warm_target150_20260519/checkpoint_iter_0002000.pth`
- `output/baseline/r141_magformer_rgbd_r98warm_target150_20260519/checkpoint_iter_0002000.pth`

Keep run directories and all logs, metrics, configs, JSON files, and other non-checkpoint artifacts. The cleanup unit is an individual checkpoint file, not a directory.

## Dry-Run Command

```bash
/home/hdd3/zhanghaonan/anaconda3/bin/python tools/plan_checkpoint_cleanup.py \
  --repo-root . \
  --manifest docs/results/checkpoint_cleanup_manifest_20260519.md
```

The planner has no delete mode.
