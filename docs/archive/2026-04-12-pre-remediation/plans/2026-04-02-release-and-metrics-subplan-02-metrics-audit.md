# Metrics Audit Subplan

1. Collect the final machine-readable artifacts for the `1024`, `512`, and `256` suites.
2. For each published run family, inspect the runner, manifest, and saved summary data to see whether the results came from the repaired single-GPU evaluation path, the old broken DDP validation path, or the offline evaluation CLI.
3. Reuse existing metrics if the artifacts already come from valid final evaluation outputs and the repair changes do not alter their semantics.
4. Mark reevaluation as required only if a published result depended on the old broken DDP best-checkpoint selection or on an unpinned offline evaluation weights source.
5. Mark retraining as required only if a run’s published best checkpoint itself is no longer trustworthy because the old evaluation path chose the wrong checkpoint.
