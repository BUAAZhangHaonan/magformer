# E18 historical output cleanup

This directory records the completed cleanup of historical MagFormer outputs.
It contains evidence only; no executable deletion code is retained.

## Released space

- Batch 1 removed obsolete datasets, caches, smoke outputs, empty RDI runs, and large reproducible predictions: about 103.73 GiB.
- Batch 2 removed the completed shared evaluation directory after recording its available metrics: about 1.216 GiB.
- Batch 3 removed 222 historical experiment directories after recording all available metrics and explicit metric gaps: about 478.75 GiB.

## Result boundary

`metrics_summary.json` contains 223 cleanup records and the protected-path whitelist.

The `preflight_1img` AP 92.0646 entry is a single-image preflight observation. It is not a formal full-validation result and must not be reported as one.
