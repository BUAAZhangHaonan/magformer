# Patches (Local)

This directory vendors UCN (Unseen Object Clustering Network) baseline code under
`baselines/icra_2026/unseen_object_clustering/`.

## 2026-02-19

### 1) Track `lib/` model code in git

The upstream baseline keeps most Python modules under `lib/` (e.g. `fcn/`, `networks/`, `utils/`).
We track this directory in git for reproducibility.

### 2) ECC runner wrapper + COCO export

- File: `baselines/run_ucn_0831_1k.py`
- Adds an ECC dataset adapter that:
  - reads RGB (BGR) images + `.npy` depth
  - converts COCO instance masks to UCN embedding labels (background=-1, instances 0..K-1)
  - runs mean-shift clustering on embeddings to produce instance masks
  - exports `coco_instances_results.json` and computes COCOeval metrics

