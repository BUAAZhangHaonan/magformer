# ICRA 2026 Baselines (Vendored, Pruned)

This folder vendors three RGB-D instance segmentation baselines (research code) for
experiments on `magformer_datasets/0831_1K`.

Included (pruned) baselines:
- `unseen_object_clustering/` (UCN / Unseen Object Clustering)
- `msmformer/` (MSMFormer / Mean Shift Mask Transformer)
- `uoais/` (UOAIS / AdelaiDet-based)

Pruning rules:
- Keep model + training/eval code required for offline experiments.
- Drop ROS/robotics integration code.
- Drop datasets, checkpoints, and runtime outputs.

See `baselines/VERSIONS.local.lock` and `baselines/ENV.lock` for provenance and
environment notes.
