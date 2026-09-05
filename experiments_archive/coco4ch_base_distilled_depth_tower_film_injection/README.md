# Cross-distillation audit: coco4ch base + distilled depth tower + FiLM injection (old code name: "CDTI")

Four arms, subset-1000 @40K iters (2026-08-28..09-02):
  W1 cdti_full      -> full 4-stage tower injection ....... segm AP 0.8441 (winner)
  W3 cdti_nodistill -> same arch, undistilled tower ...... 0.8367  (distillation worth +0.74)
  W4 cdti_shallow   -> shallow injection ................. 0.8344
  W2 concat_ctrl    -> naive concat control .............. 0.8296
zero-shot of the distilled init (no training): 0.8491.

Source = 4029:~/magformer_audit/source/cdti (full tree; vendored baselines/
omitted here, they are the repo's own baselines/). Init builder:
scripts/build_cdti_init.py (coco4ch_magformer_init + depth_tower_distilled,
FiLM keys deliberately zero-init). Run configs in ../audit_2026-08_configs/.
Weights+metrics: archive_20260906/ (both machines' staging dirs).
