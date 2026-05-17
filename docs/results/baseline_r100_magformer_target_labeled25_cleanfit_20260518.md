# R100 MagFormer target_labeled25 clean train-fit debug - 2026-05-18

Conclusion: pending. R100 tests whether the R98 warm-start MagFormer can cleanly fit the 25 labeled target images when augmentation, depth noise, VC-SUDA, EMA, offline pseudo labels, contrastive loss, and unsupervised paths are disabled.

## Goal

R99 failed to fit even `target_labeled25`: train25 external 1024 backmap segm AP75 was only `0.017616`. R100 is a clean train-fit check, not a generalization-first run.

## Config Gate

- Config: `configs/baseline_supervised_r100_magformer_r98ckpt_target_labeled25_cleanfit_1024.yaml`
- Init: `output/baseline/r98_magformer_32k_source_short/checkpoint_iter_0000499.pth`
- Data root: `magformer_datasets/pseudo_real_512`
- Train ann: `annotations/instances_target_labeled.json`
- Val/eval ann: `annotations/instances_target_labeled.json`
- Image size: `1024`
- Clean aug: `random_flip=none`, RGB photo aug zeroed, depth noise zeroed
- Solver: `base_lr=5e-5`, `weight_decay=0.01`, `warmup_iters=0`, `warmup_factor=1.0`, `max_iter=500`
- Loss weights: unchanged from R99
- Checkpoint period: `500`

## Loader Sanity

- Images: `25`
- Annotations: `1697`
- Empty segmentations: `0`
- Nonpositive area annotations: `0`
- Raw RGB shape: `[512, 512, 3]`
- Raw depth shape: `[512, 512]`
- Clean transformed RGB shape: `[3, 1024, 1024]`
- Clean transformed depth shape: `[1, 1024, 1024]`
- Mask count min/mean/max: `50 / 67.88 / 100`
- First transformed mask tensor: `[100, 1024, 1024]`, non-empty `true`

## Decision Gate

Use external 1024 backmap bbox+segm evaluation on train25 with inference topk `200` and COCO maxDets `200`.

- If train25 segm AP75 is `>= 0.10`, record that fit starts to recover, then evaluate val28 and target_unlabeled200.
- If train25 segm AP75 is close to historical clean overfit `0.56`, record R100 as a clear clean-fit recovery.
- If train25 segm AP75 is `< 0.10`, stop and compare init, data, and eval protocol before changing mask/dice weights.

## Run Log

Pending.
