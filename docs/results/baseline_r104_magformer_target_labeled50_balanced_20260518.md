# R104 MagFormer Target Labeled50 Balanced

Date: 2026-05-18
Branch: feature/vc-suda-sim2real
Config: `configs/baseline_supervised_r104_magformer_r100_resume_target_labeled50_balanced_1000.yaml`

## Goal

Run the R103 recipe with the R37 balanced target-labeled50 annotation and measure whether the extra 25 supervised target images improve fair target transfer.

## Assumptions

- R104 is R103 cleanfit resume, not a VC-SUDA run.
- Resume checkpoint is still `output/baseline/r100_magformer_r98ckpt_target_labeled25_cleanfit_1024/checkpoint_iter_0000500.pth`.
- The R37 balanced plus25 annotation contains the original 25 labeled target images plus 25 promoted target-unlabeled images by file name.
- The fair target split is `annotations/instances_target_unlabeled_r37_balanced_minus25.json`, which is full target_unlabeled200 minus the 25 promoted training images.

## Only Changed Variable

- `data.train_ann`: `annotations/instances_target_labeled.json` -> `annotations/instances_target_labeled_r37_balanced_plus25.json`.

Everything else should match R103: cleanfit settings, solver, true resume, image size 1024, disabled augmentation/noise paths, disabled VC-SUDA, disabled EMA, disabled pseudo labels, disabled source replay, disabled contrastive, and no postprocess module changes.

## Main Metrics

Primary fair metrics:

- val28 segm AP, because it has no overlap with train50 by file name.
- non-overlap175 segm AP/AP50/AP75 on `instances_target_unlabeled_r37_balanced_minus25.json`.

Reference-only metric:

- full target_unlabeled200 segm AP/AP50/AP75, because it includes the 25 promoted training images and is leakage-prone for R104.

## Early Stop Rule

Evaluate val28 first at iter0599 and iter0799. If both val28 segm AP values are below R103 val28 segm AP `0.272399` and train50 segm AP75 has not reached `0.45`, stop the run and record the result.

If the gate passes, select the best val28 checkpoint among iter0599, iter0799, and iter1000, then evaluate train50, val28, non-overlap175, and full target_unlabeled200 reference.

## Static Validation

Static validation passed before training.

- `validate_config(strict=True)`: valid. The only message was the existing DPE informational note.
- R104 uses true resume: `runtime.resume=output/baseline/r100_magformer_r98ckpt_target_labeled25_cleanfit_1024/checkpoint_iter_0000500.pth`.
- Resume checkpoint has `iter=500` and `774` model state keys.
- Config fields checked: `max_iter=1000`, `checkpoint_period=100`, `checkpoint_max_keep=0`, `image_size=1024`, `random_flip=none`, `base_lr=5e-5`, `warmup_iters=0`, and `weight_decay=0.01`.
- Annotation access checked: train50 `50` images / `3170` anns, val28 `28` images / `1892` anns, non-overlap175 `175` images / `10277` anns, full target_unlabeled200 `200` images / `11750` anns.
- Empty-image check passed for train50, val28, non-overlap175, and target_unlabeled200.
- By file name, train50 contains the original 25 labeled images plus 25 promoted images from target_unlabeled200.
- Existing non-overlap split `annotations/instances_target_unlabeled_r37_balanced_minus25.json` is exactly target_unlabeled200 minus the 25 promoted training images by file name.
- Non-overlap175 has `0` file-name overlap with train50; val28 has `0` file-name overlap with train50.
- Loader smoke checked: train/val dataset lengths `50 / 25`; first per-rank train batch images `[1, 3, 1024, 1024]`, depths `[1, 1, 1024, 1024]`, target mask counts `[100]`.


## Training

Pending.

## External 1024 Backmap Eval

Pending.

## Comparison

Pending.

## Conclusion

Pending.

## Commits

Pending.
