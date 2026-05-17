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

- tmux session: `r104_magformer_target50_balanced`
- Command shape: `/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/torchrun --nproc_per_node=4 tools/train.py --config configs/baseline_supervised_r104_magformer_r100_resume_target_labeled50_balanced_1000.yaml --gpus 4,5,6,7`
- Output dir: `output/baseline/r104_magformer_r100_resume_target_labeled50_balanced_1000`
- Resume evidence: all ranks loaded `output/baseline/r100_magformer_r98ckpt_target_labeled25_cleanfit_1024/checkpoint_iter_0000500.pth` and reported `Resumed from iteration 500`.
- Training window: started at `2026-05-18 05:01:35 +0800`, reached `training completed` at `2026-05-18 05:16:25 +0800`.
- Checkpoints used for external eval: `checkpoint_iter_0000599.pth`, `checkpoint_iter_0000799.pth`, and `checkpoint_iter_0001000.pth`.
- Final checkpoint: `output/baseline/r104_magformer_r100_resume_target_labeled50_balanced_1000/checkpoint_iter_0001000.pth`.

Built-in trainer eval is diagnostic only. The decision metrics below use the external 1024 backmap protocol.

## External 1024 Backmap Eval

Protocol for all rows: bbox+segm, model input `1024`, score threshold `0.05`, mask threshold `0.5`, inference topk `200`, COCO maxDets `200`, `MAGFORMER_MS_DEFORM_ATTN_BACKEND=pytorch`, `--force-pytorch-msda`, strict load `774/774` keys.

Val28 trajectory:

| checkpoint | predictions | bbox AP | bbox AP50 | bbox AP75 | segm AP | segm AP50 | segm AP75 | decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| iter0599 | 2473 | `0.330969` | `0.689374` | `0.274150` | `0.278326` | `0.599978` | `0.233699` | above R103 val28 |
| iter0799 | 2499 | `0.332828` | `0.688794` | `0.283529` | `0.282301` | `0.608745` | `0.230871` | above R103 val28 |
| iter1000 | 2442 | `0.334227` | `0.685862` | `0.285048` | `0.283886` | `0.609918` | `0.231551` | best val checkpoint |

Best-val checkpoint `iter1000` final eval:

| eval | images | GT anns | predictions | bbox AP | bbox AP50 | bbox AP75 | segm AP | segm AP50 | segm AP75 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train50 | 50 | 3170 | 3812 | `0.499279` | `0.828175` | `0.549972` | `0.466157` | `0.811844` | `0.495828` |
| val28 | 28 | 1892 | 2442 | `0.334227` | `0.685862` | `0.285048` | `0.283886` | `0.609918` | `0.231551` |
| non-overlap175 | 175 | 10277 | 13400 | `0.392745` | `0.738166` | `0.373761` | `0.329132` | `0.656310` | `0.299066` |
| target_unlabeled200 reference | 200 | 11750 | 15214 | `0.403330` | `0.750484` | `0.391485` | `0.340785` | `0.670643` | `0.316673` |

Artifacts:

- Val28 iter0599: `output/diagnostics/r104_magformer_target50_balanced_iter0599_val28_1024_backmap_topk200_20260518`
- Val28 iter0799: `output/diagnostics/r104_magformer_target50_balanced_iter0799_val28_1024_backmap_topk200_20260518`
- Val28 iter1000: `output/diagnostics/r104_magformer_target50_balanced_iter1000_val28_1024_backmap_topk200_20260518`
- Train50 iter1000: `output/diagnostics/r104_magformer_target50_balanced_iter1000_train50_1024_backmap_topk200_20260518`
- Non-overlap175 iter1000: `output/diagnostics/r104_magformer_target50_balanced_iter1000_nonoverlap175_1024_backmap_topk200_20260518`
- Target_unlabeled200 reference iter1000: `output/diagnostics/r104_magformer_target50_balanced_iter1000_target_unlabeled200_1024_backmap_topk200_20260518`

## Early Stop

Training was not stopped early. Iter0599 val28 segm AP was already `0.278326`, above the R103 val28 floor `0.272399`. Iter0799 also stayed above the floor at `0.282301`, and train50 segm AP75 at the final best-val checkpoint reached `0.495828`.

## Comparison

| run | checkpoint | train split | val28 segm AP | non-overlap target segm AP | target_unlabeled200 segm AP | target_unlabeled200 segm AP50 | target_unlabeled200 segm AP75 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| R103 cleanfit resume | iter0799 best-val | target_labeled25 | `0.272399` | n/a | `0.314747` | `0.636336` | `0.284706` |
| R80 historical | not rerun | mixed protocol | n/a | n/a | `0.336383` | n/a | n/a |
| R104 target-labeled50 balanced | iter1000 best-val | target_labeled50 balanced | `0.283886` | `0.329132` | `0.340785` | `0.670643` | `0.316673` |

Deltas vs R103:

- Val28 segm AP: `+0.011487`.
- Full target_unlabeled200 reference segm AP/AP50/AP75: `+0.026038 / +0.034307 / +0.031967`.
- Fair non-overlap175 segm AP/AP50/AP75: `0.329132 / 0.656310 / 0.299066`; R103 does not have this exact split in the recorded baseline.

R80 comparison:

- R104 full target_unlabeled200 reference segm AP `0.340785` is above historical R80 `0.336383`, but this R104 full200 number is leakage-prone because full200 includes the 25 promoted training images.
- R104 fair non-overlap175 segm AP `0.329132` is below the historical R80 full200 number `0.336383`; this is not an exact split match, so it should not be treated as a fair R80 win.

## Conclusion

R104 improves clearly over R103 on the fair validation signal. Val28 rises from `0.272399` to `0.283886`, and the non-overlap175 split reaches `0.329132` segm AP without containing the 25 promoted training images.

This supports the hypothesis that `target_labeled25` had a target-coverage gap. Adding the R37 balanced 25 images helps transfer, especially on AP75, but the fair non-overlap result is still not enough to claim a clean R80 beat.

The public full200 reference also improves to `0.340785`, but it must stay secondary because it includes the promoted training images.

## Commits

- Config/docs placeholder: `4a542d9c`
- Final docs-only update: this docs-only commit.
