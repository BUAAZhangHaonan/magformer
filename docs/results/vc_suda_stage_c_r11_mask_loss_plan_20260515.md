# VC-SUDA Stage C R11 Mask-Loss Continuation - 2026-05-15

## Goal

Continue from R8B `ckpt999` with one supervised mask-quality variable. The target-domain goal remains segm AP `>= 0.61`; current best remains R8B target_unlabeled200 segm AP about `0.319922`, so this run is not a completion milestone.

## Checks Before Launch

- GPU 4-7 were idle except Xorg display memory.
- Existing Teacher first50 1024 backmap evidence is valid: bbox AP `0.6519639237`, segm AP `0.6253031403` in `output/experiments/teacher_first50_1024_backmap_recheck_20260515/metrics.cocoeval.json`.
- `pseudo_real_512` target_unlabeled200, target_labeled25, and val28 have no empty images, missing files, bad boxes, or zero-area masks. Contact sheets were written to `output/diagnostics/r11_pseudo_real_data_sanity_20260515/` and are not for commit.

## Hypothesis

The remaining Stage C errors are mostly high-score low-IoU masks and false negatives in dense images. Raising the supervised mask/dice loss and matching costs should bias the continuation toward better mask localization and instance separation without changing thresholding, learning rate, noise, sampling, or pseudo-label policy.

## Single Variable

R11 changes only the mask-quality supervised weight multiplier relative to R8B continuation:

- `model.magformer.mask_former.mask_weight`: `5.0 -> 7.5`
- `model.magformer.mask_former.dice_weight`: `5.0 -> 7.5`

Both weights use the same `1.5x` multiplier and are treated as one variable because they define the supervised mask-quality term and the matching cost for masks.

## Config

- Config: `configs/vc_suda_stage_c_r11_r8b_ckpt999_mask_loss75_continue_1024_teacher8499.yaml`
- Warm start: `output/vc_suda/stage_c_r8b_lsj10_low_lr_continue_1024_teacher8499/checkpoint_iter_0000999.pth`
- Output: `output/vc_suda/stage_c_r11_r8b_ckpt999_mask_loss75_continue_1024_teacher8499`
- Checkpoint retention: `runtime.checkpoint_max_keep: null`
- GPUs: `4,5,6,7`

## Evaluation Plan

Use the external 1024 backmap protocol for model selection. First inspect `ckpt249`; continue only if target_unlabeled200 segm AP is at least R8B `ckpt999` (`0.319162`) and preferably challenges global best R8B `ckpt749` (`0.319922`). Hard stop if `ckpt249` is below `0.3171` segm AP, if loss becomes non-finite, or if GPU/CPU memory exceeds 90%.
