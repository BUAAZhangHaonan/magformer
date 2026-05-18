# R115 MagFormer R114-Warm Full Target200 Oracle

Date: 2026-05-18
Branch: feature/vc-suda-sim2real
Config: `configs/baseline_supervised_r115_magformer_r114warm_fulltarget200_oracle_2000.yaml`
Setup commit: pending

## Purpose

R115 is an oracle upper-bound diagnosis. It is not a formal UDA result and it is not a held-out target result.

This run uses all target200 ground-truth annotations as training labels. It tests whether strong model-only initialization from R114 iter2000 plus full target GT can push the train200 oracle AP to `0.61+`.

No pseudo labels, Stage C, source replay, or new modules are introduced.

## Recipe

- Train annotation: `annotations/instances_target_unlabeled.json`.
- Val annotation: `annotations/instances_val.json`.
- Warm-start: `output/baseline/r114_magformer_r113warm_target150_balanced_2000/checkpoint_iter_0002000.pth` through `model.finetune_weights`.
- `runtime.resume: null`, so optimizer, scheduler, and AMP scaler start fresh.
- `solver.max_iter: 2000`.
- R114/R113 recipe held fixed: `base_lr=5e-5`, cosine schedule, batch size `4`, image size `1024`, no RGB augmentation, no depth noise, `importance_sample_ratio=0.0`, `dice_weight=10.0`, `mask_weight=5.0`, and `runtime.depth_sanity.min_mask_fg_ratio=0.0009`.

## Success And Stop Criteria

Primary success line:

- Train200 segm AP `>= 0.61`.

Guardrail:

- Val28 segm AP `>= 0.305`.

Early stop gate at iter999:

- If train200 segm AP `< 0.56` and val28 segm AP `<= 0.318`, stop and record the run as not worth continuing.
- Otherwise continue to iter2000.

Stage-level failure signal:

- If final train200 segm AP `< 0.58`, this suggests the current stage should be closed down.

## Static Validation

Static validation passed before training.

- `validate_config(strict=True)`: valid. The only message was the existing DPE informational note.
- Warm-start checkpoint exists: `output/baseline/r114_magformer_r113warm_target150_balanced_2000/checkpoint_iter_0002000.pth`.
- Config checks: `model.finetune_weights` points to R114 iter2000, `runtime.resume=null`, `solver.max_iter=2000`, train annotation points to full target200 oracle, no RGB augmentation, no depth noise, `importance_sample_ratio=0.0`, `dice_weight=10.0`, `mask_weight=5.0`, and `runtime.depth_sanity.min_mask_fg_ratio=0.0009`.
- Annotation counts: train200 `200` images / `11750` anns, val28 `28` images / `1892` anns.
- Empty-annotation image ratio: train200 `0.000000`, val28 `0.000000`.
- Loader smoke checked: raw dataset lengths train/val `200 / 28`; DataLoader dataset lengths train/val `200 / 25` because `runtime.eval_max_images=25`; first train batch images `[4, 3, 1024, 1024]`, depths `[4, 1, 1024, 1024]`, mask counts `[49, 50, 98, 50]`; first val batch images `[4, 3, 512, 512]`, depths `[4, 1, 512, 512]`.
- Warm-start load checked on CPU: matched `774/774`, missing keys `0`, unexpected keys `0`.

## Training

Pending.

## Results

Pending.

## Conclusion

Pending.
