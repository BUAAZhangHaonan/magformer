# R115 MagFormer R114-Warm Full Target200 Oracle

Date: 2026-05-18
Branch: feature/vc-suda-sim2real
Config: `configs/baseline_supervised_r115_magformer_r114warm_fulltarget200_oracle_2000.yaml`
Setup commit: `4e697fa2`

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

Training was launched in tmux session `r115_magformer_r114warm_fulltarget200_oracle` on GPUs `4,5,6,7`.

Command:

```bash
/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 tools/train.py --config configs/baseline_supervised_r115_magformer_r114warm_fulltarget200_oracle_2000.yaml --gpus 4,5,6,7 --num-workers 2
```

Evidence:

- Log: `output/baseline/r115_magformer_r114warm_fulltarget200_oracle_2000.tmux.log`.
- Start: `2026-05-18T13:08:40+08:00`.
- Exit: `2026-05-18T13:08:56+08:00`, status `1`.
- Warm-start loaded R114 iter2000 through `model.finetune_weights` on all ranks with missing keys `0` and unexpected keys `0`.
- Depth sanity report: `output/baseline/r115_magformer_r114warm_fulltarget200_oracle_2000/depth_sanity.json`.
- GPUs 4-7 returned to idle after the abort. No CUDA OOM occurred.

## Blocker

Training aborted before iter `0` at the configured depth sanity preflight.

```text
[Train] Depth sanity preflight failed:
[Train]   - predicted masks are effectively empty: fg_ratio=0.000804
RuntimeError: Depth sanity preflight failed; aborting before full training.
```

The requested R115 recipe sets `runtime.depth_sanity.min_mask_fg_ratio=0.0009`. The observed warm-start preflight foreground ratio was `0.000804`, so the run could not legally continue under the requested config.

For comparison, R114 target150 used the same threshold and passed at `0.001035`; R110 fulltarget200 dice125 passed at `0.000984`. R115 with the R114 warm-start is below the configured gate on the first DDP preflight batch.

## Results

No iter999 or iter2000 checkpoint was produced, so no train200, val28, or original first50 eval was run.

| checkpoint | split | bbox AP | bbox AP50 | bbox AP75 | segm AP | segm AP50 | segm AP75 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| iter0999 | train200 | n/a | n/a | n/a | n/a | n/a | n/a |
| iter0999 | val28 | n/a | n/a | n/a | n/a | n/a | n/a |
| iter2000 | train200 | n/a | n/a | n/a | n/a | n/a | n/a |
| iter2000 | val28 | n/a | n/a | n/a | n/a | n/a | n/a |
| iter2000 | original first50 source sanity | n/a | n/a | n/a | n/a | n/a | n/a |

## Success Check

| condition | result |
| --- | --- |
| train200 segm AP `>= 0.61` | not measured; training blocked before iter0 |
| val28 segm AP `>= 0.305` | not measured; training blocked before iter0 |
| iter999 early stop gate | not reached |
| final train200 `< 0.58` stage-close signal | not measured |

## Conclusion

R115 did not produce oracle AP results because the requested depth sanity gate stopped training before the first iteration.

This is a configuration-faithful blocker, not an OOM or process failure. Continuing would require an explicit recipe change, such as lowering `runtime.depth_sanity.min_mask_fg_ratio` below `0.000804` or setting `runtime.skip_depth_sanity=true`; either change would no longer be the requested R115 recipe.
