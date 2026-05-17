# R105 MagFormer Full-Target200 Oracle

Date: 2026-05-18
Branch: feature/vc-suda-sim2real
Config: `configs/baseline_supervised_r105_magformer_fulltarget200_oracle_1000.yaml`

## Status

Placeholder before training.

R105 is an oracle upper-bound diagnosis. It directly uses the hidden GT in `target_unlabeled200` for supervised training. Do not report it as a formal UDA result, and do not compare it as a fair paper metric.

## Goal

Answer only two questions:

- Can MagFormer/RGB-D reach `61+` train200 segm AP when supervised by the hidden `target_unlabeled200` GT?
- Does held-out `val28` improve at the same time?

## Protocol

- Train annotation: `annotations/instances_target_unlabeled.json`
- Val annotation: `annotations/instances_val.json`
- Resume checkpoint: `output/baseline/r100_magformer_r98ckpt_target_labeled25_cleanfit_1024/checkpoint_iter_0000500.pth`
- True resume: keep optimizer, scheduler, and iteration state from the R100 checkpoint.
- Image size: `1024`
- Solver: `base_lr=5e-5`, `warmup_iters=0`, `weight_decay=0.01`, `max_iter=1000`
- Aug/noise: no flip, no RGB photo augmentation, no depth noise.
- Disabled paths: VC-SUDA, EMA, pseudo labels, offline pseudo labels, source replay, contrastive loss, and postprocess changes.
- Eval protocol: external `1024` backmap, bbox+segm, topk `200`, COCO maxDets `200`.

## Judgment Gates

- If train200 segm AP reaches `0.61+`, label coverage is not the main blocker for fitting the full target set under this recipe.
- If train200 segm AP stays clearly below `0.50` by iter0799 and the curve is flat, stop and record an optimization/scale bottleneck.
- If train200 reaches `0.61+` but val28 does not improve over R104 val28 `0.283886`, treat the result as overfit to target_unlabeled200 rather than general target improvement.
- If train200 improves strongly and val28 also improves over R104, label coverage is likely a real UDA bottleneck.

## Static Validation

Pending.

## Training

Pending.

## External Eval

Pending.

## Conclusion

Pending.

## Commits

- Config/docs placeholder: pending.
- Final docs-only update: pending.
