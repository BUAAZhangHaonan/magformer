# R106 MagFormer Full-Target Oracle Warm-Start

## Purpose

R106 tests one variable against R105: initialization changes from true resume to model-only warm-start.

## Fixed Settings

- Base config: `configs/baseline_supervised_r105_magformer_fulltarget200_oracle_1000.yaml`
- Train split: target `instances_target_unlabeled.json` full train200 oracle labels
- Val split: `instances_val.json` val28
- Image size: 1024
- Optimizer: AdamW, `base_lr=5e-5`, `weight_decay=0.01`
- Schedule: cosine, `warmup_iters=0`, `max_iter=1000`
- Aug/noise: disabled
- VC-SUDA, EMA, pseudo bank, source replay, contrastive, and postprocess: disabled

## Unique Variable

- R105 used `runtime.resume` from R100 iter500, inheriting optimizer, scheduler, scaler, and iteration state.
- R106 sets `runtime.resume: null` and uses `model.finetune_weights` from R100 iter500, so model weights are loaded but optimizer, scheduler, scaler, and iteration state restart from iter0.

## Gates

- If iter499 train200 segm AP is `>= 0.42`, inherited optimizer/scheduler/scaler state is the main R105 failure cause.
- If iter799 or iter1000 train200 segm AP is `>= 0.42`, under-training or reset training is effective.
- If train200 segm AP stays around `0.36-0.39`, the next cause is likely scale, model capacity, or point sampling.
- Separately record whether train200 reaches 61+ AP. This is expected to be unlikely and should be reported directly.

## Results

Pending.
