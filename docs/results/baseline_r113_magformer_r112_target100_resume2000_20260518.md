# R113 MagFormer R112 Target100 Resume2000

Date: 2026-05-18
Branch: feature/vc-suda-sim2real
Config: `configs/baseline_supervised_r113_magformer_r112_target100_resume2000.yaml`

## Purpose

R113 only tests whether R112 target100 was under-trained.

The data split and recipe stay fixed from R112. The run true-resumes from the R112 `checkpoint_iter_0001000.pth` training state and extends `solver.max_iter` to `2000`.

No target150 expansion, pseudo labels, Stage C, source replay, or new modules are introduced.

## Unique Variable

- R112: target100, max_iter `1000`.
- R113: same target100 data and recipe, true resume from R112 iter1000, max_iter `2000`.

The warm-start path is disabled with `model.finetune_weights: null` so runtime resume is the only initialization path.

## Under-Training Hypothesis

R112 reached only about 40 equivalent epochs on target100. R111 target50 reached about 80 equivalent epochs and had higher train-fit.

If R113 raises train100 fit while val28 and remaining125 stay stable, R112 was likely under-trained rather than limited only by the target100 data.

## Midpoint Rule

Evaluate `checkpoint_iter_0001500.pth` on:

- train100
- val28
- remaining125

Stop and record if any midpoint guard fails:

- val28 segm AP `< 0.300`
- remaining125 segm AP `< 0.356`
- train100 segm AP `< 0.548`

Otherwise continue to iter2000.

## Success Criteria

Primary criteria:

- train100 segm AP `>= 0.61`
- val28 segm AP `>= 0.305`
- remaining125 segm AP `>= 0.360`

Preferred target-retention criterion:

- remaining125 should not fall below R112 `0.365856`.

## Static Validation

Static validation passed before training.

- `validate_config(strict=True)`: valid. The only message was the existing DPE informational note.
- True resume checkpoint exists: `output/baseline/r112_magformer_r111recipe_target_labeled100_balanced_1000/checkpoint_iter_0001000.pth`.
- Resume checkpoint training state: `iter=1000`, with optimizer, LR scheduler, and scaler state present.
- `model.finetune_weights: null`, so warm-start is disabled while runtime resume is active.
- `solver.max_iter: 2000`, `runtime.eval_period: 500`, `runtime.checkpoint_period: 100`.
- Annotation counts: train100 `100` images / `6125` anns, val28 `28` images / `1892` anns, remaining125 `125` images / `7322` anns, full200 reference `200` images / `11750` anns.
- Empty-annotation image ratio: train100 `0.000000`, val28 `0.000000`, remaining125 `0.000000`, full200 `0.000000`.
- File-name overlap: train100 vs remaining125 `0`; train100 vs val28 `0`; remaining125 vs val28 `0`.
- Loader smoke checked: raw dataset lengths train/val `100 / 28`; DataLoader dataset lengths train/val `100 / 25` because `runtime.eval_max_images=25`; first train batch images `[1, 3, 1024, 1024]`, depths `[1, 1, 1024, 1024]`; first val batch images `[4, 3, 512, 512]`, depths `[4, 1, 512, 512]`.

## Training

- tmux session: `r113_magformer_target100_resume2000`.
- Command shape: `/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 tools/train.py --config configs/baseline_supervised_r113_magformer_r112_target100_resume2000.yaml --gpus 4,5,6,7 --num-workers 2`.
- Output dir: `output/baseline/r113_magformer_r112_target100_resume2000`.
- Log: `output/baseline/r113_magformer_r112_target100_resume2000.tmux.log`.
- Training window: started `2026-05-18T11:07:21+08:00`, reached `training completed` at `2026-05-18 11:30:14 +0800`, and exited `TRAIN_EXIT:0`.
- True-resume evidence: all four ranks loaded `output/baseline/r112_magformer_r111recipe_target_labeled100_balanced_1000/checkpoint_iter_0001000.pth` and printed `Resumed from iteration 1000`.
- Checkpoints used for external eval: `checkpoint_iter_0001499.pth` as the 1500-near midpoint and `checkpoint_iter_0002000.pth` as final.
- No OOM, traceback, or runtime error was found in the training log. The PyTorch NCCL destroy-process-group warning appeared after normal exit.

## Midpoint Eval

External eval protocol for pseudo-real target rows: bbox+segm, model input `1024`, score threshold `0.05`, mask threshold `0.5`, inference topk `200`, COCO maxDets `200`, `MAGFORMER_MS_DEFORM_ATTN_BACKEND=pytorch`, `--force-pytorch-msda`, strict load `774/774` keys.

| checkpoint | eval | predictions | bbox AP | bbox AP50 | bbox AP75 | segm AP | segm AP50 | segm AP75 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| iter1499 | train100 | 9315 | `0.538661` | `0.848551` | `0.622552` | `0.555200` | `0.873894` | `0.643555` |
| iter1499 | val28 | 3464 | `0.335943` | `0.691303` | `0.283732` | `0.304633` | `0.619606` | `0.267222` |
| iter1499 | remaining125 | 13332 | `0.405717` | `0.747244` | `0.399582` | `0.366712` | `0.680483` | `0.358365` |

The midpoint stop rule did not trigger:

- train100 `0.555200` is not below `0.548`.
- val28 `0.304633` is not below `0.300`.
- remaining125 `0.366712` is not below `0.356`.

## Final Eval

Best checkpoint by val28 among the externally evaluated R113 checkpoints is `checkpoint_iter_0002000.pth`.

| checkpoint | eval | predictions | bbox AP | bbox AP50 | bbox AP75 | segm AP | segm AP50 | segm AP75 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| iter2000 | train100 | 9356 | `0.546699` | `0.855137` | `0.633399` | `0.561041` | `0.875704` | `0.653386` |
| iter2000 | val28 | 3488 | `0.340289` | `0.699305` | `0.290685` | `0.305354` | `0.614782` | `0.271655` |
| iter2000 | remaining125 | 13442 | `0.410244` | `0.748608` | `0.407246` | `0.367070` | `0.679450` | `0.361141` |
| iter2000 | full200 reference | 20316 | `0.462855` | `0.788879` | `0.495006` | `0.437321` | `0.751328` | `0.468251` |
| iter2000 | original first50 source sanity | 4313 | `0.397530` | `0.715608` | `0.415575` | `0.477363` | `0.766666` | `0.522734` |

Original first50 source sanity used `tools/evaluate_teacher_first50_1024_backmap.py` with `configs/finetune_1k_full_1024.yaml`, dataset root `magformer_datasets/20260318_1K_1566`, annotation `annotations/instances_all.json`, split `all`, max images `50`, inference topk `100`, and maxDets `100`.

Artifacts:

- Train log: `output/baseline/r113_magformer_r112_target100_resume2000.tmux.log`.
- Midpoint train100: `output/diagnostics/r113_magformer_r112_target100_resume2000_iter1499_train100_1024_backmap_topk200_20260518`.
- Midpoint val28: `output/diagnostics/r113_magformer_r112_target100_resume2000_iter1499_val28_1024_backmap_topk200_20260518`.
- Midpoint remaining125: `output/diagnostics/r113_magformer_r112_target100_resume2000_iter1499_remaining125_1024_backmap_topk200_20260518`.
- Final train100: `output/diagnostics/r113_magformer_r112_target100_resume2000_iter2000_train100_1024_backmap_topk200_20260518`.
- Final val28: `output/diagnostics/r113_magformer_r112_target100_resume2000_iter2000_val28_1024_backmap_topk200_20260518`.
- Final remaining125: `output/diagnostics/r113_magformer_r112_target100_resume2000_iter2000_remaining125_1024_backmap_topk200_20260518`.
- Full200 reference: `output/diagnostics/r113_magformer_r112_target100_resume2000_iter2000_full200_reference_1024_backmap_topk200_20260518`.
- Original first50 source sanity: `output/diagnostics/r113_magformer_r112_target100_resume2000_iter2000_original_first50_1024_backmap_20260518`.

## Judgment

R113 improves train fit over R112 and preserves the held-out target checks, but it does not reach the requested train100 fit target.

| check | R112 iter1000 | R113 best | delta | pass |
| --- | ---: | ---: | ---: | --- |
| train100 segm AP `>=0.61` | `0.528334` | `0.561041` | `+0.032707` | no |
| val28 segm AP `>=0.305` | `0.308258` | `0.305354` | `-0.002904` | yes |
| remaining125 segm AP `>=0.360` | `0.365856` | `0.367070` | `+0.001214` | yes |
| preferred remaining125 `>=0.365856` | `0.365856` | `0.367070` | `+0.001214` | yes |

Conclusion:

- Success status: not achieved, because train100 stayed below `0.61`.
- Held-out retention: achieved. Val28 stayed above `0.305`; remaining125 stayed above both `0.360` and R112 `0.365856`.
- Under-training hypothesis: partially supported. Extending from 1000 to 2000 improved train100 from `0.528334` to `0.561041` and did not hurt remaining125, but the gain was not large enough to explain the full gap to the `0.61` target.

## Commits

- Config/docs placeholder: `442d43e9`.
- Final docs-only update: this docs-only commit.
