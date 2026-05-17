# R100 MagFormer target_labeled25 clean train-fit debug - 2026-05-18

Conclusion: R100 passes the clean train-fit gate. External 1024 backmap train25 segm AP75 is `0.523051`, far above the `0.10` recovery gate and close to the historical clean overfit AP75 line around `0.56`. The R99 failure was not an init/data/eval-protocol hard failure; the clean optimizer/augmentation setup restores fit.

## Goal

R99 failed to fit even `target_labeled25`: train25 external 1024 backmap segm AP75 was only `0.017616`. R100 tested whether the same R98 warm-start MagFormer can fit the 25 labeled target images when augmentation, depth noise, VC-SUDA, EMA, offline pseudo labels, contrastive loss, and unsupervised paths are disabled.

## Config

- Config: `configs/baseline_supervised_r100_magformer_r98ckpt_target_labeled25_cleanfit_1024.yaml`
- Base: R99 config, with loss weights unchanged.
- Init: `output/baseline/r98_magformer_32k_source_short/checkpoint_iter_0000499.pth`
- Data root: `magformer_datasets/pseudo_real_512`
- Train ann: `annotations/instances_target_labeled.json`
- Val/eval ann: `annotations/instances_target_labeled.json`
- Image size: `1024`
- Clean aug: `random_flip=none`, RGB photo aug zeroed, depth noise zeroed
- Solver: `base_lr=5e-5`, `weight_decay=0.01`, `warmup_iters=0`, `warmup_factor=1.0`, `max_iter=500`
- Checkpoint period: `500`
- Runtime: VC-SUDA, EMA, offline pseudo, contrastive, source retention, and unsupervised paths disabled.

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

## Validation

Static config and loader smoke passed.

- `validate_config(strict=True)`: valid.
- Train/val dataset lengths: `25 / 25`.
- Train batch: images `[4, 3, 1024, 1024]`, depths `[4, 1, 1024, 1024]`.
- Train batch target mask counts: `[50, 50, 100, 100]`, all non-empty.
- Val loader remains raw `[4, 3, 512, 512]`; external eval performs the required 1024 resize/backmap.
- Warm-start load during training: `missing=0`, `unexpected=0` on all ranks.
- Depth sanity: `should_abort=false`, depth min/max/mean/std `0.0 / 1.0 / 0.948927 / 0.136127`.

## Training

- tmux session: `r100_magformer_target_labeled25_cleanfit`
- Command shape: `torchrun --nproc_per_node=4 tools/train.py --config configs/baseline_supervised_r100_magformer_r98ckpt_target_labeled25_cleanfit_1024.yaml`
- Output dir: `output/baseline/r100_magformer_r98ckpt_target_labeled25_cleanfit_1024`
- Training completed at `2026-05-18 03:26:28 +0800`.
- Final checkpoints: `checkpoint_iter_0000499.pth` and `checkpoint_iter_0000500.pth`.
- External eval uses `checkpoint_iter_0000500.pth`.

Built-in trainer eval remains diagnostic only here. It reported train-set segm AP75 around `0.0155` because that path is not the external 1024 backmap gate. The gate below is the decision metric.

## External 1024 Backmap Eval

Protocol for all rows: bbox+segm, model input `1024`, score threshold `0.05`, mask threshold `0.5`, inference topk `200`, COCO maxDets `200`, `--force-pytorch-msda`, strict checkpoint load `774/774` keys.

| eval | images | GT anns | predictions | bbox AP | bbox AP50 | bbox AP75 | segm AP | segm AP50 | segm AP75 | topk truncated |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train25 | 25 | 1697 | 2000 | `0.503394` | `0.820518` | `0.554042` | `0.481398` | `0.824222` | `0.523051` | `0/25` |
| val28 | 28 | 1892 | 2350 | `0.316072` | `0.670011` | `0.261471` | `0.266639` | `0.588409` | `0.216622` | `0/28` |
| target_unlabeled200 | 200 | 11750 | 14481 | `0.371575` | `0.716738` | `0.348056` | `0.312443` | `0.635325` | `0.276846` | `0/200` |

Artifacts:

- Train25: `output/diagnostics/r100_magformer_cleanfit_iter0500_train_labeled25_1024_backmap_topk200_20260518`
- Val28: `output/diagnostics/r100_magformer_cleanfit_iter0500_val28_1024_backmap_topk200_20260518`
- Target_unlabeled200: `output/diagnostics/r100_magformer_cleanfit_iter0500_target_unlabeled200_1024_backmap_topk200_20260518`

## Gate Decision

R100 passes.

- Train25 segm AP75 `0.523051` is above the `0.10` gate.
- It is close to historical clean overfit AP75 around `0.56`, so this is a clear clean-fit recovery.
- Val28 and target_unlabeled200 were run because the train25 gate passed.

## Interpretation

The clean train-fit setup fixes the core R99 symptom. The most likely damaging R99 factors were optimization and augmentation/noise choices, especially `base_lr=1e-6`, warmup, `weight_decay=0.07`, RGB photo aug, and depth noise. The eval protocol and data loader are not the primary blocker.

Next step: keep this clean setup as the supervised target baseline, then add one factor at a time. The first controlled comparison should isolate optimization from augmentation: run R100 settings but re-enable only the R99 LR/WD/warmup, or keep R100 optimizer and re-enable only augment/noise. Do not change mask/dice weights before that ablation.

## Commits

- Config/docs placeholder: `a86e34a7834cb478842b1a83f449829306493ae9`
- Final docs-only update: this docs-only commit.
