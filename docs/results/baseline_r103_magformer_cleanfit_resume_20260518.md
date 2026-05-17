# R103 MagFormer Cleanfit Resume

Date: 2026-05-18
Branch: feature/vc-suda-sim2real
Config: `configs/baseline_supervised_r103_magformer_r100_resume_target_labeled25_1000.yaml`

## Goal

Test whether true-resuming R100 cleanfit from iter 500 to iter 1000 improves MagFormer/RGB-D target_unlabeled200 AP/AP75.

## Protocol

- Resume checkpoint: `output/baseline/r100_magformer_r98ckpt_target_labeled25_cleanfit_1024/checkpoint_iter_0000500.pth`
- Train split: `target_labeled25`
- Image size: `1024`
- Solver: R100 cleanfit settings, with `base_lr=5e-5`, `warmup_iters=0`, `warmup_factor=1.0`, `weight_decay=0.01`, and `max_iter=1000`.
- Initialization: true `runtime.resume`; `model.finetune_weights=null`.
- Aug/noise: `random_flip=none`, RGB photo augmentation disabled, depth noise disabled.
- Disabled paths: VC-SUDA, EMA, pseudo labels, offline pseudo, contrastive, and source retention.
- External eval protocol: `tools/evaluate_1024_backmap.py`, bbox+segm, model input `1024`, score threshold `0.05`, mask threshold `0.5`, inference topk `200`, COCO maxDets `200`, `--force-pytorch-msda`.

## Early Stop Rule

Evaluate val28 first at checkpoints near iter 600 and iter 750. If both val28 segm AP values are below `0.261`, stop training and record the result. Otherwise continue to iter 1000, then evaluate the best val checkpoint on target_unlabeled200.

## Baselines

| run | val28 segm AP | target_unlabeled200 segm AP | target_unlabeled200 segm AP50 | target_unlabeled200 segm AP75 |
| --- | ---: | ---: | ---: | ---: |
| R100 cleanfit iter500 | `0.266639` | `0.312443` | `0.635325` | `0.276846` |
| R102 aug/noise strong solver iter500 | `0.267721` | `0.310118` | `0.642171` | `0.269491` |

## Static Validation

Static config, checkpoint, data path, and loader sanity passed before training.

- `validate_config(strict=True)`: valid. The only message was the existing DPE informational note.
- R103 uses true resume: `runtime.resume=output/baseline/r100_magformer_r98ckpt_target_labeled25_cleanfit_1024/checkpoint_iter_0000500.pth`.
- `model.finetune_weights=null`, so no warm-start path is active.
- R100 checkpoint exists, has `iter=500`, and has `774` model state keys.
- Config fields checked: `max_iter=1000`, `checkpoint_period=100`, `image_size=1024`, `random_flip=none`, `base_lr=5e-5`, `warmup_iters=0`, `weight_decay=0.01`.
- Disabled flags checked: RGB photo aug, depth noise, VC-SUDA, runtime EMA, VC-SUDA EMA teacher, offline pseudo, source retention, and contrastive are all off.
- Annotation access checked: train25 `25` images / `1697` anns, val28 `28` images / `1892` anns, target_unlabeled200 `200` images / `11750` anns.
- Sample image paths under `magformer_datasets/pseudo_real_512/images/train` are accessible for all three annotation files.
- Loader smoke checked: train/val dataset lengths `25 / 25`; train batch images `[4, 3, 1024, 1024]`, depths `[4, 1, 1024, 1024]`, target mask counts `[100, 100, 50, 50]`.

## Training

Pending.

## External 1024 Backmap Eval

Pending.

## Conclusion

Pending.

## Commits

Pending.
