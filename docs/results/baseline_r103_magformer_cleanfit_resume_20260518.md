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

- tmux session: `r103_magformer_cleanfit_resume`
- Command shape: `/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/torchrun --nproc_per_node=4 tools/train.py --config configs/baseline_supervised_r103_magformer_r100_resume_target_labeled25_1000.yaml --gpus 4,5,6,7`
- Output dir: `output/baseline/r103_magformer_r100_resume_target_labeled25_1000`
- Resume evidence: all ranks loaded `output/baseline/r100_magformer_r98ckpt_target_labeled25_cleanfit_1024/checkpoint_iter_0000500.pth` and reported `Resumed from iteration 500`.
- Training window: started at `2026-05-18 04:32:36 +0800`, completed at `2026-05-18 04:43:39 +0800`.
- Checkpoints used for external eval: `checkpoint_iter_0000599.pth`, `checkpoint_iter_0000799.pth`, and `checkpoint_iter_0001000.pth`. The off-by-one checkpoint names come from the trainer saving when `(current_iter + 1) % checkpoint_period == 0`.
- Final checkpoint: `output/baseline/r103_magformer_r100_resume_target_labeled25_1000/checkpoint_iter_0001000.pth`.

Built-in trainer eval remains diagnostic only. The decision metrics below use the external 1024 backmap protocol.

## External 1024 Backmap Eval

Protocol for all rows: bbox+segm, model input `1024`, score threshold `0.05`, mask threshold `0.5`, inference topk `200`, COCO maxDets `200`, `MAGFORMER_MS_DEFORM_ATTN_BACKEND=pytorch`, `--force-pytorch-msda`, strict load `774/774` keys.

Val28 trajectory:

| checkpoint | predictions | bbox AP | bbox AP50 | bbox AP75 | segm AP | segm AP50 | segm AP75 | decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| iter0599 | 2354 | `0.315387` | `0.663157` | `0.258500` | `0.266663` | `0.582805` | `0.218312` | above early-stop floor |
| iter0799 | 2297 | `0.315715` | `0.659388` | `0.268678` | `0.272399` | `0.589279` | `0.227188` | best val checkpoint |
| iter1000 | 2296 | `0.315590` | `0.660504` | `0.274256` | `0.270209` | `0.588452` | `0.224371` | below iter0799 |

Best-val checkpoint `iter0799` full eval:

| eval | images | GT anns | predictions | bbox AP | bbox AP50 | bbox AP75 | segm AP | segm AP50 | segm AP75 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train25 | 25 | 1697 | 1872 | `0.536601` | `0.840128` | `0.614568` | `0.527663` | `0.846610` | `0.610345` |
| val28 | 28 | 1892 | 2297 | `0.315715` | `0.659388` | `0.268678` | `0.272399` | `0.589279` | `0.227188` |
| target_unlabeled200 | 200 | 11750 | 14012 | `0.371695` | `0.717595` | `0.347796` | `0.314747` | `0.636336` | `0.284706` |

Artifacts:

- Val28 iter0599: `output/diagnostics/r103_magformer_cleanfit_resume_iter0599_val28_1024_backmap_topk200_20260518`
- Val28 iter0799: `output/diagnostics/r103_magformer_cleanfit_resume_iter0799_val28_1024_backmap_topk200_20260518`
- Val28 iter1000: `output/diagnostics/r103_magformer_cleanfit_resume_iter1000_val28_1024_backmap_topk200_20260518`
- Train25 iter0799: `output/diagnostics/r103_magformer_cleanfit_resume_iter0799_train_labeled25_1024_backmap_topk200_20260518`
- Target_unlabeled200 iter0799: `output/diagnostics/r103_magformer_cleanfit_resume_iter0799_target_unlabeled200_1024_backmap_topk200_20260518`

## Early Stop

Training was not stopped early. The 600-near val point was already above the `0.261` floor, and the 750-near point improved to `0.272399` segm AP.

## Comparison

| run | checkpoint | train25 segm AP75 | val28 segm AP | target_unlabeled200 segm AP | target_unlabeled200 segm AP50 | target_unlabeled200 segm AP75 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| R100 cleanfit | iter0500 | `0.523051` | `0.266639` | `0.312443` | `0.635325` | `0.276846` |
| R102 aug/noise strong solver | iter0500 | `0.351640` | `0.267721` | `0.310118` | `0.642171` | `0.269491` |
| R103 cleanfit resume | iter0799 best-val | `0.610345` | `0.272399` | `0.314747` | `0.636336` | `0.284706` |

## Conclusion

R103 beats R100 on the main target_unlabeled200 segmentation metrics. Target AP improves by `+0.002304`, AP50 by `+0.001011`, and AP75 by `+0.007860`. Val28 also improves by `+0.005760` AP.

The gain is small but consistent with the resume hypothesis. The best checkpoint is `iter0799`, not the final `iter1000`, so model selection should use external val28 rather than the final training step.

## Commits

- Config/docs placeholder: `212c4ec5fef3e97719bb6b327cc921863e648477`
- Final docs-only update: this docs-only commit.
