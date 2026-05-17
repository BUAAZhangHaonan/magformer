# R99 MagFormer R98 Target-Labeled25 Finetune - 2026-05-18

Status: config added; training and target_unlabeled200 gate pending.

## Config

- Config: `configs/baseline_supervised_r99_magformer_r98ckpt_target_labeled25_1024.yaml`
- Base: R98 supervised-only MagFormer RGB-D config.
- Warm start: `output/baseline/r98_magformer_32k_source_short/checkpoint_iter_0000499.pth`
- Train ann: `magformer_datasets/pseudo_real_512/annotations/instances_target_labeled.json`
- Eval ann: `magformer_datasets/pseudo_real_512/annotations/instances_target_unlabeled.json`
- VC-SUDA, EMA, pseudo labels, contrastive, and unsupervised losses: disabled.

## Pending

- Static validation and loader smoke.
- 500 iter tmux training.
- target_unlabeled200 1024 backmap bbox+segm eval with topk/maxDets 200.
- Gate decision.
