# VC-SUDA Stage C R12 32K Source Plan - 2026-05-15

## Conclusion

R12 tests one variable: replace the Stage C supervised source branch from the pseudo-real 1.5K-derived source split to the original 32K train split. Loss weights, LR, threshold, LSJ scale, depth noise, target splits, and target-domain eval protocol stay inherited from R8B.

## Recovery Check

- Repo HEAD before edits: `5da733f108cbea08da2bda2d9c891a7e2aec00af`.
- Dirty files before edits were only untracked `output/diagnostics/` and `output/upper_bound/`; no config/docs/code partial from the disconnected worker was present.
- Existing tmux sessions were old eval/R3/R6/R8B/R10 shells. Captured panes showed completed or interrupted commands, not active training.
- `ps` showed no active MAGFormer train/eval process owned by this repo. GPU4-7 had only Xorg memory, about `15 MiB` each.
- Partial-name files found were older R4/smoke configs or managed symlink markers. They were unrelated to R12 and did not block continuation.

## 32K Data Check

- 32K root: `/home/hdd3/zhanghaonan/magformer/magformer_datasets/20260318_1K_32254`.
- Train annotation: `annotations/instances_train.json`.
- Train split: `images/train`, `depth/depth_npy/train`.
- Train images: `25,654`; train annotations: `1,398,531`.
- Val images: `3,276`; val annotations: `181,712`.
- Test images: `3,324`; test annotations: `176,389`.
- Category contract: one class, id `1`, name `component`.
- File check: train/val/test RGB and depth files had `0` missing files against annotation file names.
- Depth normalization used by R12 remains the Stage C/R8B contract: `norm=minmax`, `per_sample_norm=true`, `clip_min=0.0`, `clip_max=2.095623016357422`.

## Current Stage C Source Check

Current Stage C/R8B uses `data.dataset_root: magformer_datasets/pseudo_real_512` and `vc_suda.source_ann: annotations/instances_source.json`. That split has `1,008` source images and `61,652` annotations. It is not the 32K source split.

## R12 Design

- Config: `configs/vc_suda_stage_c_r12_32k_source_r8b_ckpt999_continue_1024_teacher8499.yaml`.
- Output: `output/vc_suda/stage_c_r12_32k_source_r8b_ckpt999_continue_1024_teacher8499`.
- Warm start: `output/vc_suda/stage_c_r8b_lsj10_low_lr_continue_1024_teacher8499/checkpoint_iter_0000999.pth`.
- Source root: `magformer_datasets/20260318_1K_32254`.
- Source annotation: `annotations/instances_train.json`.
- Target root remains `magformer_datasets/pseudo_real_512` through `data.dataset_root`.
- Target labeled/unlabeled/val annotations remain `instances_target_labeled.json`, `instances_target_unlabeled.json`, and `instances_val.json`.
- `checkpoint_max_keep: null` keeps all checkpoints.

## First Checkpoint Eval Gate

Evaluate the first checkpoint `checkpoint_iter_0000249.pth` on target-domain `target_unlabeled200` with the external 1024 backmap protocol used for R8B/R10/R11. Use bbox+segm for this gate, not training quick eval.

Hard stop R12 if first-checkpoint target_unlabeled200 segm AP is below `0.319162`, the known R8B ckpt999 target_unlabeled200 segm AP. Continue only if it is at least comparable and prediction counts/diagnostics do not show collapse.

## Launch Command

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 python -m torch.distributed.run --nproc_per_node=4 tools/train.py   --config configs/vc_suda_stage_c_r12_32k_source_r8b_ckpt999_continue_1024_teacher8499.yaml   --output-dir output/vc_suda/stage_c_r12_32k_source_r8b_ckpt999_continue_1024_teacher8499   --num-workers 2
```

## Launch Result

R12 was launched in tmux session `vc_suda_stage_c_r12_32k_source_20260515` using the command above with `--gpus 0,1,2,3` under `CUDA_VISIBLE_DEVICES=4,5,6,7`. The first attempt omitted the logical GPU override and failed before training with `RuntimeError: CUDA error: invalid device ordinal`; this was a launch mapping error, not an experiment variable change.

The corrected launch loaded the 32K source split on all four ranks, built the model, loaded R8B `ckpt999` with `missing keys: 0, unexpected keys: 0`, and entered training. It reached about iter `78/1000` with finite losses. Resource monitoring then showed GPU memory above the hard 90% cap on GPUs 4 and 7 (`22701/24576 MiB` and `22953/24576 MiB`). The run was stopped before the first checkpoint to obey the resource constraint.

No first-checkpoint target_unlabeled200 eval exists yet because `checkpoint_iter_0000249.pth` was not produced. The first-checkpoint gate remains: external target_unlabeled200 segm AP must be `>=0.319162`; otherwise hard stop.

## R12B Resource Policy Clarification

A later clarification changed the resource interpretation: the 90% hard cap applies to server RAM/CPU, not GPU memory. The stopped R12 run did not record CUDA OOM, DDP crash, non-finite loss, or eval failure; it was stopped only because GPU4/GPU7 memory exceeded the previously assumed GPU 90% cap. Under the clarified policy, that is not a stop condition.

R12B should therefore keep the R12 training configuration unchanged and only use a fresh output directory to avoid mixing logs with the stopped R12 output. Stop R12B on CUDA OOM, training crash, non-finite loss, or sustained CPU/RAM `>=90%`; do not stop solely for GPU memory above 90%.
