# VC-SUDA R42 Point65536 Original Split Plan - 2026-05-16

## Conclusion

R42 tests one variable: training-side mask point sampling changes from `train_num_points: 12544` to `65536`.

Use the R35/R12 original-split true-resume line as the baseline. Do not use R37 balanced `+25`, because R37 changed the split and already failed.

## Baseline Choice

- Base config: `configs/vc_suda_stage_c_r35_original_split_true_resume_1024_teacher8499.yaml`.
- R42 config: `configs/vc_suda_stage_c_r42_point65536_original_split_true_resume_1024_teacher8499.yaml`.
- Smoke config: `configs/vc_suda_stage_c_r42_point65536_original_split_true_resume_1024_teacher8499_smoke.yaml`.
- True resume: `runtime.resume: output/vc_suda/stage_c_r12_32k_source_r8b_ckpt999_continue_1024_teacher8499/checkpoint_iter_0000499.pth`.
- Model-only warm-start remains disabled: `model.finetune_weights: null`.

R35 is the correct control because it keeps the original R12 split and true-resumes from R12 `ckpt499`. Its external `target_unlabeled200` 1024 backmap segm AP was `0.319300`, close to the R12 current-code baseline `0.320048`.

## Fixed Variables

Keep these unchanged from R35/R12:

- Original split: `annotations/instances_target_labeled.json` and `annotations/instances_target_unlabeled.json`.
- `data.image_size: 1024`.
- `solver.max_iter: 750` in the formal config.
- `solver.ims_per_batch: 4`.
- `solver.base_lr: 1.0e-05`.
- `vc_suda.unsupervised_weight: 0.02`.
- `vc_suda.pseudo_label.quality_threshold: 0.1`.
- `model.magformer.mask_former.num_object_queries: 200`.
- `model.magformer.mask_former.oversample_ratio: 3.0`.
- `model.magformer.mask_former.importance_sample_ratio: 0.75`.

The only formal training variable is `model.magformer.mask_former.train_num_points: 12544 -> 65536`.

## Why R41 Stops The Resolution Route

R41 changed only eval image size from `1024` to `1536` for R12 `ckpt499`. It failed: formal segm AP was `0.284495`, below the 1024 baseline `0.320048`.

So R42 does not continue inference-resolution changes. It tests whether denser training mask point sampling can improve the mask-set coverage and localization bottleneck found by R38/R40.

## Smoke Gate

Run the smoke config in tmux with clean env, CUDA MSDA, physical GPUs `4,5,6,7`, local `--gpus 0,1,2,3`, and `--num-workers 0`.

Pass requires all of these:

- Strictly load R12 `checkpoint_iter_0000499.pth` through `runtime.resume`.
- No missing, unexpected, or shape mismatch exception.
- Complete at least one training step from iter `499` to iter `500`.
- Loss stays finite.
- No Traceback, CUDA OOM, or non-finite error.
- Server RAM stays below `90%`.
- Record peak GPU memory and wall time.

If smoke OOMs, stop R42 and record the OOM. Do not lower to `32768` in this run.

## Smoke Result

Smoke passed on 2026-05-16 in tmux session `r42_point65536_smoke`.

Command:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 CUDA_DEVICE_ORDER=PCI_BUS_ID \
MAGFORMER_MS_DEFORM_ATTN_BACKEND=cuda \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
PYTHONUNBUFFERED=1 \
python -m torch.distributed.run --standalone --nproc_per_node=4 tools/train.py \
  --config configs/vc_suda_stage_c_r42_point65536_original_split_true_resume_1024_teacher8499_smoke.yaml \
  --gpus 0,1,2,3 \
  --num-workers 0
```

Evidence:

- Log: `output/vc_suda/stage_c_r42_point65536_original_split_true_resume_1024_teacher8499_smoke/smoke_tmux.log`.
- Resource monitor: `output/vc_suda/stage_c_r42_point65536_original_split_true_resume_1024_teacher8499_smoke/smoke_resource_monitor_recovery.csv`.
- Exit code: `0`.
- Elapsed time: `202` seconds.
- R12 `ckpt499` loaded through `runtime.resume`.
- Trainer resumed from iter `499` and completed iter `500`.
- Loss at iter `500` was finite on all ranks: `21.2013`, `23.0383`, `26.1506`, `19.5734`.
- No Traceback, CUDA OOM, non-finite error, missing/unexpected key error, or shape mismatch error was found in the log scan.
- Peak sampled GPU memory: GPU4 `21459 MiB`, GPU5 `21219 MiB`, GPU6 `21219 MiB`, GPU7 `21399 MiB`.
- Peak sampled server RAM: `164807426048/270094807040` bytes, about `61.0%`.

This smoke only validates plumbing and memory for the one-step true-resume path. It does not authorize the formal 250-iter gate by itself.

## Formal 250 Iter Gate

Do not start the formal gate until the smoke passes and the main thread decides to continue.

If approved later, run the formal R42 config to iter `750`, then evaluate external `target_unlabeled200` at 1024 backmap with bbox+segm and topk/maxDets `200`.

Decision:

- `segm AP >= 0.320048`: point65536 improves or preserves the R12 line; consider a controlled follow-up.
- `0.319162 <= segm AP < 0.320048`: record-only; do not extend without a new reason.
- `segm AP < 0.319162`: fail and stop R42.

## Stop Conditions

Stop immediately on any of these:

- CUDA OOM.
- Traceback or launcher failure.
- Non-finite loss.
- Server RAM at or above `90%`.
- Checkpoint resume is not R12 `ckpt499`.
- Config diff shows any formal variable drift beyond identity/output/log paths and `train_num_points`.
- Smoke config diff shows any drift beyond smoke identity/output paths and `solver.max_iter: 500`.
