# VC-SUDA Stage C R12B 32K Source Policy - 2026-05-15

## Conclusion

R12B should run the original R12 training setup unchanged. The previous R12 was stopped because GPU memory exceeded 90%, but the clarified resource rule says the 90% cap applies to server RAM/CPU, not GPU memory. GPU memory can approach full capacity as long as the run does not OOM and does not affect other processes.

## R12B Design

- Config: `configs/vc_suda_stage_c_r12b_32k_source_r8b_ckpt999_continue_1024_teacher8499.yaml`.
- Output: `output/vc_suda/stage_c_r12b_32k_source_r8b_ckpt999_continue_1024_teacher8499`.
- R12B differs from failed R12 only in `name`, `runtime.output_dir`, and logger path/run name so the new run does not mix logs with the stopped R12 output.
- Training config remains unchanged: `solver.ims_per_batch: 4`, `runtime.grad_accum_steps: 1`, `runtime.eval_batch_size: 4`, `image_size: 1024`, bbox-only val28 quick eval, and `checkpoint_max_keep: null`.
- No activation checkpointing, batch reduction, LR change, loss change, threshold change, augmentation change, or pseudo-label change is introduced.

## Evidence From Failed R12

The failed R12 log reached about iter `78/1000` with finite losses. `runtime.eval_period: 1000`, so quick eval was not running. No checkpoint was produced. The stop was triggered by GPU4/GPU7 memory readings of `22701/24576 MiB` and `22953/24576 MiB`, but there was no recorded CUDA OOM, DDP crash, non-finite loss, or target eval failure. Under the clarified policy, that is not a stop condition.

## Difference From R8B

The only experiment variable remains the source split: R8B used `magformer_datasets/pseudo_real_512/annotations/instances_source.json`, while R12B uses `magformer_datasets/20260318_1K_32254/annotations/instances_train.json`. Target splits, loss, LR, thresholds, LSJ, depth noise, and target eval protocol stay inherited from R8B/R12.

## Hard Stop Conditions

- Stop on CUDA OOM, non-finite loss, DDP hang/crash, or missing/invalid 32K source data.
- Stop if server RAM or CPU is `>= 90%` for sustained monitoring samples.
- Do not stop solely because GPU memory is above 90%.
- At `checkpoint_iter_0000249.pth`, run external target_unlabeled200 1024 backmap bbox+segm eval. Stop if segm AP is below R8B `ckpt999` target_unlabeled200 segm AP `0.319162`; continue only if prediction diagnostics do not collapse.

## Launch Command

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 python -m torch.distributed.run --nproc_per_node=4 tools/train.py   --config configs/vc_suda_stage_c_r12b_32k_source_r8b_ckpt999_continue_1024_teacher8499.yaml   --output-dir output/vc_suda/stage_c_r12b_32k_source_r8b_ckpt999_continue_1024_teacher8499   --gpus 0,1,2,3   --num-workers 2
```
