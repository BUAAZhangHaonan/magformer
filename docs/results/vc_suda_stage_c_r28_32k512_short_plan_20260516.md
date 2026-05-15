# VC-SUDA Stage C R28 32K_512 Short Gate - 2026-05-16

## Conclusion

R28 is a short single-variable check for the 32K_512 Stage C path. It keeps the R27 data, R8B `ckpt999` warm-start, pseudo threshold, unsupervised weight, source/target split contract, and 512 input size unchanged, then extends the smoke into a bounded 250-iteration run with bbox+segm quick eval.

## Single Variable

- Config: `configs/vc_suda_stage_c_r28_32k512_short_teacher8499.yaml`.
- Output: `output/vc_suda/stage_c_r28_32k512_short_teacher8499`.
- Source root: `magformer_datasets/20260318_1K_32254_512`.
- Target root: `magformer_datasets/pseudo_real_512`.
- Warm start: `output/vc_suda/stage_c_r8b_lsj10_low_lr_continue_1024_teacher8499/checkpoint_iter_0000999.pth`.
- Training length: `solver.max_iter: 250`.
- Built-in quick eval: `runtime.eval_iou_types: [bbox, segm]`, `eval_max_images: 28`, `eval_batch_size: 4`.
- Checkpoint retention: `runtime.checkpoint_max_keep: null`.

## Launch Command

Do not launch without the explicit PyTorch MSDeformAttn backend:

```bash
MAGFORMER_MS_DEFORM_ATTN_BACKEND=pytorch CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 python tools/train.py \
  --config configs/vc_suda_stage_c_r28_32k512_short_teacher8499.yaml \
  --output-dir output/vc_suda/stage_c_r28_32k512_short_teacher8499 \
  --gpus 0 \
  --num-workers 0
```

## Gate

Use the formal external `target_unlabeled200` 1024-backmap evaluation for the decision, not the built-in 28-image quick eval. The gate must report bbox+segm.

Success gate:

- Continue only if formal `target_unlabeled200` segm AP is at least `0.320048`.

Loose no-drop line:

- `0.319162` is only the wide no-drop reference from R8B `ckpt999`.
- A result between `0.319162` and `0.320048` is not a success. It can be recorded as non-collapsed, but it should not justify continuing R28.

Failure gate:

- Stop if formal `target_unlabeled200` segm AP is below `0.319162`.
- Stop if the eval cannot complete with bbox+segm under `MAGFORMER_MS_DEFORM_ATTN_BACKEND=pytorch`.

## Preflight

Run this before any training:

```bash
conda run -n magformer python tools/verify_vc_suda_stage.py \
  --config configs/vc_suda_stage_c_r28_32k512_short_teacher8499.yaml \
  --stage C \
  --emit-data-evidence \
  --evidence-max-samples 1
```

This preflight is read-only for model state and does not start training.
