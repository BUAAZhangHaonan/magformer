# VC-SUDA R33 R12 ckpt499 sanity eval - 2026-05-16

## Conclusion

R33 no-train sanity reproduced the R12/R15 `ckpt499` target_unlabeled200 1024 backmap baseline with the current code, data, and eval script.

The new segm AP is `0.3200483457`. Against the historical R15 reference `0.320048`, the delta is `+0.0000003457`, well inside the `+/-0.002` tolerance. This does not show evaluation benchmark drift.

## Historical Reference

- R12 checkpoint: `output/vc_suda/stage_c_r12_32k_source_r8b_ckpt999_continue_1024_teacher8499/checkpoint_iter_0000499.pth`
- Historical R12/R15 protocol: `target_unlabeled200`, 1024 backmap, bbox+segm, `--inference-topk 200 --max-dets 200`, forced PyTorch MSDA.
- Historical R15 segm AP reference: `0.320048`.
- Historical R15 prediction count: `13,806`.

## R33 Eval

- tmux session: `r33_r12_ckpt499_sanity_eval_20260516`
- Output dir: `output/diagnostics/r33_r12_ckpt499_sanity_unlabeled200_20260516`
- Full log: `output/diagnostics/r33_r12_ckpt499_sanity_unlabeled200_20260516/eval.log`
- Metrics: `output/diagnostics/r33_r12_ckpt499_sanity_unlabeled200_20260516/metrics.cocoeval.json`
- Predictions: `output/diagnostics/r33_r12_ckpt499_sanity_unlabeled200_20260516/coco_instances_results.json`
- Inference stats: `output/diagnostics/r33_r12_ckpt499_sanity_unlabeled200_20260516/inference_stats.json`

Command contract:

```bash
CUDA_VISIBLE_DEVICES=4 \
MAGFORMER_MS_DEFORM_ATTN_BACKEND=pytorch \
/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python tools/evaluate_1024_backmap.py \
  --base-config output/experiments/vc_suda_stage_c_r12_32ksource_iter0500_target_unlabeled200_1024_backmap_20260515_1606/eval_base_r12_ckpt499.yaml \
  --dataset-root magformer_datasets/pseudo_real_512 \
  --ann annotations/instances_target_unlabeled.json \
  --split train \
  --weights output/vc_suda/stage_c_r12_32k_source_r8b_ckpt999_continue_1024_teacher8499/checkpoint_iter_0000499.pth \
  --output-dir output/diagnostics/r33_r12_ckpt499_sanity_unlabeled200_20260516 \
  --image-size 1024 \
  --batch-size 4 \
  --num-workers 4 \
  --score-threshold 0.05 \
  --mask-threshold 0.5 \
  --iou-types bbox,segm \
  --inference-topk 200 \
  --max-dets 200 \
  --dump-inference-stats output/diagnostics/r33_r12_ckpt499_sanity_unlabeled200_20260516/inference_stats.json \
  --force-pytorch-msda
```

The command was launched from a clean `env -i` tmux shell with only the required home/user/shell/path, CUDA, thread-count, and PyTorch-MSDA variables set.

## Metrics

| Run | bbox AP | bbox AP50 | bbox AP75 | segm AP | segm AP50 | segm AP75 | Predictions | topk truncated |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Historical R15 | 0.394476 | 0.741115 | 0.379187 | 0.320048 | 0.648363 | 0.284144 | 13,806 | 0/200 |
| R33 sanity | 0.3944763203 | 0.7411151776 | 0.3791865294 | 0.3200483457 | 0.6483631209 | 0.2841442923 | 13,806 | 0/200 |

Readout:

- segm AP delta vs `0.320048`: `+0.0000003457`.
- The result is inside `+/-0.002`, so the current eval baseline still reproduces.
- No benchmark drift is indicated by this sanity check.

## Validation

- Evaluated all `200` target-unlabeled images.
- Weight load was strict: `774/774` matched, `0` missing, `0` unexpected, `0` shape mismatch.
- Forced PyTorch MSDA path was active: `[MSDeformAttn] forced PyTorch core path`.
- Log scan found no `Traceback`, `CUDA error`, `OOM`, `out of memory`, `RuntimeError`, or `non-finite` hits.
- Sampled RAM stayed below 90%; observed high sample during eval was about `18,889 MiB / 257,582 MiB` (`7.33%`).
- GPU4 was released after eval: final sample `15 MiB / 24,576 MiB`, utilization `0%`.
- No code, config, or training changes were made.
