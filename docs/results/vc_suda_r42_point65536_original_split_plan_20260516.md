# VC-SUDA R42 Point65536 Original Split Plan - 2026-05-16

## Conclusion

R42 tested one variable: training-side mask point sampling changed from `train_num_points: 12544` to `65536`.

The gate fails. Formal segm AP/AP75 are both below the R12/R33 baseline, and the R38/R40 oracle diagnostics do not show a compensating coverage gain.

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

The formal R42 run completed before this result update.

Decision:

- `segm AP >= 0.320048`: point65536 improves or preserves the R12 line; consider a controlled follow-up.
- `0.319162 <= segm AP < 0.320048`: record-only; do not extend without a new reason.
- `segm AP < 0.319162`: fail and stop R42.

Result: fail. R42 reached segm AP `0.3189000934`, below the fail line `0.319162`.

## Formal Eval Result

Checkpoint:

- `output/vc_suda/stage_c_r42_point65536_original_split_true_resume_1024_teacher8499/checkpoint_iter_0000750.pth`

External eval directory:

- `output/experiments/vc_suda_stage_c_r42_point65536_iter0750_target_unlabeled200_1024_backmap_topk200_20260516`

Formal metrics from `metrics.cocoeval.json`:

| Metric | AP | AP50 | AP75 |
|---|---:|---:|---:|
| bbox | `0.394389` | `0.739593` | `0.379503` |
| segm | `0.318900` | `0.648652` | `0.282661` |

Prediction and topk stats:

- Prediction JSON: `13,851` predictions.
- Inference stats exported count sum: `13,851`.
- Topk limit: `200`.
- Topk truncated: `0 / 200` images.

Baseline comparison:

| Line | segm AP | segm AP75 |
|---|---:|---:|
| R12/R33 baseline | `0.320048` | `0.284144` |
| R42 point65536 | `0.318900` | `0.282661` |
| Delta | `-0.001148` | `-0.001483` |

## R38 Oracle-Score Bound

Existing R42 oracle output was read from:

- `output/diagnostics/r42_point65536_oracle_score_upper_bound_20260516/r42_point65536_oracle_bound_summary.json`

Oracle-score matched-IoU segm metrics:

| Metric | Value |
|---|---:|
| segm AP | `0.359406` |
| segm AP50 | `0.683168` |
| segm AP75 | `0.346535` |

Coverage metrics:

| Metric | R38 baseline | R42 point65536 | Delta |
|---|---:|---:|---:|
| Global oracle R@75 | `0.347830` | `0.343745` | `-0.004085` |
| Dense `>90` R@75 | `0.172577` | `0.167204` | `-0.005373` |
| Small `<=256` R@75 | `0.009884` | `0.009202` | `-0.000682` |

The oracle-score AP is also below the R38 baseline oracle-score segm AP `0.361386`. The miss source did not improve.

## R40 Miss Atlas

The first R42 atlas attempt failed because the temporary script does not accept `--dataset-root`. Help confirms the real CLI:

```bash
/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python /tmp/r40_oracle_miss_atlas.py --help
```

Supported arguments:

```text
--ann ANN --pred PRED --out-dir OUT_DIR
```

Corrected command:

```bash
/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python /tmp/r40_oracle_miss_atlas.py \
  --ann magformer_datasets/pseudo_real_512/annotations/instances_target_unlabeled.json \
  --pred output/experiments/vc_suda_stage_c_r42_point65536_iter0750_target_unlabeled200_1024_backmap_topk200_20260516/coco_instances_results.json \
  --out-dir output/diagnostics/r42_point65536_miss_atlas_20260516
```

Output summary:

- `output/diagnostics/r42_point65536_miss_atlas_20260516/r40_oracle_miss_atlas_summary.json`

Overall instance classes against `11,750` GT masks:

| Class | Definition | Count | Rate |
|---|---|---:|---:|
| no-cover | best IoU `<0.5` | `3,710` | `31.57%` |
| low-quality | `0.5<=IoU<0.75` | `4,001` | `34.05%` |
| good | `0.75<=IoU<0.9` | `3,610` | `30.72%` |
| high-quality | IoU `>=0.9` | `429` | `3.65%` |

R40 atlas R@50/R@75/R@90 was `0.684255 / 0.343745 / 0.036511`. RGB and depth side channels were readable for `200 / 200` images, and the three contact sheets were generated.

## Gate Decision

Fail and stop R42.

Reasons:

- Formal segm AP `0.318900` is below both the R42 fail line `0.319162` and the R12/R33 baseline `0.320048`.
- Formal segm AP75 `0.282661` is below the R12/R33 baseline `0.284144`.
- Oracle and miss-atlas metrics are slightly worse than R38/R40 baseline, not a hidden improvement: global oracle R@75 `0.343745 < 0.347830`, dense `>90` R@75 `0.167204 < 0.172577`, and small `<=256` R@75 `0.009202 < 0.009884`.

Do not extend R42 and do not start a follow-up from this result.

## Stop Conditions

Stop immediately on any of these:

- CUDA OOM.
- Traceback or launcher failure.
- Non-finite loss.
- Server RAM at or above `90%`.
- Checkpoint resume is not R12 `ckpt499`.
- Config diff shows any formal variable drift beyond identity/output/log paths and `train_num_points`.
- Smoke config diff shows any drift beyond smoke identity/output paths and `solver.max_iter: 500`.
