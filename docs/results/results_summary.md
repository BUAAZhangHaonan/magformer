# Multi-Model Evaluation Results
- [VC-SUDA Stage C R28 32K_512 short gate plan, 2026-05-16](vc_suda_stage_c_r28_32k512_short_plan_20260516.md): planned a bounded 250-iteration single-variable continuation from R27's 32K_512 Stage C smoke, with explicit `MAGFORMER_MS_DEFORM_ATTN_BACKEND=pytorch`, bbox+segm quick eval on 28 images, and a formal external target_unlabeled200 bbox+segm gate requiring segm AP `>=0.320048` to continue; `0.319162` is only the loose no-drop reference.
- [VC-SUDA R27 32K_512 preflight, 2026-05-16](vc_suda_r27_32k512_preflight_20260516.md): validated the materialized 32K_512 data path, documented the explicit PyTorch MSDeformAttn backend requirement, passed Stage A and Stage C smoke runs, and recorded the eval-only 28-image bbox-only baseline: 28 images, 1,456 predictions, bbox AP/AP50/AP75 `0.117524/0.380337/0.039159`, APs `0.172600`, score p50/p90 `0.922438/0.949866`. This is not the formal 1024 backmap segm metric.
- [VC-SUDA Stage C R25 32K->512 shard validation, 2026-05-15](vc_suda_stage_c_r25_32k_512_shard_validation_20260515.md): validated a real 50-image train shard from `20260318_1K_32254` through `build_multires_dataset.py` at 512. RGB/depth counts matched, all 2,095 masks were non-empty with exact bbox/area agreement, category stayed `component` id=1, manifests/stats existed, and Detectron2 RGBD plus UCN DataLoader read-only sanity passed. Full 32K_512 is safe to materialize with about 200GB disk budget; no full conversion or training was run.
- [VC-SUDA Stage C R23 pseudo bank tool, 2026-05-15](vc_suda_stage_c_r23_pseudo_bank_20260515.md): added `tools/build_pseudo_bank.py` for reproducible offline 1024 pseudo-bank construction and hidden-GT quality reports. R15 topk200 simple high-precision bank kept `6,052` with P50/R50/P75/R75 `0.9377/0.4830/0.6067/0.3125`; dynamic density bank kept `5,461` with `0.9515/0.4422/0.6484/0.3014`. Dense `90-100` recall remains low, so do not connect these banks to training yet.
- [VC-SUDA Stage C R20 exterior ring stopped, 2026-05-15](vc_suda_stage_c_r20_exterior_ring_plan_20260515.md): R20 `ckpt249` external target_unlabeled200 1024 backmap topk200/maxDets200 eval produced bbox/segm AP `0.390561/0.316952`, segm AP50/AP75 `0.648726/0.280995`, with `13,652` predictions and `0/200` topk-truncated images. This misses the hard lines, segm AP `0.3195` and AP75 `0.284144`, so R20 was stopped and `ckpt499` was not evaluated.
- [VC-SUDA Stage C R19 exterior ring dry-run, 2026-05-15](vc_suda_stage_c_r19_exterior_ring_dryrun_20260515.md): added default-off matched-mask exterior-ring instrumentation to the pseudo diagnostic path. On two R12 `ckpt499` dry-run batches, matched pseudo count was `79`, exterior ring prob mean/p50/p90 was `0.111685/0.091848/0.217193`, interior prob mean/p50 was `0.833223/0.898117`, far-background mean was `0.000129`, and pred/target area ratio mean/p90 was `1.085093/1.269596`. The ring signal is high enough versus far background to justify one short, low-weight radius-2 train check, but not a sweep.
- [VC-SUDA Stage C R18 pseudo unmatched negative stopped, 2026-05-15](vc_suda_stage_c_r18_pseudo_unmatched_negative_plan_20260515.md): R18 `ckpt249` external target_unlabeled200 1024 backmap topk200/maxDets200 eval produced bbox/segm AP `0.393006/0.318701`, segm AP50/AP75 `0.648298/0.282244`, with `13,760` predictions. This clears the R7 stopline `0.3171` by `+0.001601`, but is below current best R15/R12 `0.320048` by `-0.001347` and below `0.319`, so R18 was stopped after first checkpoint. Remote branch includes R18 commit `6e43d0c840e542142967632eb091de214940d8de`.
- [VC-SUDA Stage C R17 signal diagnostics, 2026-05-15](vc_suda_stage_c_r17_signal_diagnostics_20260515.md): added no-step dry-run instrumentation for source, target_labeled, and pseudo branch losses plus pseudo matched/unmatched high-score query counts. On two R12-config batches, R12 `ckpt499` kept/matched `79/79` pseudo instances but still had `37` unmatched queries above score `0.7` and `29` above `0.9`; R8B `ckpt999` had `41` and `37`. Current pseudo loss supervises only matched pseudo positives, so these high-score unmatched queries have no direct pseudo negative gradient.
- [VC-SUDA Stage C R15 topk/maxDets recovery, 2026-05-15](vc_suda_stage_c_r15_topk200_maxdets200_20260515.md): eval-only R12 `ckpt499` target_unlabeled200 1024 backmap with `--inference-topk 200 --max-dets 200` removed topk truncation (`0/200` images truncated vs R14 `200/200`) and increased exported predictions from `13,122` to `13,806`. Metrics changed only slightly: bbox AP `0.393266 -> 0.394476`, segm AP `0.320046 -> 0.320048`. Dense truncation is solved, but it is not the main segm AP bottleneck.
- [VC-SUDA Stage C R14 inference instrumentation, 2026-05-15](vc_suda_stage_c_r14_inference_instrumentation_20260515.md): added opt-in `--dump-inference-stats` instrumentation for pre-topk, post-topk, post-score, post-mask-nonempty, exported count, and GT-density buckets. Full R12 `ckpt499` target_unlabeled200 1024 backmap diagnostic showed `pre_topk_candidate_count=200`, `topk_limit=100`, and `topk_truncated_images=200/200`; 10 dense images exported exactly 100 predictions. This confirms a topk/maxDets blocking risk and supports one eval-only raised-topk/maxDets experiment before any more training.
- [VC-SUDA Stage C R12B 32K source policy, 2026-05-15](vc_suda_stage_c_r12b_32k_source_policy_20260515.md): R12B keeps the failed R12 training setup unchanged after clarification that the 90% cap applies to server RAM/CPU, not GPU memory. The previous stop was triggered by GPU4/GPU7 memory above 90% without CUDA OOM, non-finite loss, checkpoint, or eval. R12B uses a fresh output directory only; stop on OOM/crash/non-finite loss or sustained CPU/RAM >=90%, and keep the first checkpoint external target_unlabeled200 segm AP gate `>=0.319162`.

## Overview

- [VC-SUDA Stage C R12 32K source final eval, 2026-05-15](vc_suda_stage_c_r12_32k_source_plan_20260515.md): restarted R12 completed `1000/1000` and external 1024 backmap target_unlabeled200 `ckpt999` eval produced bbox/segm AP `0.391647/0.319292`, segm AP50/AP75 `0.649748/0.283178`, with 12,946 predictions. This is above R8B `ckpt999` segm AP `0.319162` by `+0.000130`, but below R12 `ckpt499` `0.320046` by `-0.000754`; R12 best checkpoint remains `ckpt499`.
- **Dataset**: 1,566 synthetic PCB component images (~95,895 annotations, ~61 objects/image avg)
- **Resolutions**: 512 px and 1024 px
- **Models evaluated**: 20 baselines + MagFormer v13 (final optimized version)
- **Training modes**: fine-tuned (pretrained on COCO/ImageNet then finetuned on PCB data) or from-scratch
- **Metrics**: COCO-style bbox AP and segm AP (AP, AP50, AP75), F1@50, model params, training time, training/inference VRAM, inference FPS
- **Hardware**: 2-8 NVIDIA GPUs (varies by model, see notes)

Data source: automated benchmark pipeline, April 2026. MagFormer v13 results from May 2026.

Recent VC-SUDA diagnostics:

- [VC-SUDA Stage C R11 mask-loss continuation, 2026-05-15](vc_suda_stage_c_r11_mask_loss_plan_20260515.md): continued from R8B `ckpt999` with the single variable `mask_weight/dice_weight 5.0 -> 7.5`. External 1024 backmap `ckpt249` target_unlabeled200 reached bbox/segm AP `0.3890897318/0.3176062533` with segm AP50/AP75 `0.6497872641/0.2799362356` and 12,970 predictions. This clears the hard-stop floor R7 `0.3171` by `+0.0005062533`, but remains below R8B `ckpt999` `0.3191621968` by `-0.0015559435` and below global best R8B `ckpt749` `0.319922` by `-0.0023157467`; R11 was not stopped at `ckpt249`, but `ckpt499` needs the next gate check.
- [VC-SUDA Stage C R10 no-depth-noise stopped, 2026-05-15](vc_suda_stage_c_r10_no_depth_noise_plan_20260515.md): continued from R8B `ckpt999`, kept R8B settings, and set only `data.depth_noise.gaussian_std=0.0`. External 1024 backmap `ckpt249` target_unlabeled200 reached bbox/segm AP `0.3903716504/0.3164998465` with segm AP50/AP75 `0.6427576211/0.2760591309` and 12,772 predictions. This is below R8B `ckpt999` segm AP `0.3191621968` by `-0.0026623503` and below global best R8B `ckpt749` segm AP `0.319922` by `-0.0034221535`; disabling Gaussian depth noise did not improve early AP, so R10 was stopped and no val28 run was done.
- [VC-SUDA Stage C R9B threshold and retention findings, 2026-05-15](vc_suda_stage_c_r9_threshold_and_retention_20260515.md): the planned R9 `ckpt749` mask sweep was blocked because old hardcoded retention deleted `checkpoint_iter_0000749.pth`. Retention was fixed in `cc134100cd9d8f23db33463c6f596045eef2eb16` with `runtime.checkpoint_max_keep`; `null` or `0` keeps all checkpoints and default `2` is unchanged. The available R9B `ckpt999` mask-threshold sweep at `0.35/0.40/0.45/0.55/0.60/0.65` did not beat the `ckpt999` baseline target_unlabeled200 segm AP `0.319162` or the global best `0.319922`; no val28 reruns were done.
- [VC-SUDA Stage C R8/R8B results, 2026-05-15](vc_suda_stage_c_r8_r8b_results_20260515.md): R8 score threshold sweep at `0.075/0.10/0.125/0.15` did not improve over R7 target_unlabeled200 segm AP `0.3171`. R8B low-LR continuation from R7 `ckpt1999` reaches a new target_unlabeled200 best at `ckpt749` with segm AP `0.319922`, but val28 best is only `ckpt249` segm AP `0.278051`; gains are small and still far below 61+.
- [VC-SUDA Stage C R8B low-LR continuation plan, 2026-05-15](vc_suda_stage_c_r8b_low_lr_continue_plan_20260515.md): R8 score sweep at `0.075/0.10/0.125/0.15` did not beat R7 target_unlabeled200 segm AP `0.3171`; higher thresholds only reduced predictions. R8B continues from R7 `ckpt1999` with `base_lr=1e-5`, `max_iter=1000`, and must stop at `ckpt249` unless external target_unlabeled200 segm AP is `>=0.3171`.
- [VC-SUDA Stage C R7 fixed LSJ scale result, 2026-05-15](vc_suda_stage_c_r7_lsj10_plan_20260515.md): fixed LSJ to `1.0/1.0` while keeping the R3-A10 contrastive setup. External 1024 backmap `ckpt1999` is current best with target_unlabeled200 bbox/segm AP `0.3916/0.3171`, `+0.0103` segm AP over R3 `0.3068`; prediction count dropped from R3 `15421` to R7 `13017`, but the result remains far below 61+.
- [VC-SUDA Stage C R6 no-contrast stopped, 2026-05-15](vc_suda_stage_c_r6_nocontrast_plan_20260515.md): external 1024 backmap `ckpt499` target_unlabeled200 segm AP `0.2410`, `-0.0658` vs R3 `0.3068`; below stop threshold `0.02`, so the R6 no-contrast run was stopped and should not continue.
- [Clean target_labeled overfit diagnostic, 2026-05-15](vc_suda_clean_overfit_diagnostic_20260515.md): ckpt1499 reaches same-set 1024 backmap bbox/segm AP `0.6669/0.6222`, but generalization remains low. Use external 1024 backmap for model selection; do not use training-time 512 quick eval as best-checkpoint selection.
- [Target unlabeled diagnostic, 2026-05-15](vc_suda_target_unlabeled_diagnostic_20260515.md): Stage B vs R3-A10 on target_unlabeled200.
- [Target hidden-GT upper-bound diagnosis, 2026-05-15](target_hidden_gt_upper_bound_plan_20260515.md): built-in 512 eval is not comparable to 1024 backmap reporting.

---

## Table 1: 1024 px Resolution (sorted by bbox AP)

| # | Model | bbox AP | AP50 | AP75 | segm AP | AP50 | AP75 | F1@50 | Params (M) | Train (h) | Train VRAM (GB) | Inf. (ms) | Inf. VRAM (GB) | FPS |
|---|-------|---------|------|------|---------|------|------|-------|------------|-----------|-----------------|-----------|----------------|-----|
| 1 | **MagFormer v13 (TTA)** | **70.55** | 88.32 | 77.34 | 66.86 | 89.55 | 77.33 | — | 50.1 | 10.41 | 16.4 | †† | — | 0.1 |
| 2 | **MagFormer v13** | **69.42** | 87.61 | 77.43 | — | — | — | — | 50.1 | 10.41 | 16.4 | ~2000 | — | 0.5 |
| 3 | MagFormer (Swin-T) | 62.28 | 82.95 | 69.24 | 68.42 | 87.97 | 77.72 | 92.1 | 79.7 | 2.63 | 29.3 | 467.8 | 3.3 | 2.1 |
| 4 | YOLOv8-seg-X | 61.99 | 83.32 | 70.87 | 40.32 | 78.76 | 38.79 | 85.1 | 71.8 | 0.51 | 27.9 | 23.7 | 1.0 | 42.3 |
| 5 | YOLOv8-seg-L | 61.73 | 83.23 | 70.21 | 40.56 | 78.61 | 39.87 | 84.8 | 45.9 | 0.48 | 26.1 | 24.5 | 0.9 | 40.8 |
| 6 | MGM-Mask2Former | 61.53 | 83.53 | 69.83 | **72.80** | 87.92 | 78.70 | 91.6 | 79.7 | 3.07 | 59.8 | 152.5 | 2.7 | 6.6 |
| 7 | YOLOv8-seg-M | 60.89 | 83.07 | 68.91 | 40.08 | 77.56 | 39.18 | 84.0 | 27.2 | 0.48 | 22.2 | 18.0 | 0.9 | 55.5 |
| 8 | Mask R-CNN | 59.49 | 83.68 | 67.80 | 54.10 | 77.61 | 61.44 | 84.4 | 44.0 | 0.51 | 9.3 | 30.4 | 1.5 | 32.9 |
| 9 | MagFormer-Lite (ConvNext-Lite) | 57.65 | 81.96 | 66.32 | 65.45 | 86.86 | 74.65 | 90.7 | 50.2 | 2.39 | 15.2 | 410.9 | 2.4 | 2.4 |
| 10 | MagFormer-Lite (MV3-SA) | 57.65 | 82.10 | 66.27 | 64.50 | 85.87 | 73.63 | 90.1 | 50.0 | 2.50 | 13.8 | 400.0 | 2.4 | 2.5 |
| 11 | MagFormer-Lite (MV3-SG) | 57.63 | 82.02 | 65.55 | 64.74 | 85.90 | 73.67 | 90.5 | 49.9 | 2.48 | 13.8 | 407.4 | 2.4 | 2.5 |
| 12 | YOLOv8-seg-S | 56.85 | 81.74 | 65.29 | 36.45 | 75.08 | 31.56 | 81.6 | 11.8 | 0.48 | 18.9 | 16.1 | 0.8 | 62.3 |
| 13 | MagFormer (no depth) | 52.58 | 79.84 | 59.79 | 59.52 | 82.66 | 68.03 | 87.3 | 79.6 | 4.41 | 25.9 | 391.2 | 2.5 | 2.6 |
| 14 | YOLOv8-seg-N | 51.32 | 79.03 | 58.38 | 32.24 | 69.14 | 24.83 | 75.7 | 3.3 | 0.47 | 14.4 | 19.2 | 0.8 | 52.1 |
| 15 | Mask2Former | 50.60 | 77.83 | 58.30 | 58.76 | 80.53 | 66.05 | 85.6 | 44.1 | 1.93 | 34.9 | 66.5 | 2.6 | 15.0 |
| 16 | MGM-Mask2Former (no depth) | 41.08 | 74.49 | 41.23 | 39.61 | 68.61 | 40.92 | 75.1 | 79.6 | 2.27 | 29.2 | 81.2 | 2.7 | 12.3 |
| 17 | UOAIS | 31.50 | 70.22 | 22.98 | 17.67 | 50.33 | 5.97 | 60.0 | 81.7 | 1.39 | 24.6 | 46.9 | 1.9 | 21.3 |
| 18 | UNet (Boundary) | 12.63 | 36.63 | 6.29 | 13.60 | 35.06 | 8.71 | 48.9 | 1.9 | 1.84 | 0.6 | 13.6 | 0.6 | 73.6 |
| 19 | UNet++ (Boundary) | 10.05 | 30.01 | 5.11 | 10.88 | 27.82 | 6.88 | 41.7 | 0.5 | 1.72 | 0.7 | 18.9 | 0.7 | 53.0 |
| 20 | UNet (Semantic) | 3.34 | 4.01 | 3.25 | 3.57 | 4.29 | 3.80 | 7.5 | 1.9 | 0.75 | 0.6 | 13.5 | 0.6 | 73.9 |
| 21 | UCN | 2.08 | 7.67 | 0.53 | 1.03 | 4.89 | 0.06 | 21.4 | 42.6 | 2.51 | — | 365.0 | — | 2.7 |
| 22 | MSMFormer | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0 | 50.7 | 3.43 | 75.4 | 341.9 | 3.8 | 2.9 |

---

## Table 2: 512 px Resolution (sorted by bbox AP)

| # | Model | bbox AP | AP50 | AP75 | segm AP | AP50 | AP75 | F1@50 | Params (M) | Train (h) | Train VRAM (GB) | Inf. (ms) | Inf. VRAM (GB) | FPS |
|---|-------|---------|------|------|---------|------|------|-------|------------|-----------|-----------------|-----------|----------------|-----|
| 1 | YOLOv8-seg-X | **63.86** | 83.55 | 72.07 | 42.93 | 80.57 | 44.48 | 86.2 | 71.8 | 0.42 | 7.6 | 33.6 | 2.1 | 29.8 |
| 2 | YOLOv8-seg-L | 63.67 | 84.33 | 71.31 | 42.68 | 80.48 | 44.55 | 86.1 | 45.9 | 0.39 | 6.0 | 30.7 | 2.0 | 32.6 |
| 3 | YOLOv8-seg-M | 62.33 | 83.39 | 70.17 | 42.11 | 79.59 | 43.08 | 85.2 | 27.2 | 0.39 | 4.7 | 29.2 | 1.9 | 34.2 |
| 4 | YOLOv8-seg-S | 59.98 | 82.30 | 67.94 | 40.21 | 77.43 | 40.23 | 83.5 | 11.8 | 0.38 | 3.6 | 17.5 | 1.6 | 57.0 |
| 5 | MagFormer-Lite (MV3-SG) | 58.43 | 80.55 | 63.60 | **60.36** | 83.52 | 68.96 | 87.7 | 49.9 | 4.64 | 8.0 | 402.1 | 2.4 | 2.5 |
| 6 | YOLOv8-seg-N | 54.24 | 80.58 | 61.48 | 35.88 | 72.83 | 32.10 | 79.2 | 3.3 | 0.38 | 3.0 | 16.8 | 0.8 | 59.5 |
| 7 | MagFormer-Lite (MV3-SA) | 55.01 | 79.68 | 59.34 | 59.75 | 82.64 | 68.11 | 87.6 | 50.0 | 2.80 | 8.0 | 397.3 | 2.4 | 2.5 |
| 8 | MagFormer-Lite (ConvNext-Lite) | 54.91 | 78.92 | 59.25 | 59.74 | 82.47 | 67.98 | 86.8 | 50.2 | 2.82 | 8.7 | 400.4 | 2.4 | 2.5 |
| 9 | MGM-Mask2Former | 53.17 | 78.25 | 56.15 | 69.90 | 85.63 | 74.54 | 89.3 | 79.7 | 1.85 | 20.9 | 149.6 | 2.7 | 6.7 |
| 10 | MagFormer (Swin-T) | 49.45 | 74.06 | 51.24 | 59.91 | 82.11 | 66.74 | 86.0 | 79.7 | 6.24 | 15.6 | 455.9 | 3.3 | 2.2 |
| 11 | MagFormer (no depth) | 49.26 | 77.35 | 52.35 | 53.88 | 78.93 | 60.03 | 83.6 | 79.6 | 4.61 | 8.0 | 387.0 | 2.5 | 2.6 |
| 12 | Mask R-CNN | 45.67 | 75.36 | 49.69 | 38.78 | 67.03 | 41.63 | 76.0 | 44.0 | 0.44 | 6.2 | 28.2 | 1.4 | 35.4 |
| 13 | Mask2Former | 40.98 | 74.52 | 42.28 | 43.08 | 72.20 | 47.56 | 79.5 | 44.1 | 1.07 | 10.8 | 48.7 | 2.3 | 20.5 |
| 14 | MGM-Mask2Former (no depth) | 34.88 | 69.83 | 31.38 | 32.22 | 61.49 | 30.88 | 68.2 | 79.6 | 1.37 | 12.9 | 82.3 | 2.7 | 12.2 |
| 15 | UOAIS | 15.43 | 47.45 | 4.85 | 6.16 | 24.77 | 1.14 | 35.3 | 81.7 | 0.67 | 9.5 | 35.9 | 1.7 | 27.9 |
| 16 | UNet++ (Boundary) | 9.49 | 28.71 | 4.58 | 10.18 | 25.82 | 6.69 | 39.8 | 0.5 | 1.38 | 0.7 | 18.8 | 0.7 | 53.2 |
| 17 | MSMFormer | 8.35 | 17.85 | 6.71 | 8.57 | 17.56 | 8.32 | 29.5 | 52.5 | 8.90 | 68.0 | 451.3 | 5.1 | 2.2 |
| 18 | UNet (Boundary) | 4.59 | 14.43 | 2.22 | 5.79 | 15.25 | 3.61 | 27.1 | 1.9 | 1.32 | 0.6 | 13.5 | 0.6 | 73.8 |
| 19 | UNet (Semantic) | 3.15 | 4.03 | 3.28 | 3.25 | 4.20 | 3.28 | 7.5 | 1.9 | 0.62 | 0.6 | 13.5 | 0.6 | 73.9 |
| 20 | UCN | 0.37 | 1.98 | 0.03 | 0.05 | 0.31 | 0.00 | 4.9 | 42.6 | 0.68 | — | 355.7 | — | 2.8 |

---

## Table 3: MagFormer v13 — Detailed Results (1024 px only)

MagFormer v13 is the final optimized version, finetuned from v10's best checkpoint (AP 60.5) with the following improvements:
- 200 object queries (up from 100)
- 9,000 training iterations with cosine LR schedule (lr=5e-5)
- Backbone multiplier 0.5 (Swin-T unfrozen)
- AGPE (Attention-Guided Pyramid Enhancement, reduction=16, kernel=7)
- Mask-piloted training
- Contrastive loss (weight=0.5, temperature=0.07)
- EMA (decay=0.9999)
- MobileNetV3-small depth backbone with SA-Gate fusion

| Eval Method | Checkpoint | bbox AP | AP50 | AP75 | AP_S | AP_M | AP_L | segm AP | AP50 | AP75 |
|---|---|---|---|---|---|---|---|---|---|---|
| Single-scale | iter 8499 | **69.42** | 87.61 | 77.43 | 31.03 | 81.93 | — | — | — | — |
| Single-scale | iter 8999 | 69.05 | 87.60 | 76.65 | 30.79 | 81.93 | — | — | — | — |
| TTA (3-scale + hflip, NMS) | iter 8499 | **70.55** | 88.32 | 77.34 | 32.74 | 83.33 | — | **66.86** | 89.55 | 77.33 |
| TTA (logit-space masks, full COCO, segm-best) | `logit_s075_100_125_hflip_nms055_c050_e075_pre0005_g7` | 66.38 | 87.13 | 75.38 | 25.67 | 79.47 | — | **71.31** | 90.83 | 80.53 |
| TTA (3-scale + hflip, NMS) | iter 8999 | 69.90 | 88.27 | 76.44 | 32.11 | — | — | 66.23 | — | — |
| TTA WBF (iou=0.55, avg) | iter 8999 | 58.82 | — | — | — | — | — | — | — | — |
| TTA WBF (iou=0.55, max) | iter 8499 | 44.62 | — | — | — | — | — | — | — | — |

**Best full COCO TTA run**:
- Run: `logit_s075_100_125_hflip_nms055_c050_e075_pre0005_g7`
- Metrics: bbox AP 66.38, bbox AP50 87.13, bbox AP75 75.38, bbox AP_S 25.67; segm AP 71.31, segm AP50 90.83, segm AP75 80.53, segm AP_S 32.41
- Runtime: 47.6 min
- Params: 50 M, unchanged from MagFormer v13; this is an evaluation-only TTA change
- Config: logit-space mask fusion; scales [0.75, 1.0, 1.25] + horizontal flip; NMS IoU 0.55; cluster mask threshold 0.50; export mask threshold 0.75

**Export mask threshold sweep (logit-space full COCO TTA)**:

| Export mask threshold | Run | bbox AP | bbox AP50 | bbox AP_S | segm AP | segm AP50 | segm AP75 | segm AP_S | Runtime |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.650 | `logit_s075_100_125_hflip_nms055_c050_e065_pre0005_g7` | 68.31 | 87.99 | 28.06 | 70.43 | 90.83 | — | 32.30 | 47.5 min |
| 0.675 | `logit_s075_100_125_hflip_nms055_c050_e0675_pre0005_g4` | 67.83 | 87.37 | 27.49 | 70.66 | 90.83 | 79.65 | 32.34 | 48.9 min |
| 0.700 | `logit_s075_100_125_hflip_nms055_c050_e070_pre0005_g5` | 67.47 | 87.33 | 26.94 | 70.91 | 90.83 | 79.68 | 32.40 | 47.2 min |
| 0.725 | `logit_s075_100_125_hflip_nms055_c050_e0725_pre0005_g6` | 66.94 | 87.27 | 26.32 | 71.07 | 90.83 | 79.69 | 32.35 | 47.8 min |
| 0.750 | `logit_s075_100_125_hflip_nms055_c050_e075_pre0005_g7` | 66.38 | 87.13 | 25.67 | **71.31** | 90.83 | 80.53 | 32.41 | 47.6 min |

**Training details**:
- Params: 50.1 M
- Training time: 10 h 24 min (37,461 s)
- Training VRAM: 16.4 GB per GPU (2 GPUs, DDP)
- Inference speed: ~0.5 img/s (single-scale), ~0.1 img/s (TTA 6 aug)
- TTA config: scales [0.75, 1.0, 1.25] x horizontal flip = 6 augmentations. Earlier bbox-best run used NMS IoU=0.5; latest full COCO run uses logit-space mask fusion and NMS IoU=0.55.

---

## Table 4: MagFormer v13 Efficiency Summary

| Metric | Value |
|---|---|
| Model params | 50.1 M |
| Trainable params | 50.1 M (all) |
| Backbone | SwinTransformer-T (RGB) + MobileNetV3-small (Depth) |
| Fusion | SA-Gate (spatial attention) |
| Training iterations | 9,000 |
| Training time | 10 h 24 min |
| Peak training VRAM | 16.4 GB / GPU (2 GPUs) |
| Inference speed (single) | 0.5 img/s |
| Inference speed (TTA 6x) | 0.1 img/s |
| Best checkpoint | iter 8499 |
| Best bbox AP (single) | 69.42 |
| Best bbox AP (TTA) | **70.55** |
| Best full COCO TTA run | `logit_s075_100_125_hflip_nms055_c050_e075_pre0005_g7` |
| Best full COCO bbox AP (TTA) | 66.38 |
| Best segm AP (TTA) | **71.31** |

---

## Official CellPose / StarDist Baselines

These official baseline runs are listed separately from the main model ranking above.

| Model | Resolution | Params (M) | bbox AP | bbox AP50 | segm AP | segm AP50 | segm AP_S | Runtime |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CellPose | 1024 | 6.6 | 54.9 | 75.2 | 59.0 | 80.5 | 17.0 | 2 h 35 min |
| CellPose | 512 | 6.6 | 54.4 | 76.3 | 57.7 | 79.6 | 12.7 | 2 h 26 min |
| StarDist | 1024 | 1.4 | 34.6 | 68.2 | 48.1 | 77.4 | 7.8 | 4 h 26 min |
| StarDist | 512 | 1.4 | 33.4 | 66.3 | 43.0 | 73.2 | 4.7 | 56 min |

IAUNet is not included in this baseline table yet because the current implementation has not converged and still differs from the paper implementation.

---

## Model Name Mapping

| Short Name | Internal ID | Input Modality | Training |
|---|---|---|---|
| MagFormer v13 | (final optimized version) | RGB + Depth | fine-tuned |
| MagFormer (Swin-T) | magformer_depthnorm_on | RGB + Depth | fine-tuned |
| MagFormer-Lite (ConvNext-Lite) | magformer_lightdepth_convnextlite_spatialgate_edge_validhole | RGB + Depth | fine-tuned |
| MagFormer-Lite (MV3-SG) | magformer_lightdepth_mobilenetv3_spatialgate_edge_validhole | RGB + Depth | fine-tuned |
| MagFormer-Lite (MV3-SA) | magformer_lightdepth_mobilenetv3_sagate_edge_validhole | RGB + Depth | fine-tuned |
| MagFormer (no depth) | magformer_nodpth_ref | RGB only | fine-tuned |
| MGM-Mask2Former | mgm_mask2former_depthnorm_on | RGB + Depth | fine-tuned |
| MGM-Mask2Former (no depth) | mgm_mask2former_nodpth_ref | RGB only | fine-tuned |
| Mask2Former | mask2former | RGB only | fine-tuned |
| Mask R-CNN | maskrcnn | RGB only | fine-tuned |
| YOLOv8-seg-N/S/M/L/X | yolov8_seg_n/s/m/l/x | RGB only | fine-tuned |
| UOAIS | uoais | RGB + Depth | from-scratch |
| MSMFormer | msmformer | RGB + Depth | from-scratch |
| UCN | ucn | RGB + Depth | from-scratch |
| UNet (Boundary) | unet_boundary_inst | RGB only | from-scratch |
| UNet (Semantic) | unet_semantic_inst | RGB only | from-scratch |
| UNet++ (Boundary) | unetpp_boundary_inst | RGB only | from-scratch |

---

## Key Findings

1. **MagFormer v13 achieves the highest bbox AP at 70.55 (TTA)**, surpassing all baselines including MGM-Mask2Former (72.80 segm AP but only 61.53 bbox AP).

2. **MagFormer v13 reaches 71.31 segm AP with logit-space full COCO TTA**, improving the previous MagFormer TTA segm AP 66.86 by +4.45 AP without changing model parameters.

3. **MGM-Mask2Former still leads in historical segm AP** (72.80 at 1024px, 69.90 at 512px) but lags in bbox AP due to mask-to-bbox conversion.

4. **YOLOv8 variants dominate bbox AP at 512px** (63.86 for X) due to fast inference and strong bbox localization, but segm AP is much lower (~40-43).

5. **Depth helps significantly**: models with depth input consistently outperform their no-depth counterparts (e.g., MGM-Mask2Former 72.80 vs 39.61 segm AP at 1024px).

5. **From-scratch models perform poorly** on this small dataset (1,566 images): UOAIS, UCN, MSMFormer, and UNet variants all achieve <20 AP.

6. **WBF fusion hurts for dense PCB scenes**: coordinate averaging merges nearby small objects and shifts boxes. NMS is clearly better (70.55 vs 44-58 AP).

---

## Notes

- "—" indicates the metric was not measured for that model/configuration
- AP_S / AP_M / AP_L: AP for small/medium/large objects (COCO definition)
- F1@50: F1 score at IoU threshold 0.5
- Train VRAM: peak GPU memory during training (per GPU)
- Inf. VRAM: peak GPU memory during single-image inference
- Inf. (ms): mean inference latency per image (lower is better)
- FPS: frames per second (higher is better)
- TTA: test-time augmentation with multi-scale [0.75, 1.0, 1.25] x horizontal flip
- All models evaluated on the full 1,566-image dataset
- Baseline results collected April 2026; MagFormer v13 results from May 2026
