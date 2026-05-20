# VC-SUDA Baseline Comparison Results - 2026-05-20

This note records the current horizontal baseline pass and the repaired R142-DPE external bbox+segm eval. All AP values below are AP points. The source metrics were cross-read from JSON or Detectron2 summary logs listed in each section.

## Scope

- Branch: `feature/vc-suda-sim2real`.
- New explicit split entry commit: `3c15f814` (`baseline: support explicit detectron2 ecc splits`).
- Main repo path: `/home/hdd3/zhanghaonan/magformer`.
- No `.worktree` path was used for this record.

## R142-DPE 512 External Eval

These evals use external bbox+segm evaluation on val28 and remaining75.

| checkpoint | split | bbox AP | bbox AP50 | bbox AP75 | segm AP | segm AP50 | segm AP75 | output path |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| iter0499 | val28 | 33.4507 | 70.7601 | 28.6234 | 26.2098 | 60.1258 | 20.8724 | `output/diagnostics/r142_dpe_512_iter0499_val28_bbox_segm_eval` |
| iter0499 | remaining75 | 39.4305 | 76.3760 | 36.9088 | 31.4338 | 65.3984 | 27.0737 | `output/diagnostics/r142_dpe_512_iter0499_remaining75_bbox_segm_eval` |
| iter0750 | val28 | 33.6690 | 70.6517 | 28.2509 | 26.2093 | 59.8806 | 20.3883 | `output/diagnostics/r142_dpe_512_iter0750_val28_bbox_segm_eval` |
| iter0750 | remaining75 | 39.4389 | 76.1078 | 36.8607 | 31.6131 | 65.9529 | 27.3008 | `output/diagnostics/r142_dpe_512_iter0750_remaining75_bbox_segm_eval` |

Metric sources:

- `output/diagnostics/r142_dpe_512_iter0499_val28_bbox_segm_eval/metrics.cocoeval.json`
- `output/diagnostics/r142_dpe_512_iter0499_remaining75_bbox_segm_eval/metrics.cocoeval.json`
- `output/diagnostics/r142_dpe_512_iter0750_val28_bbox_segm_eval/metrics.cocoeval.json`
- `output/diagnostics/r142_dpe_512_iter0750_remaining75_bbox_segm_eval/metrics.cocoeval.json`

## Mask R-CNN R144 Target150 Baseline

The new Detectron2 ECC entry at commit `3c15f814` supports explicit split registration. That entry was used for the R144 supervised baseline chain.

The 50-iter smoke at LR `0.00125` passed with exit code `0`.

- Smoke output: `output/baseline/r143_maskrcnn_target150_512_smoke_lr0p00125_20260520_173800`
- Smoke val28 eval at iter50: bbox AP/AP50/AP75 `17.5942 / 49.9775 / 7.0727`; segm AP/AP50/AP75 `11.7685 / 35.6003 / 4.6132`

Full 2000-iter R144 val28 result:

| split | checkpoint | bbox AP | bbox AP50 | bbox AP75 | segm AP | segm AP50 | segm AP75 | output path |
|---|---|---:|---:|---:|---:|---:|---:|---|
| val28 | iter2000 / model_final | 32.0621 | 65.8939 | 27.5004 | 24.4749 | 53.8199 | 18.9126 | `output/baseline/r144_maskrcnn_target150_512_full_lr0p00125_20260520_174211` |

Remaining75 external eval:

| split | checkpoint | bbox AP | bbox AP50 | bbox AP75 | segm AP | segm AP50 | segm AP75 | output path |
|---|---|---:|---:|---:|---:|---:|---:|---|
| remaining75 | iter2000 / model_final | 37.8825 | 71.3712 | 36.2512 | 30.0783 | 61.3787 | 26.2019 | `output/diagnostics/r144_maskrcnn_target150_512_remaining75_eval_20260520_175055` |

Metric sources:

- Val28: `output/baseline/r144_maskrcnn_target150_512_full_lr0p00125_20260520_174211/metrics.json`
- Remaining75: `output/diagnostics/r144_maskrcnn_target150_512_remaining75_eval_20260520_175055/remaining75_horizontal_summary.json`
- Remaining75 log cross-check: `output/diagnostics/r144_maskrcnn_target150_512_remaining75_eval_20260520_175055/log.txt`

## Horizontal Comparison

| method | val28 bbox AP | val28 segm AP | remaining75 bbox AP | remaining75 segm AP | val28 source | remaining75 source |
|---|---:|---:|---:|---:|---|---|
| MagFormer R114 | 35.6494 | 32.1831 | 43.0296 | 39.2173 | `output/diagnostics/r114_magformer_r113warm_target150_iter2000_val28_1024_backmap_topk200_20260518/metrics.cocoeval.json` | `output/diagnostics/r114_magformer_r113warm_target150_iter2000_remaining75_1024_backmap_topk200_20260518/metrics.cocoeval.json` |
| MagFormer R139 RGB-only | 36.2034 | 32.2550 | 42.9653 | 38.8223 | `output/diagnostics/r139_rgb_only_val28_1024_backmap_topk200_20260519/metrics.cocoeval.json` | `output/diagnostics/r139_rgb_only_remaining75_1024_backmap_topk200_20260519/metrics.cocoeval.json` |
| MagFormer R141 RGB-D | 36.0386 | 31.9539 | 43.1195 | 38.8543 | `output/diagnostics/r141_rgbd_val28_1024_backmap_topk200_20260519/metrics.cocoeval.json` | `output/diagnostics/r141_rgbd_remaining75_1024_backmap_topk200_20260519/metrics.cocoeval.json` |
| Mask2Former R137 | 27.1572 | 24.9466 | 34.2840 | 32.2779 | `output/diagnostics/r137_official_m2f_val28_fixed1024_eval/log.txt` | `output/diagnostics/r137_official_m2f_remaining75_fixed1024_eval/log.txt` |
| Mask R-CNN R144 | 32.0621 | 24.4749 | 37.8825 | 30.0783 | `output/baseline/r144_maskrcnn_target150_512_full_lr0p00125_20260520_174211/metrics.json` | `output/diagnostics/r144_maskrcnn_target150_512_remaining75_eval_20260520_175055/remaining75_horizontal_summary.json` |
| VC-SUDA R142-DPE 512 iter0499 | 33.4507 | 26.2098 | 39.4305 | 31.4338 | `output/diagnostics/r142_dpe_512_iter0499_val28_bbox_segm_eval/metrics.cocoeval.json` | `output/diagnostics/r142_dpe_512_iter0499_remaining75_bbox_segm_eval/metrics.cocoeval.json` |
| VC-SUDA R142-DPE 512 iter0750 | 33.6690 | 26.2093 | 39.4389 | 31.6131 | `output/diagnostics/r142_dpe_512_iter0750_val28_bbox_segm_eval/metrics.cocoeval.json` | `output/diagnostics/r142_dpe_512_iter0750_remaining75_bbox_segm_eval/metrics.cocoeval.json` |

## Conclusion

The repaired R142-DPE 512 mask result does not exceed the MagFormer labeled-only baselines. On remaining75, R142-DPE reaches `31.6131` segm AP at iter0750, while MagFormer R114 reaches `39.2173`, R139 reaches `38.8223`, and R141 reaches `38.8543`.

R142-DPE is also below the Mask2Former R137 remaining75 result. R142-DPE iter0750 reaches `31.6131` segm AP, while Mask2Former R137 reaches `32.2779`.

So the current VC-SUDA result does not prove an advantage over strong supervised baselines. The useful result from this pass is that the engineering chain and baseline protocol are now filled in: explicit split support exists, Mask R-CNN has a target150 supervised baseline, and R142-DPE has comparable external bbox+segm evals on val28 and remaining75.

## Next Steps

- Run pseudo quality, threshold, and unsupervised budget ablations before any more long training.
- Do not keep extending VC-SUDA runs blindly without showing that the pseudo signal and budget are healthy.
- To test generality, adapt VC-SUDA to Mask2Former next. Keep Mask R-CNN as a supervised baseline first, not the main adaptation proof point.
