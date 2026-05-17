# VC-SUDA R79 no-train pseudo signal diagnosis, 2026-05-17

R79 says the R74 `target_unlabeled` branch is active, but its weighted signal is almost zero.

The scorer is not broadly filtering dense images. Dense keep-rate is high in the 32-image no-train probe. Tiny and bottom-20 area objects are weaker: candidate coverage is already low, and the 0.1 scorer threshold removes part of that weak tiny signal. The main blocker is still the unsupervised schedule, because the weighted pseudo loss is only `0.005%` to `0.018%` of logged total loss during the R74 resume window.

## Scope

- Host: `4029`.
- Project: `/home/hdd3/zhanghaonan/magformer`.
- Conda env: `/home/hdd3/zhanghaonan/anaconda3/envs/magformer`.
- Config: `configs/vc_suda_stage_c_r74_balanced_ce_minfg005_1024.yaml`.
- Checkpoint: `output/vc_suda/stage_c_r74_balanced_ce_minfg005_1024/checkpoint_iter_0000750.pth`.
- R74 metrics log: `output/vc_suda/stage_c_r74_balanced_ce_minfg005_1024/metrics_log.csv`.
- R74 TensorBoard event file exists at `output/vc_suda/stage_c_r74_balanced_ce_minfg005_1024/logs/tensorboard/events.out.tfevents.1778971958.WS-4029GP-TRT.952657.0`, but it has no scalar tags in `EventAccumulator`; the CSV/JSONL logger output is the usable scalar source.
- R78/R52 bucket stats: `output/diagnostics/r52_target_unlabeled_sampling_20260516/target_sampling_stats.json`.
- Target GT proxy: `magformer_datasets/pseudo_real_512/annotations/instances_target_unlabeled.json`.

No training was started. The only GPU run was teacher/scorer inference with `CUDA_VISIBLE_DEVICES=4`.

## R74 Training Signal From Logs

R74 resumed at iter `500` and logged 13 train scalar rows through iter `740`. The trainer logs `pseudo_total` before multiplying by `vc_suda.unsupervised_weight`, so R79 recomputed the weighted pseudo loss from the schedule:

`unsup_weight = 0.02 * (((iter + 1) / 252) / 10)^2` during warmup.

| segment | iter rows | mean unsup weight | source_total_loss | tl_total_loss_weighted | pseudo_total raw | pseudo weighted | weighted pseudo / total | pseudo kept | pseudo keep-rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| early | 500-580, n=5 | `0.000924` | `15.1947` | `8.0717` | `1.1041` | `0.001026` | `0.0051%` | `26.2` | `0.262` |
| mid | 600-680, n=5 | `0.001297` | `15.5459` | `18.9926` | `1.2707` | `0.001637` | `0.0050%` | `31.0` | `0.310` |
| late | 700-740, n=3 | `0.001638` | `4.5599` | `8.6520` | `1.4261` | `0.002314` | `0.0178%` | `44.0` | `0.440` |
| overall | 500-740, n=13 | `0.001232` | `12.8756` | `12.4059` | `1.2425` | `0.001558` | `0.0080%` | `32.2` | `0.322` |

This confirms the branch is not dead. It produces pseudo labels and non-zero raw pseudo loss. It also confirms the effective gradient weight is tiny.

## 32-Image Teacher/Scorer Probe

Command output:

- Summary: `output/diagnostics/r79_pseudo_signal_diagnosis_20260517/pseudo_summary_32.json`.
- Buckets: `output/diagnostics/r79_pseudo_signal_diagnosis_20260517/pseudo_buckets_32.json`.
- Command log: `output/diagnostics/r79_pseudo_signal_diagnosis_20260517/pseudo_32_command.log`.

Sampled image ids, in loader order:

```text
[8, 6, 227, 109, 150, 159, 172, 162, 201, 171, 21, 215, 209, 219, 195, 74, 38, 137, 127, 15, 135, 194, 128, 41, 20, 60, 50, 23, 99, 40, 12, 159]
```

This is 32 loader draws and 31 unique image ids. Image `159` appears twice because the Stage C dataset length follows source coverage while target ids are indexed modulo the 200-image target pool.

Overall scorer result at threshold `0.1`:

| images | candidates | kept | keep-rate | empty images | mean quality | median quality | p75 quality | p95 quality |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 3200 | 1264 | `0.395` | 0 | `0.0872` | `0.0558` | `0.1686` | `0.2500` |

Threshold sweep:

| threshold | kept | keep-rate | empty images |
|---:|---:|---:|---:|
| 0.05 | 1650 | `0.516` | 0 |
| 0.10 | 1264 | `0.395` | 0 |
| 0.15 | 907 | `0.283` | 0 |
| 0.20 | 599 | `0.187` | 0 |
| 0.25 | 158 | `0.049` | 1 |
| 0.30 | 0 | `0.000` | 32 |

## Bucket Table

For `normal`, `dense`, and `dense_tiny`, candidate and kept counts are all pseudo predictions in sampled images from that R78/R52 image bucket.

For `tiny_area_le_256` and `bottom20_area`, candidate and kept counts are predictions whose pseudo bbox has IoU `>=0.50` to a GT object in that area bucket. The coverage columns are a bbox-IoU proxy against target GT. Training does not receive these GT labels.

| bucket | images | GT/proxy count | candidates | kept | keep-rate | candidate mean q | kept mean q | candidate cov@50/@75 | kept cov@50/@75 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| normal | 26 | 1174 | 2600 | 796 | `0.306` | `0.0698` | `0.1879` | `0.510 / 0.253` | `0.295 / 0.160` |
| dense | 5 | 497 | 500 | 384 | `0.768` | `0.1583` | `0.1902` | `0.392 / 0.115` | `0.316 / 0.101` |
| dense_tiny | 1 | 99 | 100 | 84 | `0.840` | `0.1820` | `0.2043` | `0.606 / 0.303` | `0.566 / 0.273` |
| tiny_area_le_256 | 28 | 410 | 140 | 75 | `0.536` | `0.1062` | `0.1723` | `0.283 / 0.041` | `0.183 / 0.027` |
| bottom20_area | 27 | 313 | 92 | 53 | `0.576` | `0.1117` | `0.1715` | `0.240 / 0.035` | `0.166 / 0.026` |

## Answer

R74 target_unlabeled is active, but its weighted loss is too small to matter.

Dense is not being filtered out by the scorer in this probe. Dense keep-rate is `0.768`, and dense_tiny keep-rate is `0.840`. The dense failure seen in R78 is more consistent with localization/separation quality than with scorer removal.

Tiny is partly filtered, but it is not the whole story. Tiny candidate coverage is already low before thresholding: candidate cov@50 is `0.283` for `area<=256` and `0.240` for bottom20. After the scorer threshold, kept cov@50 falls to `0.183` and `0.166`. So tiny has both weak candidate coverage and threshold loss.

## Decision

Recommend first changing the unsupervised schedule.

Do not start with R74+32K source. Do not start with pseudo scorer changes as the next main run. The evidence says the branch exists, the scorer keeps many dense labels, and the raw pseudo loss is non-zero, but the schedule scales it to about `0.008%` of total loss on average. A schedule-only no-long-train smoke should come before changing scorer/objective details.

## Validation

- `py_compile` passed for `tools/diagnose_vc_suda_pseudo_labels.py`.
- 2-image GPU4 smoke passed and wrote `pseudo_smoke_summary.json` plus `pseudo_smoke_buckets.json`.
- 32-image GPU4 diagnosis passed and wrote finite JSON outputs.
- Main command log scan: no `Traceback`, no `CUDA out of memory`, no `non-finite`.
- No training command was run.
