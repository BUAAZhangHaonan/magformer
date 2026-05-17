# VC-SUDA R80 unsupervised schedule, 2026-05-17

R80 tests one variable: a stronger target_unlabeled pseudo-label loss schedule on top of R74.

It is a pass for the narrow schedule question, but not a meaningful new best. The weighted pseudo signal became visible without dominating training, and target AP moved from `0.336010` to `0.336383` segm AP. The gain is only `+0.000373`, and R78 hard buckets did not improve.

## Scope

- Host: `4029`.
- Project: `/home/hdd3/zhanghaonan/magformer`.
- Conda env: `/home/hdd3/zhanghaonan/anaconda3/envs/magformer`.
- Base config: `configs/vc_suda_stage_c_r74_balanced_ce_minfg005_1024.yaml`.
- R80 config: `configs/vc_suda_stage_c_r80_unsup_schedule_1024.yaml`.
- Output dir: `output/vc_suda/stage_c_r80_unsup_schedule_1024`.
- Checkpoint: `output/vc_suda/stage_c_r80_unsup_schedule_1024/checkpoint_iter_0000750.pth`.
- Local config commit: `c4810a31df86b7600ff2e713f0129e20a8ae9744`.
- GitHub connector config commit: `db3784fdc3efeafa56c53993aa1eebc13cd02d10`.

GitHub SSH from `4029` timed out on `ssh.github.com:443`, so the config was pushed through the GitHub connector. The local repo still has the local config commit.

## Config Diff

Only identity and unsupervised schedule changed:

- `name`: `vc_suda_stage_c_r74_balanced_ce_minfg005_1024` -> `vc_suda_stage_c_r80_unsup_schedule_1024`.
- `runtime.output_dir`: `output/vc_suda/stage_c_r80_unsup_schedule_1024`.
- `runtime.logger.log_dir`: `output/vc_suda/stage_c_r80_unsup_schedule_1024/logs`.
- `runtime.logger.run_name`: `vc_suda_stage_c_r80_unsup_schedule_1024`.
- `vc_suda.unsupervised_weight`: `0.02` -> `0.5`.
- `vc_suda.unsupervised_warmup_epochs`: `10` -> `1`.

No source split, scorer, target sampling, model, or training logic changed. Source stayed `magformer_datasets/pseudo_real_512`.

## Validation Before Training

- `tools/verify_vc_suda_stage.py --config configs/vc_suda_stage_c_r80_unsup_schedule_1024.yaml --stage C --skip-batch` passed.
- Config assertions passed for identity fields, `unsupervised_weight=0.5`, `unsupervised_warmup_epochs=1`, `solver.max_iter=750`, `runtime.gpus=[4,5,6,7]`, and `source_root=magformer_datasets/pseudo_real_512`.
- No smoke was needed because schema and static preflight passed.

## Training

Training ran in tmux session `r80_unsup_schedule_20260517` with:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
python -m torch.distributed.run --standalone --nproc_per_node=4 tools/train.py \
  --config configs/vc_suda_stage_c_r80_unsup_schedule_1024.yaml \
  --output-dir output/vc_suda/stage_c_r80_unsup_schedule_1024 \
  --gpus 0,1,2,3 \
  --num-workers 2
```

The first direct non-torchrun launch failed before training with the expected DDP preflight error. The formal run used torchrun, resumed R12 `ckpt499`, and exited `0`.

Training health:

| segment | logged rows | mean loss | mean pseudo raw | mean pseudo weighted | mean weighted pseudo / total | max weighted pseudo / total | mean kept | mean keep-rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| first100 | 6 | `27.3506` | `1.1314` | `0.5657` | `2.4805%` | `4.7481%` | `27.3` | `0.273` |
| all logged | 13 | `25.9749` | `1.2186` | `0.6093` | `2.9222%` | `7.4845%` | `31.8` | `0.318` |
| late | 3 | `13.7965` | `1.3864` | `0.6932` | `5.1049%` | `7.4845%` | `44.7` | `0.447` |

No `Traceback`, OOM, or non-finite signal appeared in the formal training log. RAM stayed far below 90%. GPU memory was high but did not OOM.

## External Eval

Formal target eval:

- Output: `output/diagnostics/r80_unsup_schedule_target_unlabeled200_20260517`.
- Protocol: `pseudo_real_target_unlabeled200`, 200 images, 1024 backmap, `bbox,segm`, `inference_topk=200`, `max_dets=200`.
- Protocol checker: pass.

Formal source sanity:

- Output: `output/diagnostics/r80_unsup_schedule_original_first50_20260517`.
- Protocol: original 1.5K `first50`, 1024 backmap, `bbox,segm`, `inference_topk=100`, `max_dets=100`.
- Protocol checker: pass with `--allow-nondefault-weights`.

| eval | metric | R74 | R80 | delta |
|---|---|---:|---:|---:|
| target_unlabeled200 | bbox AP | `0.387567` | `0.389937` | `+0.002370` |
| target_unlabeled200 | bbox AP75 | `0.373927` | `0.374118` | `+0.000191` |
| target_unlabeled200 | segm AP | `0.336010` | `0.336383` | `+0.000373` |
| target_unlabeled200 | segm AP75 | `0.315963` | `0.317290` | `+0.001327` |
| original first50 | bbox AP | `0.471960` | `0.474794` | `+0.002834` |
| original first50 | segm AP | `0.437474` | `0.439126` | `+0.001652` |

## Diagnostics

R78 bucket diagnosis on R80 target predictions:

| bucket | images | annotations | bbox AP/AP50/AP75 | segm AP/AP50/AP75 | mask oracle R@50/R@75 |
|---|---:|---:|---:|---:|---:|
| overall | 200 | 11750 | `0.389937 / 0.729336 / 0.374118` | `0.336383 / 0.655934 / 0.317290` | `0.690553 / 0.373447` |
| normal | 158 | 7634 | `0.463505 / 0.816762 / 0.473654` | `0.412351 / 0.756361 / 0.414225` | `0.786089 / 0.473146` |
| dense | 30 | 2971 | `0.236382 / 0.552216 / 0.173465` | `0.173411 / 0.434174 / 0.115247` | `0.500841 / 0.177381` |
| dense_tiny | 12 | 1145 | `0.264830 / 0.591118 / 0.198906` | `0.213297 / 0.490444 / 0.158565` | `0.545852 / 0.217467` |
| tiny_area_le_256 | 185 | 2934 | `0.080563 / 0.328419 / 0.011653` | `0.010156 / 0.054272 / 0.000354` | `0.183708 / 0.016360` |
| bottom20_area | 181 | 2350 | `0.062073 / 0.273175 / 0.005934` | `0.003265 / 0.019235 / 0.000079` | `0.122979 / 0.007234` |

Oracle:

- `oracle_r75`: `0.373447`.
- `matched_oracle_score_segm_ap`: `0.377228`.

Dense geometry:

- predictions: `14366`.
- mask TP/FP/FN: `8114/6252/3636`.
- bbox TP/FP/FN: `8979/5387/2771`.
- dense `90-100` bucket mask TP/FP/FN: `2368/2816/2285`.

## Decision

R80 shows the schedule was indeed too weak, because the weighted pseudo signal rose from R74's `0.0080%` average to about `2.9%` of total logged loss without destabilizing training.

But target AP barely moved. The hard-bucket diagnosis is still the same shape as R78: normal is strong, while dense, dense_tiny, tiny, and bottom20 remain weak. This says schedule strength alone is not enough to solve the target bottleneck.

## Artifacts

- Train log: `output/logs/r80_unsup_schedule_train_20260517.log`.
- Train resource monitor: `output/vc_suda/stage_c_r80_unsup_schedule_1024/logs/resource_monitor.csv`.
- Target metrics: `output/diagnostics/r80_unsup_schedule_target_unlabeled200_20260517/metrics.cocoeval.json`.
- Source metrics: `output/diagnostics/r80_unsup_schedule_original_first50_20260517/metrics.cocoeval.json`.
- Bucket AP: `output/diagnostics/r80_unsup_schedule_target_unlabeled200_20260517/r80_bucket_ap.md`.
- Dense audit: `output/diagnostics/r80_unsup_schedule_target_unlabeled200_20260517/r80_unsup_schedule_target_unlabeled200_audit.md`.
- Oracle: `output/diagnostics/r80_unsup_schedule_target_unlabeled200_20260517/oracle/r53_prediction_union_oracle_summary.json`.
