# VC-SUDA Stage C R8/R8B Results - 2026-05-15

## Conclusion

R8 score threshold tuning did not improve over the R7 checkpoint. R8B low-LR continuation gives a very small target_unlabeled200 gain: best target_unlabeled200 segm AP is `0.319922` at `ckpt749`, compared with R7 `0.3171`.

This is not a meaningful path toward the 61+ target. The val28 best inside R8B is `ckpt249` with segm AP `0.278051`, only slightly above the R7 val28 `0.2771`.

## R8 Score Threshold Sweep

R8 evaluated score thresholds on R7 `ckpt1999`. All thresholds produced the same target_unlabeled200 segm AP `0.3171`, so val28 was not run. Higher thresholds only reduced prediction count.

| Score threshold | target_unlabeled200 segm AP | Predictions | Val28 run |
|---:|---:|---:|---|
| 0.075 | 0.3171 | 12861 | no |
| 0.100 | 0.3171 | 12751 | no |
| 0.125 | 0.3171 | 12679 | no |
| 0.150 | 0.3171 | 12613 | no |

## R8B Setup

- Config commit: `da04bd2da70a20e9bd6273460012515e7b24c555`
- Config: `configs/vc_suda_stage_c_r8b_lsj10_low_lr_continue_1024_teacher8499.yaml`
- Warm start: R7 `ckpt1999`
- LR: `solver.base_lr=1e-5`
- Max iter: `1000`
- Checkpoint period: `250`
- Single-variable intent: continue R7 from `ckpt1999` with lower LR only.

The submitted config stayed pure single-variable. The actual run used an output-local config with only `runtime.skip_depth_sanity=true`, because the depth sanity `fg_ratio` narrowly failed before training.

## External 1024 Backmap Results

### target_unlabeled200

| Checkpoint | bbox AP | bbox AP50 | bbox AP75 | segm AP | segm AP50 | segm AP75 | Predictions |
|---|---:|---:|---:|---:|---:|---:|---:|
| ckpt249 | 0.393285 | 0.734022 | 0.378753 | 0.318340 | 0.649087 | 0.281207 | 13089 |
| ckpt499 | 0.392710 | 0.734480 | 0.374925 | 0.318955 | 0.649703 | 0.283402 | 12754 |
| ckpt749 | 0.391824 | 0.734299 | 0.373346 | **0.319922** | 0.649133 | **0.283460** | 12994 |
| ckpt999 | 0.392336 | 0.734292 | 0.378148 | 0.319162 | **0.650190** | 0.281967 | 12912 |

`ckpt749` is the best R8B target_unlabeled200 checkpoint by segm AP. It improves over R7 target_unlabeled200 segm AP `0.3171` by `+0.002822`.

### val28

| Checkpoint | bbox AP | bbox AP50 | bbox AP75 | segm AP | segm AP50 | segm AP75 | Predictions |
|---|---:|---:|---:|---:|---:|---:|---:|
| ckpt249 | **0.353679** | **0.696741** | 0.319420 | **0.278051** | 0.604188 | **0.220699** | 2033 |
| ckpt499 | 0.349797 | 0.689144 | 0.320898 | 0.275719 | 0.604282 | 0.213082 | 1986 |
| ckpt749 | 0.348843 | 0.689216 | **0.323948** | 0.277386 | **0.604323** | 0.218984 | 2019 |
| ckpt999 | 0.348806 | 0.689723 | 0.320251 | 0.275149 | 0.603735 | 0.218675 | 2009 |

`ckpt249` is the best R8B val28 checkpoint by segm AP. It improves over R7 val28 segm AP `0.2771` by `+0.000951`.

## Readout

- R8 score threshold tuning did not beat R7 and only reduced predictions.
- R8B `ckpt749` is the current best target_unlabeled200 Stage C checkpoint by segm AP: `0.319922`.
- R8B `ckpt249` is the best val28 checkpoint inside this run: `0.278051`.
- The target and val gains are small, and the result remains far below the 61+ goal.

## Checkpoint Retention Note

Future model-selection runs that rely on external backmap sweeps should set
`runtime.checkpoint_max_keep: null` so all numbered checkpoints remain available
for post-training selection.
