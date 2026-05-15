# VC-SUDA Stage C R9B Threshold and Retention Findings - 2026-05-15

## Conclusion

R9B mask threshold tuning is not useful from the available `ckpt999`. None of the tested mask thresholds beat the `ckpt999` baseline target_unlabeled200 segm AP `0.319162`, and none beat the current global best `0.319922` from R8B `ckpt749`.

The planned R9 `ckpt749` mask sweep remains blocked because `checkpoint_iter_0000749.pth` was already removed by the old hardcoded checkpoint retention rule.

## Retention Root Cause and Fix

The missing `ckpt749` was caused by hardcoded retention in both trainers:

- `magformer/engine/trainer.py` called `_cleanup_old_checkpoints(max_keep=2)`.
- `magformer/engine/vc_suda_trainer.py` called `_cleanup_old_checkpoints(max_keep=2)`.

The fix landed in commit `cc134100cd9d8f23db33463c6f596045eef2eb16`:

- Added `runtime.checkpoint_max_keep`.
- `null` or `0` keeps all numbered checkpoints.
- Default remains `2`, so existing behavior is unchanged unless configured.
- Validation at fix time: `16` checkpoint-retention tests passed, and ruff passed on touched files.

Future model-selection runs that need post-training external sweeps should set:

```yaml
runtime:
  checkpoint_max_keep: null
```

## R9B Available Checkpoint

R9B could only sweep the available `ckpt999`, not `ckpt749`.

Baseline external 1024 backmap metrics:

| Split | Checkpoint | bbox AP | segm AP |
|---|---|---:|---:|
| target_unlabeled200 | ckpt999 baseline | 0.392336 | 0.319162 |
| val28 | ckpt999 baseline | 0.348806 | 0.275149 |

No val28 reruns were done for the mask threshold sweep because target_unlabeled200 did not improve.

## Mask Threshold Sweep on target_unlabeled200

| Mask threshold | bbox AP | segm AP | Beats ckpt999 baseline | Beats global best 0.319922 |
|---:|---:|---:|---|---|
| 0.35 | 0.382159 | 0.301310 | no | no |
| 0.40 | 0.382159 | 0.301310 | no | no |
| 0.45 | 0.382159 | 0.301310 | no | no |
| 0.55 | 0.392336 | 0.319162 | ties | no |
| 0.60 | 0.392336 | 0.319162 | ties | no |
| 0.65 | 0.392336 | 0.319162 | ties | no |

Thresholds `0.35` through `0.45` reduce both bbox and mask AP. Thresholds `0.55` through `0.65` reproduce the `ckpt999` baseline exactly.

## Readout

- R9B mask threshold tuning from `ckpt999` does not improve target_unlabeled200.
- The R8B `ckpt749` target_unlabeled200 segm AP `0.319922` remains the global best recorded Stage C result.
- The original R9 `ckpt749` mask sweep cannot be recovered from this run because that checkpoint was deleted before retention became configurable.
- The retention fix prevents this specific loss mode for future runs when `runtime.checkpoint_max_keep: null` is set.
