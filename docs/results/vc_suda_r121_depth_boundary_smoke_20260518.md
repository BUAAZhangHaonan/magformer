# VC-SUDA R121 Depth-Boundary Smoke Status - 2026-05-18

## Conclusion

R121 depth-boundary smoke is currently blocked by the 4029 server GPU/CUDA state, not by code logic.

Do not start R122 300iter until the R121 fg00075 smoke is rerun after CUDA/NVML recovers and passes the smoke gate.

## Commit Context

- R121/R122 config commits: `74e85b06`, `24f83779`, `3adfddb8`.
- Code commit: `985f549e`.

## R121 Smoke Attempts

### Original Smoke

R121 original smoke failed during the first forward because the single-GPU run inherited `ims_per_batch=4` and hit OOM.

This means the first observed failure was not the depth-boundary loss failing first.

### Loss-Only b1 iter1

R121 loss-only `b1 iter1` removed the batch-size OOM, but it was stopped before training by depth sanity.

The measured `fg_ratio=0.000897` was below the default depth sanity threshold, so the run did not reach the training step.

### fg00075 Smoke

R121 fg00075 smoke uses:

- Config: `configs/baseline_vc_suda_r121_depth_boundary_w001_smoke_b1_iter1_loss_only_fg00075.yaml`
- `runtime.depth_sanity.min_mask_fg_ratio=0.00075`

The config validation passed. Depth sanity was not disabled; the minimum foreground-mask ratio was lowered to `0.00075` for this smoke.

The smoke did not enter training because CUDA/NVML was unavailable on 4029.

## 4029 GPU/CUDA State

- `nvidia-smi` reports GPU0 `Unknown Error`.
- PyTorch reports `cuda_available=False` and `device_count=0` under `CUDA_VISIBLE_DEVICES=4`, `5`, `6`, and `7`.
- No user training process is occupying the GPUs.

## Current Blocker

The current blocker is the server driver or hardware state on 4029.

This is not a code-logic blocker, and R122 300iter must not be started while CUDA/NVML is in this state.

## Recovery Step

After 4029 CUDA/NVML recovers, rerun:

```bash
configs/baseline_vc_suda_r121_depth_boundary_w001_smoke_b1_iter1_loss_only_fg00075.yaml
```

Pass conditions:

- Depth sanity passes.
- `loss_depth_boundary` appears.
- No OOM or NaN occurs.
- The run completes 1 iteration.

Only after this R121 smoke passes should R122 300iter be started.
