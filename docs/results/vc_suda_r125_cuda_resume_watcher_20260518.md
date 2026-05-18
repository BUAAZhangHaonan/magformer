# VC-SUDA R125 CUDA Resume Watcher - 2026-05-18

## Conclusion

R125 is a CUDA resume watcher for the 4029 host state. It avoids repeated manual probes while CUDA/NVML is blocked, and it does not consume GPU before recovery because the probe fails before any training launch.

It only launches the R121 smoke after the CUDA probe succeeds. It does not start R122 automatically.

## Watcher Record

- Watcher session: `r125_cuda_resume_r121_watcher`
- Watcher log: `output/diagnostics/r125_cuda_resume_r121_watcher_20260518.log`
- Probe interval: `300s`
- Probe target: GPU4
- Probe Python: `/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python`
- Initial probe status: failed
- Initial probe error: `CUDA unknown error`
- Initial probe exit code: `1`

## Launch Behavior

When the probe succeeds, the watcher launches only the R121 smoke with:

```text
configs/baseline_vc_suda_r121_depth_boundary_w001_smoke_b1_iter1_loss_only_fg00075.yaml
```

The watcher does not start R122 automatically. R122 still needs a manual go/no-go after the R121 smoke result is checked.

## View Watcher

```bash
tmux ls
tmux attach -t r125_cuda_resume_r121_watcher
tail -f output/diagnostics/r125_cuda_resume_r121_watcher_20260518.log
```

## Stop Watcher

```bash
tmux kill-session -t r125_cuda_resume_r121_watcher
```

## Why This Exists

4029 is blocked by the CUDA/NVML host state. The watcher records the recovery probe loop so a human does not need to rerun the same GPU4 check every few minutes.

The watcher waits on the probe first, so it avoids occupying GPU before CUDA recovery. After recovery, it runs the smallest R121 smoke gate before any R122 decision.
