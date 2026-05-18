# R126 CUDA Resume Watcher G4567

Date: 2026-05-18

## Why R125 was replaced

The previous watcher session, `r125_cuda_resume_r121_watcher`, only probed GPU 4. If CUDA recovered first on GPU 5, 6, or 7, it would keep missing the available device and would not start the R121 smoke run.

R126 replaces it with a multi-GPU watcher that probes GPU 4, 5, 6, and 7 in order every 300 seconds. The watcher launches only the first available GPU it finds.

## Active watcher

Session:

```bash
tmux attach -t r126_cuda_resume_r121_watcher_g4567
```

Log:

```bash
tail -f output/diagnostics/r126_cuda_resume_r121_watcher_g4567_20260518.log
```

Stop command:

```bash
tmux kill-session -t r126_cuda_resume_r121_watcher_g4567
```

## Launch guard

Before launching R121, the watcher checks all of the following again:

- `output/vc_suda/r121_depth_boundary_w001_smoke_b1_iter1_loss_only_fg00075` does not exist.
- No `output/vc_suda/r121_depth_boundary_w001_smoke_b1_iter1_loss_only_fg00075.retry_gpu*.tmux.log` file exists.
- No `train.py`, `torchrun`, or `torch.distributed.run` process is already running.
- A local lock directory can be created under `output/diagnostics/` so two watcher processes do not launch the same smoke at the same time.

## Smoke command

When a probed GPU becomes available, the watcher runs R121 smoke only:

```bash
CUDA_VISIBLE_DEVICES=<available_gpu> MAGFORMER_MS_DEFORM_ATTN_BACKEND=cuda \
/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python \
-m torch.distributed.run --standalone --nproc_per_node=1 tools/train.py \
--config configs/baseline_vc_suda_r121_depth_boundary_w001_smoke_b1_iter1_loss_only_fg00075.yaml \
--gpus 0 --num-workers 0 \
2>&1 | tee output/vc_suda/r121_depth_boundary_w001_smoke_b1_iter1_loss_only_fg00075.retry_gpu<gpu>_$(date +%Y%m%d_%H%M%S).tmux.log
```

This watcher does not start R122.

## Initial status

At startup on 2026-05-18, the first probe cycle checked GPU 4, 5, 6, and 7. All four probes failed with CUDA unknown error, so no R121 smoke was launched and the watcher went to the 300 second sleep interval.
