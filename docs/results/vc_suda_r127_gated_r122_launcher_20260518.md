# R127 Gated R122 Launcher

Date: 2026-05-18

R127 adds a gated launcher for the R122 300iter run. It does not replace R126. R126 still owns CUDA recovery and the R121 smoke launch. R127 only watches the readonly R121/R122 resume state and may start R122 after R121 smoke has passed.

## Active watcher

Session:

```bash
tmux attach -t r127_gated_r122_launcher
```

Log:

```bash
tail -f output/diagnostics/r127_gated_r122_launcher_20260518.log
```

Stop command:

```bash
tmux kill-session -t r127_gated_r122_launcher
```

## State gate

Every 300 seconds, the watcher runs:

```bash
/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python \
tools/check_r121_r122_resume_state.py \
--output-json /tmp/r127_state.json \
--output-md /tmp/r127_state.md
```

R127 can only consider launching R122 when the JSON field `state` is exactly `NEED_R122_TRAIN`.

The checker derives CUDA and R121-smoke readiness from the R126 watcher log when it exists, with the older R125 watcher log used only as a fallback.

For these states, R127 does not start training:

- `BLOCKED_CUDA`: record the next action and continue checking every 300 seconds.
- `NEED_R121_SMOKE`: record the next action and continue checking every 300 seconds.
- `NEED_R121_TRAIN`: record the next action and continue checking every 300 seconds.
- `R121_FAILED`: record the failure and exit.
- `R122_TRAINING`: record the train process summary, wait for R122 to exit, and do not run evaluation yet.
- `NEED_R122_EVAL`: record the next action and exit.
- `NEED_GO_NO_GO`: record the next action and exit.
- `READY_TO_DECIDE`: record the next action and exit.

## Safety checks before R122

Before launching R122, the watcher must pass all of these checks:

- No `train.py`, `torchrun`, or `torch.distributed.run` training process is running.
- `output/vc_suda/r122_depth_boundary_w001_r114warm_pseudo300` does not exist. If it exists, R127 records the conflict and exits without launching.
- GPU 4, 5, 6, and 7 each pass a PyTorch small tensor CUDA probe.
- The R122 tmux log path is timestamped as `output/vc_suda/r122_depth_boundary_w001_r114warm_pseudo300.retry_YYYYMMDD_HHMMSS.tmux.log`. R127 checks that this path does not already exist before launch, so a retry cannot overwrite an older fixed-name log.

R127 never starts R121. It only monitors the state checker and can launch R122 after the state checker reports `NEED_R122_TRAIN`.

## Active watcher maintenance

The current active R127 watcher may be a `/tmp` copy rather than this tracked repository script. Do not switch it during training.

If the active `/tmp` watcher must be replaced with the tracked script, first confirm that no R121 or R122 train/eval process is running. Then perform the switch only while CUDA remains blocked or during a planned maintenance window.

## R122 launch command

When the state gate and all safety checks pass, R127 runs:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 MAGFORMER_MS_DEFORM_ATTN_BACKEND=cuda \
/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python \
-m torch.distributed.run --standalone --nproc_per_node=4 tools/train.py \
--config configs/baseline_vc_suda_r122_depth_boundary_w001_pseudo300.yaml \
--gpus 0,1,2,3 --num-workers 2 \
2>&1 | tee output/vc_suda/r122_depth_boundary_w001_r114warm_pseudo300.retry_$(date +%Y%m%d_%H%M%S).tmux.log
```

After the R122 command exits, R127 records the exit code and exits. R127 does not run any evaluation automatically.
R122 evaluation is allowed only after the readonly checker no longer reports `R122_TRAINING` and no matching R122 training process remains.
