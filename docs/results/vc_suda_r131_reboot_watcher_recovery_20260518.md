# VC-SUDA R131 Reboot Watcher Recovery

Date: 2026-05-18

R131 adds a tracked reboot recovery entry point for the R126-R130 watcher chain. It is for host reboot or driver recovery cases where tmux sessions have been cleared.

## Run after reboot

From the repository root on 4029:

```bash
cd /home/hdd3/zhanghaonan/magformer
bash tools/start_vc_suda_watchers.sh
```

The startup log is:

```bash
tail -f output/diagnostics/start_vc_suda_watchers_20260518.log
```

The script is idempotent. If a tmux session already exists, it records the skip and does not create a duplicate.

## Watchers started

The script restores these sessions:

| Session | Role | Implementation |
| --- | --- | --- |
| `r126_cuda_resume_r121_watcher_g4567` | Probes GPU 4,5,6,7 and may launch only the gated R121 smoke after CUDA recovery. | Inline loop in `tools/start_vc_suda_watchers.sh`, matching R126 doc logic. |
| `r127_gated_r122_launcher` | Watches readonly R121/R122 state and may launch R122 only when state is `NEED_R122_TRAIN`. | Inline loop in `tools/start_vc_suda_watchers.sh`, matching R127 doc logic. |
| `r128_gated_r122_evaluator` | Watches for R122 eval readiness. | Delegates to `tools/run_r128_gated_r122_evaluator.sh`. |
| `r129_gated_go_no_go` | Watches completed eval outputs and runs post-eval go/no-go only after inputs exist. | Delegates to `tools/run_r129_gated_go_no_go.sh`. |
| `r130_final_goal_audit_watcher` | Watches go/no-go output and runs the final audit only after it exists. | Delegates to `tools/run_r130_final_goal_audit_watcher.sh`. |

## Safety boundary

`tools/start_vc_suda_watchers.sh` starts watcher tmux sessions only. It does not directly run training or evaluation in the startup path.

R126 and R127 can later launch their gated jobs only through their own documented checks. R128, R129, and R130 keep their existing tracked runner gates.

R127 writes R122 retry output to a timestamped log:

```bash
output/vc_suda/r122_depth_boundary_w001_r114warm_pseudo300.retry_YYYYMMDD_HHMMSS.tmux.log
```

The tracked script checks that the chosen log path does not already exist before launching R122. This avoids overwriting an older fixed-name tmux log.

The script uses the absolute interpreter:

```bash
/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python
```

It does not require conda activation or a bare `python` command.

## Inspect

List sessions:

```bash
tmux ls
```

Attach to a watcher:

```bash
tmux attach -t r126_cuda_resume_r121_watcher_g4567
tmux attach -t r127_gated_r122_launcher
tmux attach -t r128_gated_r122_evaluator
tmux attach -t r129_gated_go_no_go
tmux attach -t r130_final_goal_audit_watcher
```

Watch per-stage logs:

```bash
tail -f output/diagnostics/r126_cuda_resume_r121_watcher_g4567_20260518.log
tail -f output/diagnostics/r127_gated_r122_launcher_20260518.log
tail -f output/diagnostics/r128_gated_r122_evaluator_20260518.log
tail -f output/diagnostics/r129_gated_go_no_go_20260518.log
tail -f output/diagnostics/r130_final_goal_audit_watcher_20260518.log
```

## Active `/tmp` watcher maintenance

The active R127 watcher can differ from the tracked repository script when it was launched from a `/tmp` copy. Do not switch the active watcher while R121 or R122 train/eval work is running.

If the active `/tmp` watcher needs to be replaced with the tracked script, first confirm no R121 or R122 train/eval process exists. Then do the switch only while CUDA remains blocked or inside a planned maintenance window.

## Stop

Stop only the watcher session that you intentionally want to stop:

```bash
tmux kill-session -t r126_cuda_resume_r121_watcher_g4567
tmux kill-session -t r127_gated_r122_launcher
tmux kill-session -t r128_gated_r122_evaluator
tmux kill-session -t r129_gated_go_no_go
tmux kill-session -t r130_final_goal_audit_watcher
```
