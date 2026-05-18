# R130 Final Goal Audit Watcher

Date: 2026-05-18
Repo: `/home/hdd3/zhanghaonan/magformer`

R130 is a CPU-only final audit watcher. It waits for the R129 go/no-go JSON and then runs `tools/audit_vc_suda_goal.py` once. It does not start R121, R122, evaluation, or go/no-go generation. It does not call `update_goal`.

## Active Watcher

Session:

```bash
tmux attach -t r130_final_goal_audit_watcher
```

Log:

```bash
tail -f output/diagnostics/r130_final_goal_audit_watcher_20260518.log
```

Script:

```bash
tools/run_r130_final_goal_audit_watcher.sh
```

## Gate

Every 300 seconds, R130 checks:

```bash
output/diagnostics/r122_depth_boundary_w001_go_no_go_20260518/go_no_go.json
```

The audit runs only when that file exists and is non-empty. If it is missing or empty, R130 logs the state and sleeps. It does not launch any upstream watcher, training job, evaluation job, or comparator.

## Outputs

R130 writes:

```bash
output/diagnostics/r130_final_goal_audit_20260518/final_goal_audit.json
output/diagnostics/r130_final_goal_audit_20260518/final_goal_audit.md
```

If either output already exists, R130 logs that it will not overwrite existing audit output and exits.

## Status Handling

After `tools/audit_vc_suda_goal.py` finishes, R130 reads `status` from the final audit JSON and records it in the watcher log. If the status is `COMPLETE`, R130 still only records the status. It never calls `update_goal`.

## Safety

R130 uses:

```bash
CUDA_VISIBLE_DEVICES=""
```

It launches only:

```bash
tools/audit_vc_suda_goal.py
```

It never launches `tools/train.py`, `torchrun`, `torch.distributed.run`, `tools/evaluate_1024_backmap.py`, `tools/build_r122_bucket_compare.py`, or `tools/compare_r122_go_no_go.py`.
