#!/bin/bash
# g2_launch.sh — launch one 4-rank g2 arm on GPUs 4-7 (physical, no
# CUDA_VISIBLE_DEVICES remap: runtime.gpus addresses cuda:4-7 directly)
# + start the host-memory watchdog on its process group.
# Usage: g2_launch.sh <config-name> [master_port]
set -eu
CFG="${1:?usage: g2_launch.sh <config-name> [port]}"
PORT="${2:-29611}"
PY=/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python
SRC=/home/hdd3/zhanghaonan/magformer/experiments_archive/v317_next_stage_2026-07/source
RUNS=/home/hdd3/zhanghaonan/magformer/output/aps_20260913/g2_runs
cd "$SRC"
mkdir -p "$RUNS"
export MALLOC_ARENA_MAX=2 MALLOC_TRIM_THRESHOLD_=268435456 MALLOC_MMAP_THRESHOLD_=134217728
setsid "$PY" -m torch.distributed.run --nproc_per_node=4 --master_port="$PORT" \
  tools/train.py --config "configs/aps_20260913_full/${CFG}.yaml" \
  > "$RUNS/${CFG}_r1.launch.log" 2>&1 &
PID=$!
echo "$PID" > "$RUNS/${CFG}_r1.pgid"
sleep 2
setsid bash "$SRC/ops/g2_watch.sh" "$PID" "$CFG" >/dev/null 2>&1 &
echo "launched $CFG train_pgid=$PID watchdog_pid=$!"
