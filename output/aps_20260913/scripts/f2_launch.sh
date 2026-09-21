#!/bin/bash
# F2 256K launch: full architecture + design package + seven campaign winners,
# ImageNet from-scratch init, 4-GPU DDP on GPUs 4,5,6,7, port 29516.
# Run AFTER the G1 judgement (B1-A0 >= +1.5pt APs gate) and any winner-key
# flips recorded in configs/next_stage/f2_full_design_winners_256k.yaml.
#
# Also retires the stale s1c/s0c-era panel_watch daemons (they poll dead dirs)
# and starts fresh F2 monitoring (hourly digest + eval-dets subbucket panel).
set -eu
SRC=/home/hdd3/zhanghaonan/magformer/experiments_archive/v317_next_stage_2026-07/source
OUT=/home/hdd3/zhanghaonan/magformer/output/aps_20260913
export PATH=/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ts() { date '+%m-%d %H:%M:%S'; }

RUN=$OUT/p5_runs/f2_full_design_winners_256k
mkdir -p "$RUN"

# retire stale s1c/s0c panel watchers (plan step: avoid old-PID false alerts)
pkill -f "panel_watch.sh" 2>/dev/null || true
pkill -f "f1_five_hour.sh" 2>/dev/null || true

# Core pinning: foreign CPU-heavy jobs (holocue chromium/blender/vLLM) roam
# all 64 cores and starved the Python/data orchestration during the smoke
# (bursty 0-100% GPU util, up to 8s/it). Pin the training tree to 32-63.
echo "[$(ts)] F2 launching: torchrun 4-GPU (port 29516, GPUs 4,5,6,7; cores 32-63)"
cd "$SRC"
setsid nohup taskset -c 32-63 torchrun --nproc_per_node=4 --master_port=29516 \
  tools/train.py --config configs/next_stage/f2_full_design_winners_256k.yaml --gpus 4,5,6,7 \
  > "$RUN/run.log" 2>&1 < /dev/null &
echo "[$(ts)] trainer pid $!"

# fresh monitoring: hourly digest loop + eval-dets subbucket panel watcher
setsid nohup bash "$OUT/scripts/f2_five_hour.sh" > "$OUT/p5_runs/f2_digest_daemon.out" 2>&1 < /dev/null &
echo "[$(ts)] digest daemon pid $!"
setsid nohup bash "$OUT/scripts/f2_panel_watch.sh" > "$OUT/p5_runs/f2_panel_daemon.out" 2>&1 < /dev/null &
echo "[$(ts)] panel daemon pid $!"
echo "[$(ts)] F2 LAUNCHED"
