#!/usr/bin/env bash
set -uo pipefail

# Watcher: keep all 4 GPUs occupied and auto-launch OOM retry
# Phase 1: Wait for Stream B to finish → launch GPU holder on GPU 6
# Phase 2: Wait for Stream C to finish → kill holder, launch OOM retry on GPUs 6,7
# Phase 3: Wait for retry to finish → if Stream A still running, wait. Then all done.

LOGDIR="/home/hdd3/zhanghaonan/magformer/output/experiments/20260502_1k_1566_20ep_1024_full19/parallel_logs"
STAGING="/home/hdd3/zhanghaonan/magformer/output/experiments/20260502_1k_1566_20ep_1024_full19/_staging"
RETRY_LOG="${LOGDIR}/retry_oom.log"
HOLDER_PID=""

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [watcher] $*"; }

source /home/hdd3/zhanghaonan/anaconda3/etc/profile.d/conda.sh

log "Watcher started. Monitoring streams to keep GPUs 4-7 occupied."

# ---- Phase 1: Wait for Stream B to finish ----
log "Phase 1: Waiting for Stream B to finish..."
while true; do
  if grep -q 'Stream B: ALL DONE' "${LOGDIR}/stream_b.log" 2>/dev/null; then
    log "Stream B finished!"
    break
  fi
  # Also check if parallel script died
  if ! pgrep -f 'run_parallel_1024_full19' > /dev/null 2>&1; then
    log "Parallel script exited. Launching retry immediately."
    # Parallel done, all GPUs free → launch retry on 4,5
    nohup bash /home/hdd3/zhanghaonan/magformer/retry_full19_oom.sh > "${RETRY_LOG}" 2>&1 &
    log "Retry launched on GPUs 4,5 (PID=$!). Waiting..."
    wait $!
    log "All done."
    exit 0
  fi
  sleep 60
done

log "Phase 1 complete. Starting GPU holder on GPU 6 to keep it occupied..."
# Launch GPU holder: 10GB on cuda:0 (which maps to GPU 6 via CUDA_VISIBLE_DEVICES)
export CUDA_VISIBLE_DEVICES=6
conda run -n magformer python /home/hdd3/zhanghaonan/magformer/gpu_holder.py 0 10 &
HOLDER_PID=$!
log "GPU holder started (PID=${HOLDER_PID}). Holding 10GB on GPU 6."

# ---- Phase 2: Wait for Stream C to finish ----
log "Phase 2: Waiting for Stream C to finish..."
while true; do
  if grep -q 'Stream C: ALL DONE' "${LOGDIR}/stream_c.log" 2>/dev/null; then
    log "Stream C finished!"
    break
  fi
  if ! pgrep -f 'run_parallel_1024_full19' > /dev/null 2>&1; then
    log "Parallel script exited."
    break
  fi
  sleep 60
done

# Kill GPU holder - we need GPU 6 for real work now
log "Killing GPU holder (PID=${HOLDER_PID})..."
kill ${HOLDER_PID} 2>/dev/null || true
sleep 5
kill -9 ${HOLDER_PID} 2>/dev/null || true
log "GPU holder killed. GPUs 6,7 now free for retry."

# ---- Phase 3: Launch OOM retry on GPUs 6,7 ----
log "Phase 3: Launching OOM retry on GPUs 6,7..."
nohup bash /home/hdd3/zhanghaonan/magformer/retry_full19_oom_gpu67.sh > "${RETRY_LOG}" 2>&1 &
RETRY_PID=$!
log "Retry script launched (PID=${RETRY_PID}). Log: ${RETRY_LOG}"

# Wait for retry to finish
wait ${RETRY_PID}
RETRY_RC=$?
log "Retry script finished (rc=${RETRY_RC})"

# ---- Phase 4: Check if Stream A is still running ----
if pgrep -f 'run_parallel_1024_full19' > /dev/null 2>&1; then
  log "Stream A still running on GPUs 4,5. Waiting for it to finish..."
  while pgrep -f 'run_parallel_1024_full19' > /dev/null 2>&1; do
    sleep 60
  done
  log "Stream A finished."
fi

log "========================================"
log "All 1024 full19 jobs completed!"
log "========================================"
