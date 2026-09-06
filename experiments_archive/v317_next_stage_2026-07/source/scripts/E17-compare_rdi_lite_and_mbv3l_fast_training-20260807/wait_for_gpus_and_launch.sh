#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/g203-4028/magformer
EXPERIMENT_DIR="$ROOT/scripts/E17-compare_rdi_lite_and_mbv3l_fast_training-20260807"
OUTPUT_ROOT="$ROOT/output/experiments/E17-compare_rdi_lite_and_mbv3l_fast_training-20260807"
LOG="$OUTPUT_ROOT/gpu_wait.log"
STATUS="$OUTPUT_ROOT/gpu_wait.status"

mkdir -p "$OUTPUT_ROOT"
exec >>"$LOG" 2>&1

while true; do
  compute_pids=$(nvidia-smi -i 4,5 --query-compute-apps=pid --format=csv,noheader,nounits)

  if [[ -z "${compute_pids//[[:space:]]/}" ]]; then
    printf 'launching\n' >"$STATUS"
    "$EXPERIMENT_DIR/launch.sh"
    printf 'launched\n' >"$STATUS"
    exit 0
  fi

  printf 'waiting\n' >"$STATUS"
  printf '%s waiting for GPU4 and GPU5; compute PIDs: %s\n' "$(date --iso-8601=seconds)" "$(tr '\n' ' ' <<<"$compute_pids")"
  sleep 30
done
