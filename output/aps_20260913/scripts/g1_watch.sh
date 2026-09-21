#!/bin/bash
RUNS=/home/hdd3/zhanghaonan/magformer/output/aps_20260913/g1_runs
ts() { date '+%m-%d %H:%M:%S'; }
while true; do
  {
    echo "===== G1 DIGEST $(ts) ====="
    for arm in a0_control_16k b1_merged_16k; do
      d="$RUNS/$arm"
      if [ -d "$d" ]; then
        echo "--- $arm ---"
        grep '"phase": *"train"' "$d/metrics_log.jsonl" 2>/dev/null | tail -1 | grep -o '"iter": [0-9]*' | head -1
        grep '"phase": *"val"' "$d/metrics_log.jsonl" 2>/dev/null | tail -3 | grep -oE '"iter": [0-9]+|"val/segm_AP": [0-9.]+|"val/segm_APs": [0-9.]+'
        ls "$d"/*.pt 2>/dev/null | wc -l
      fi
    done
    tail -2 "$RUNS/relay.log" 2>/dev/null
  } > "$RUNS/G1_STATUS.txt"
  grep -q "G1 TRUE DONE" "$RUNS/relay.log" 2>/dev/null && { echo "[$(ts)] watcher exit" >> "$RUNS/arms.log"; break; }
  sleep 1800
done
