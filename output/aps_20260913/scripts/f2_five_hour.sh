#!/bin/bash
# Hourly digest loop for the f1 full run (server-side daemon, survives app
# closure). Digests append to p5_runs/digest_history.log; a rolling snapshot
# of the newest digest is kept at p5_runs/LATEST_STATUS.txt for one-glance.
OUT=/home/hdd3/zhanghaonan/magformer/output/aps_20260913
while true; do
  {
    bash "$OUT/scripts/f2_digest.sh"
    echo
  } >> "$OUT/p5_runs/digest_history.log" 2>&1
  # refresh the one-glance status file (last 12 lines of history)
  tail -12 "$OUT/p5_runs/digest_history.log" > "$OUT/p5_runs/LATEST_STATUS.txt"
  sleep 3600
done
