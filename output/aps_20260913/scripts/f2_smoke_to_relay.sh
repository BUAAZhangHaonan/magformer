#!/bin/bash
# Wait for the f2 smoke chain to fully finish, then resume the suspended G1
# relay (PID arg) so B1 launches on GPUs 4-7 (alone - only-one-4-rank rule).
OUT=/home/hdd3/zhanghaonan/magformer/output/aps_20260913
while ! grep -q "SMOKE CHAIN DONE" "$OUT/p5_runs/f2_smoke/chain.log" 2>/dev/null; do
  sleep 60
done
echo "[relay-resume $(date +%H:%M:%S)] smoke chain done; compare output:"
cat "$OUT/p5_runs/f2_smoke/compare.txt" 2>/dev/null
kill -CONT "$1" && echo "[relay-resume] relay $1 resumed -> B1 will launch within 300s"
