#!/bin/bash
# Convergence-run panel watcher: snapshots each arm's eval dets (the trainer
# overwrites coco_instances_results.json every eval_period) and runs the
# offline indicator panel (P3-P6). P1/P2/P3-checkpoint probes run on the
# checkpoint every 4 snapshots (32K steps).
PY=/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python
OUT=/home/hdd3/zhanghaonan/magformer/output/aps_20260913
for arm in f2_full_design_winners_256k; do
  (
    last_mtime=0
    while true; do
      f="$OUT/p5_runs/$arm/coco_instances_results.json"
      if [ -f "$f" ]; then
        mtime=$(stat -c %Y "$f")
        if [ "$mtime" != "$last_mtime" ] && [ "$(( $(date +%s) - mtime ))" -gt 120 ]; then
          last_mtime=$mtime
          step=$(ls -t "$OUT/p5_runs/$arm"/*.pt 2>/dev/null | head -1 | xargs -r stat -c %Y || echo 0)
          snap="$OUT/p5_runs/$arm/eval_snapshots/$(date +%H%M)_$(date +%m%d)"
          mkdir -p "$snap"
          cp "$f" "$snap/dets.json"
          cd "$OUT" && $PY scripts/rescore_dets.py --dets "$snap/dets.json" \
            --max-dets 100 --save "$snap/panel.json" > "$snap/panel.log" 2>&1
          $PY scripts/subbucket_scores.py "$snap/dets.json" "$snap/subbucket.json" 2>> "$snap/panel.log"
          # keep only the latest 12 snapshots per arm to bound disk
          ls -dt "$OUT/p5_runs/$arm/eval_snapshots"/*/ 2>/dev/null | tail -n +13 | xargs -r rm -rf
        fi
      fi
      sleep 300
    done
  ) &
done
wait
