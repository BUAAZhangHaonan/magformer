#!/bin/bash
# Smoke recovery sequence after the find_unused_parameters fix:
# 1. wait for the running chain (its single-GPU leg is valid: no DDP, runs
#    the fixed criterion end-to-end) to write SMOKE CHAIN DONE;
# 2. rerun ONLY the 4-GPU DDP leg with the fixed config (find_unused=true);
# 3. rerun the comparison;
# 4. resume the suspended G1 relay (PID arg) so B1 launches solo on 4-7.
set -u
SRC=/home/hdd3/zhanghaonan/magformer/experiments_archive/v317_next_stage_2026-07/source
PY=/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python3.11
OUT=/home/hdd3/zhanghaonan/magformer/output/aps_20260913
S=$OUT/p5_runs/f2_smoke
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

while ! grep -q "SMOKE CHAIN DONE" "$S/chain.log" 2>/dev/null; do sleep 60; done
echo "[recovery $(date +%H:%M:%S)] chain done; rerunning DDP leg with find_unused=true"

rm -rf "$S/ddp4"
cd "$SRC" || exit 1
$PY -m torch.distributed.run --nproc_per_node=4 --master_port=29515 \
    tools/train.py --config configs/next_stage/f2_smoke_ddp4.yaml --gpus 0,1,2,3 \
    > "$S/ddp4.log" 2>&1
echo "[recovery $(date +%H:%M:%S)] DDP rerun exit=$? ; comparing"
$PY "$OUT/scripts/f2_smoke_compare.py" > "$S/compare.txt" 2>&1
cat "$S/compare.txt"
kill -CONT "$1" && echo "[recovery $(date +%H:%M:%S)] relay $1 resumed -> B1 launches within 300s"
echo "[recovery] SEQUENCE DONE"
