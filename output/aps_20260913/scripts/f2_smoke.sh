#!/bin/bash
# F2 merge smoke: 200-step 4-GPU DDP correctness+throughput vs single-GPU
# (accum 4, same effective batch) on the full winners config, from-scratch
# ImageNet init. Runs sequentially on GPUs 0-3; G1 arms on 4-7 untouched.
set -u
cd /home/hdd3/zhanghaonan/magformer/experiments_archive/v317_next_stage_2026-07/source
PY=/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python3.11
OUT=/home/hdd3/zhanghaonan/magformer/output/aps_20260913/p5_runs/f2_smoke
mkdir -p "$OUT"

echo "[f2_smoke $(date +%H:%M:%S)] phase 1: 4-GPU DDP 200 steps (port 29515)"
$PY -m torch.distributed.run --nproc_per_node=4 --master_port=29515 \
    tools/train.py --config configs/next_stage/f2_smoke_ddp4.yaml --gpus 0,1,2,3 \
    > "$OUT/ddp4.log" 2>&1
echo "[f2_smoke $(date +%H:%M:%S)] DDP exit=$? ; phase 2: single-GPU 200 opt steps (accum 4)"
$PY tools/train.py --config configs/next_stage/f2_smoke_single.yaml --gpus 0 \
    > "$OUT/single.log" 2>&1
echo "[f2_smoke $(date +%H:%M:%S)] single exit=$? ; phase 3: comparison"
$PY "$OUT/../.."/scripts/f2_smoke_compare.py > "$OUT/compare.txt" 2>&1
echo "[f2_smoke $(date +%H:%M:%S)] SMOKE CHAIN DONE"
