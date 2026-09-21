#!/bin/bash
# G1 arms: A0 control (uniform 16K tail-out) then B1 merged (all winners ON),
# sequential on GPUs 4-7 (DDP effective batch 4 = F1 recipe).
SRC=/home/hdd3/zhanghaonan/magformer/experiments_archive/v317_next_stage_2026-07/source
PY=/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python
RUNS=/home/hdd3/zhanghaonan/magformer/output/aps_20260913/g1_runs
ts() { date '+%m-%d %H:%M:%S'; }
echo "[$(ts)] G1-ARMS: launching A0 control" >> "$RUNS/arms.log"
cd "$SRC"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun --nproc_per_node=4 --master_port=29513 \
  tools/train.py --config configs/aps_20260913_full/g1_a0_control_16k.yaml --gpus 4,5,6,7 \
  > "$RUNS/a0_control_16k.log" 2>&1
echo "[$(ts)] G1-ARMS: A0 exit=$? ; launching B1 merged" >> "$RUNS/arms.log"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun --nproc_per_node=4 --master_port=29513 \
  tools/train.py --config configs/aps_20260913_full/g1_b1_merged_16k.yaml --gpus 4,5,6,7 \
  > "$RUNS/b1_merged_16k.log" 2>&1
echo "[$(ts)] G1-ARMS: B1 exit=$? ; ALL ARMS DONE" >> "$RUNS/arms.log"
