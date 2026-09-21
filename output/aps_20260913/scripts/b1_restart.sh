#!/bin/bash
# B1 restart on GPUs 4-7 ONLY (user rule; GPUs 0-3 forbidden), sync-fixed
# code (329977e3), one 4-rank bundle alone (251GB host rule).
SRC=/home/hdd3/zhanghaonan/magformer/experiments_archive/v317_next_stage_2026-07/source
OUT=/home/hdd3/zhanghaonan/magformer/output/aps_20260913
export PATH=/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin:$PATH
export MALLOC_ARENA_MAX=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MAGFORMER_PROF_STEP=1
cd "$SRC"
exec taskset -c 32-63 torchrun --nproc_per_node=4 --master_port=29514 \
  tools/train.py --config configs/aps_20260913_full/g1_b1_merged_16k.yaml --gpus 4,5,6,7 \
  >> "$OUT/g1_runs/b1_merged_16k.log" 2>&1
