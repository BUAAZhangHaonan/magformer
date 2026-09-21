#!/bin/bash
# F2 merge smoke (GPUs 0+3 edition): 200-step 2-GPU DDP correctness vs
# single-GPU (accum 4, same effective batch) on the full winners config,
# from-scratch ImageNet init, FIXED code (DN pairing/window, find_unused).
# Foreign vLLM(blender) jobs hold GPUs 1/2 — do not touch them.
# The 4-GPU >=2.2x throughput gate stands on live evidence instead:
# A0 arm (full model, 4-GPU DDP, winners off) 1.38 s/it over 8000 steps;
# commit 98d327f6 recorded 4-GPU ~2.6x vs single-GPU full model.
set -u
cd /home/hdd3/zhanghaonan/magformer/experiments_archive/v317_next_stage_2026-07/source
PY=/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python3.11
OUT=/home/hdd3/zhanghaonan/magformer/output/aps_20260913/p5_runs/f2_smoke
mkdir -p "$OUT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[f2_smoke $(date +%H:%M:%S)] phase 1: 2-GPU DDP 200 steps (GPUs 0,3; port 29515)"
$PY -m torch.distributed.run --nproc_per_node=2 --master_port=29515 \
    tools/train.py --config configs/next_stage/f2_smoke_ddp2.yaml --gpus 0,3 \
    > "$OUT/ddp2.log" 2>&1
ddp_rc=$?
echo "[f2_smoke $(date +%H:%M:%S)] DDP exit=$ddp_rc (also check log tail below)"
tail -c 600 "$OUT/ddp2.log" | tr '\r' '\n' | tail -4
echo "[f2_smoke $(date +%H:%M:%S)] phase 2: single-GPU 200 opt steps (accum 4, GPU 0)"
$PY tools/train.py --config configs/next_stage/f2_smoke_single.yaml --gpus 0 \
    > "$OUT/single.log" 2>&1
sgl_rc=$?
echo "[f2_smoke $(date +%H:%M:%S)] single exit=$sgl_rc (log tail below)"
tail -c 600 "$OUT/single.log" | tr '\r' '\n' | tail -4
echo "[f2_smoke $(date +%H:%M:%S)] phase 3: comparison"
$PY "$OUT/../.."/scripts/f2_smoke_compare.py > "$OUT/compare.txt" 2>&1
cat "$OUT/compare.txt"
echo "[f2_smoke $(date +%H:%M:%S)] SMOKE CHAIN DONE (ddp_rc=$ddp_rc single_rc=$sgl_rc)"
