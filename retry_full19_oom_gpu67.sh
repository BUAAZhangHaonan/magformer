#!/usr/bin/env bash
set -euo pipefail

# Retry OOM models from full19 run at 1024 with 2 GPUs (6,7 DDP)
# These models OOM'd on a single 24GB GPU at 1024 resolution
# Using 2 GPUs halves per-GPU memory while keeping total batch size

export CUDA_VISIBLE_DEVICES=6,7
export EVAL_PERIOD_OVERRIDE=999999

STAGING="/home/hdd3/zhanghaonan/magformer/output/experiments/20260502_1k_1566_20ep_1024_full19/_staging"
REGISTER="20260318_1K_1566"
DATASET_ROOT="/home/hdd3/zhanghaonan/magformer_datasets/20260318_1K_1566"
SCRIPTS="/home/hdd3/zhanghaonan/magformer/scripts/experiments"

source /home/hdd3/zhanghaonan/anaconda3/etc/profile.d/conda.sh

run_model() {
  local model_id="$1"
  shift
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] [retry-oom-67] Starting: ${model_id}"

  # Clean staging dir to avoid stale checkpoints
  rm -rf "${STAGING}/${model_id}"

  if "$@"; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [retry-oom-67] DONE: ${model_id}"
  else
    local rc=$?
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [retry-oom-67] FAILED rc=${rc}: ${model_id}"
  fi
}

echo "[$(date '+%Y-%m-%d %H:%M:%S')] [retry-oom-67] Starting OOM retry with CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] [retry-oom-67] EVAL_PERIOD_OVERRIDE=${EVAL_PERIOD_OVERRIDE}"

# 1. MGM nodpth_ref (IMS_PER_BATCH=4 → 2 per GPU)
run_model mgm_mask2former_nodpth_ref \
  bash "${SCRIPTS}/run_0831_1k_20ep_1024_revisit_mgm_mask2former.sh" \
    --register "${REGISTER}" \
    --dataset-root "${DATASET_ROOT}" \
    --output-root "${STAGING}" \
    --variant nodpth_ref \
    --num-gpus 2 \
    --image-size 1024 \
    --run

# 2. MGM depthnorm_on (IMS_PER_BATCH=4 → 2 per GPU)
run_model mgm_mask2former_depthnorm_on \
  bash "${SCRIPTS}/run_0831_1k_20ep_1024_revisit_mgm_mask2former.sh" \
    --register "${REGISTER}" \
    --dataset-root "${DATASET_ROOT}" \
    --output-root "${STAGING}" \
    --variant depthnorm_on \
    --num-gpus 2 \
    --image-size 1024 \
    --run

# 3. Official Mask2Former (IMS_PER_BATCH=8 → 4 per GPU)
run_model official_mask2former_pretrained \
  bash "${SCRIPTS}/run_0831_1k_20ep_scratch_official_mask2former.sh" \
    --register "${REGISTER}" \
    --dataset-root "${DATASET_ROOT}" \
    --output-root "${STAGING}" \
    --candidate-id C1 \
    --run-tag final \
    --image-size 1024 \
    --num-gpus 2 \
    --pretrained \
    --run

# 4. MSMFormer (IMS_PER_BATCH=8 → 4 per GPU)
run_model msmformer \
  bash "${SCRIPTS}/run_0831_1k_20ep_scratch_msmformer.sh" \
    --register "${REGISTER}" \
    --dataset-root "${DATASET_ROOT}" \
    --output-root "${STAGING}" \
    --candidate-id C1 \
    --run-tag final \
    --image-size 1024 \
    --num-gpus 2 \
    --run

# 5. UOAIS (IMS_PER_BATCH=8 → 4 per GPU)
run_model uoais_scratch \
  bash "${SCRIPTS}/run_0831_1k_20ep_scratch_uoais.sh" \
    --register "${REGISTER}" \
    --dataset-root "${DATASET_ROOT}" \
    --output-root "${STAGING}" \
    --candidate-id C1 \
    --run-tag final \
    --image-size 1024 \
    --num-gpus 2 \
    --run

echo "[$(date '+%Y-%m-%d %H:%M:%S')] [retry-oom-67] All OOM retries completed"

# 6. Stardist (single GPU, uses stardist conda env with TF)
# CUDA_VISIBLE_DEVICES still 6,7 but stardist only uses GPU 6 via its own CUDA_VISIBLE_DEVICES
echo "[$(date '+%Y-%m-%d %H:%M:%S')] [retry-oom-67] Starting stardist retry (TF GPU fix)"
rm -rf "${STAGING}/stardist"
CUDA_VISIBLE_DEVICES=6 \
  bash "${SCRIPTS}/run_0831_1k_20ep_1024_revisit_stardist_inst.sh" \
    --register "${REGISTER}" \
    --dataset-root "${DATASET_ROOT}" \
    --output-root "${STAGING}" \
    --image-size 1024 \
    --epochs 20 \
    --batch 4 \
    --num-workers 0 \
    --ram-limit-pct 50 \
    --max-train-images 0 \
    --max-val-images 0 \
    --train-n-val-patches 8 \
    --run \
  && echo "[$(date '+%Y-%m-%d %H:%M:%S')] [retry-oom-67] DONE: stardist" \
  || echo "[$(date '+%Y-%m-%d %H:%M:%S')] [retry-oom-67] FAILED: stardist"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] [retry-oom-67] All retries (OOM + stardist) completed"
