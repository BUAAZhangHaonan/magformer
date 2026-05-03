#!/usr/bin/env bash
set -euo pipefail

# Comprehensive retry for all failed models
# GPU 1 is dead (hardware error), so use GPUs 2,3 for DDP
# Free GPUs: 0, 2, 3, 6

STAGING="/home/hdd3/zhanghaonan/magformer/output/experiments/20260502_1k_1566_20ep_1024_full19/_staging"
REGISTER="20260318_1K_1566"
DATASET_ROOT="/home/hdd3/zhanghaonan/magformer_datasets/20260318_1K_1566"
SCRIPTS="/home/hdd3/zhanghaonan/magformer/scripts/experiments"

source /home/hdd3/zhanghaonan/anaconda3/etc/profile.d/conda.sh

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [retry-v2] $*"; }

run_model() {
  local model_id="$1"
  local clean="${2:-yes}"
  shift 2
  log "Starting: ${model_id}"

  if [ "$clean" = "yes" ]; then
    rm -rf "${STAGING}/${model_id}"
  else
    log "Keeping existing checkpoints for resume: ${model_id}"
  fi

  if "$@"; then
    log "DONE: ${model_id}"
  else
    local rc=$?
    log "FAILED rc=${rc}: ${model_id}"
  fi
}

log "=== Phase 0: Fix cellpose version ==="
conda run -n magformer pip install "cellpose==3.1.1.1" 2>&1 | tail -5 || log "cellpose install may have failed"

log "=== Phase 1: IAUNet (resume on GPU 6) ==="
export CUDA_VISIBLE_DEVICES=6
export EVAL_PERIOD_OVERRIDE=5
run_model iaunet no \
  bash "${SCRIPTS}/run_0831_1k_20ep_1024_revisit_iaunet_inst.sh" \
    --register "${REGISTER}" \
    --dataset-root "${DATASET_ROOT}" \
    --output-root "${STAGING}" \
    --image-size 1024 \
    --batch 8 \
    --val-batch 4 \
    --num-workers 4 \
    --num-queries 100 \
    --eval-every 5 \
    --grad-accum-steps 1 \
    --max-train-steps 0 \
    --max-val-images 0 \
    --run
unset EVAL_PERIOD_OVERRIDE

log "=== Phase 2: Cellpose (GPU 0) ==="
export CUDA_VISIBLE_DEVICES=0
run_model cellpose yes \
  bash "${SCRIPTS}/run_0831_1k_20ep_1024_revisit_cellpose_inst.sh" \
    --register "${REGISTER}" \
    --dataset-root "${DATASET_ROOT}" \
    --output-root "${STAGING}" \
    --image-size 1024 \
    --batch 16 \
    --eval-every 0 \
    --num-workers 4 \
    --inference-batch 4 \
    --max-train-steps 0 \
    --max-val-images 0 \
    --run

log "=== Phase 3: DDP models (GPUs 2,3) ==="
export CUDA_VISIBLE_DEVICES=2,3
export EVAL_PERIOD_OVERRIDE=999999

# 3a. MGM nodpth_ref
run_model mgm_mask2former_nodpth_ref yes \
  bash "${SCRIPTS}/run_0831_1k_20ep_1024_revisit_mgm_mask2former.sh" \
    --register "${REGISTER}" \
    --dataset-root "${DATASET_ROOT}" \
    --output-root "${STAGING}" \
    --variant nodpth_ref \
    --num-gpus 2 \
    --image-size 1024 \
    --run

# 3b. MGM depthnorm_on
run_model mgm_mask2former_depthnorm_on yes \
  bash "${SCRIPTS}/run_0831_1k_20ep_1024_revisit_mgm_mask2former.sh" \
    --register "${REGISTER}" \
    --dataset-root "${DATASET_ROOT}" \
    --output-root "${STAGING}" \
    --variant depthnorm_on \
    --num-gpus 2 \
    --image-size 1024 \
    --run

# 3c. Official Mask2Former
run_model official_mask2former_pretrained yes \
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

# 3d. MSMFormer
run_model msmformer yes \
  bash "${SCRIPTS}/run_0831_1k_20ep_scratch_msmformer.sh" \
    --register "${REGISTER}" \
    --dataset-root "${DATASET_ROOT}" \
    --output-root "${STAGING}" \
    --candidate-id C1 \
    --run-tag final \
    --image-size 1024 \
    --num-gpus 2 \
    --run

# 3e. UOAIS
run_model uoais_scratch yes \
  bash "${SCRIPTS}/run_0831_1k_20ep_scratch_uoais.sh" \
    --register "${REGISTER}" \
    --dataset-root "${DATASET_ROOT}" \
    --output-root "${STAGING}" \
    --candidate-id C1 \
    --run-tag final \
    --image-size 1024 \
    --num-gpus 2 \
    --run

unset EVAL_PERIOD_OVERRIDE

log "=== Phase 4: Stardist (GPU 0, TF) ==="
export CUDA_VISIBLE_DEVICES=0
rm -rf "${STAGING}/stardist"
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
  && log "DONE: stardist" \
  || log "FAILED: stardist (may need TF/CUDA fix)"

log "=== All retries completed ==="
