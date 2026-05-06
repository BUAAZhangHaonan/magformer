#!/bin/bash
set -eo pipefail

# Prevent HuggingFace downloads -- use offline mode and Chinese mirror
export HF_HUB_OFFLINE=1
export HF_ENDPOINT=https://hf-mirror.com
export TRANSFORMERS_OFFLINE=1
source /home/hdd3/zhanghaonan/anaconda3/etc/profile.d/conda.sh
conda activate magformer
cd /home/hdd3/zhanghaonan/magformer

export CUDA_VISIBLE_DEVICES=4,5,6,7
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p logs output/vc_suda

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a logs/vc_suda_pipeline.log
}

run_stage() {
  local STAGE=$1
  local CONFIG=$2
  local CKPT=$3
  local PORT=$4
  local NUM_GPUS=${5:-4}

  log "=== Starting Stage ${STAGE} (${NUM_GPUS} GPUs) ==="
  log "Config: ${CONFIG}"
  log "Resume from: ${CKPT}"
  log "Using port: ${PORT}"
  log "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

  if [ ! -f "${CKPT}" ]; then
    log "ERROR: Checkpoint ${CKPT} not found. Aborting."
    exit 1
  fi

  local LOG=logs/stage_${STAGE}.log

  # Wait for port to be free
  while ss -tlnp | grep -q ":${PORT} "; do
    log "Port ${PORT} still in use, waiting 10s..."
    sleep 10
  done

  # Run training
  torchrun --nproc_per_node=${NUM_GPUS} --master_port=${PORT} tools/train_vc_suda.py       --config ${CONFIG}       --finetune-weights ${CKPT}       2>&1 | tee ${LOG}

  log "=== Stage ${STAGE} complete ==="
}

# Stage A: source pretrain (port 29501, 4 GPUs)
log "=== Starting Stage A (4 GPUs) ==="
log "Config: configs/vc_suda/stage_a_source_pretrain.yaml"
log "Using port: 29511"
log "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
torchrun --nproc_per_node=4 --master_port=29511 tools/train_vc_suda.py   --config configs/vc_suda/stage_a_source_pretrain.yaml   2>&1 | tee logs/stage_A.log
log "=== Stage A complete ==="

# Stage B: target warmup (port 29502, 4 GPUs)
run_stage B configs/vc_suda/stage_b_target_warmup.yaml output/vc_suda/stage_a/model_best.pth 29502 4

# Stage C: semi-supervised EMA (port 29503, 4 GPUs)
run_stage C configs/vc_suda/stage_c_semi_supervised.yaml output/vc_suda/stage_b/model_best.pth 29503 4

# Stage D: domain alignment (port 29504, 4 GPUs)
run_stage D configs/vc_suda/stage_d_domain_alignment.yaml output/vc_suda/stage_c/model_best.pth 29504 4

# Stage E: final finetune (port 29505, 4 GPUs)
run_stage E configs/vc_suda/stage_e_final_finetune.yaml output/vc_suda/stage_d/model_best.pth 29505 4

log "=== All VC-SUDA stages complete ==="
