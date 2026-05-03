#!/usr/bin/env bash
set -uo pipefail

# Parallel full19 training script for GPUs 4,5,6,7
# Stream A (GPUs 4,5): magformer models + yolov8 models
# Stream B (GPU 6): 6 single-GPU models
# Stream C (GPU 7): 6 single-GPU models

EXP_DIR="/home/hdd3/zhanghaonan/magformer/output/experiments/20260502_1k_1566_20ep_1024_full19"
STAGING="${EXP_DIR}/_staging"
SCRIPTS="/home/hdd3/zhanghaonan/magformer/scripts/experiments"
DS_ROOT="/home/hdd3/zhanghaonan/magformer_datasets/20260318_1K_1566"
REGISTER="20260318_1K_1566"
LOG_DIR="${EXP_DIR}/parallel_logs"
mkdir -p "${LOG_DIR}"

CONDA_SH="/home/hdd3/zhanghaonan/anaconda3/etc/profile.d/conda.sh"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [parallel-full19] $*"; }

# Clean up failed staging directories
for d in magformer_nodpth_ref magformer_depthnorm_on magformer_lightdepth_convnextlite_spatialgate_edge_validhole magformer_lightdepth_mobilenetv3_sagate_edge_validhole magformer_lightdepth_mobilenetv3_spatialgate_edge_validhole; do
  if [ -d "${STAGING}/${d}" ]; then
    log "Cleaning failed staging: ${d}"
    rm -rf "${STAGING}/${d}"
  fi
done

# Common args
COMMON="--register ${REGISTER} --dataset-root ${DS_ROOT} --output-root ${STAGING} --image-size 1024 --run"

# ============================================================
# Stream A: GPUs 4,5 — magformer + yolov8
# ============================================================
run_stream_a() {
  source "${CONDA_SH}" 2>/dev/null || true
  export CUDA_VISIBLE_DEVICES=4,5

  local LOG_A="${LOG_DIR}/stream_a.log"
  log "Stream A starting on GPUs 4,5" | tee -a "${LOG_A}"

  # Magformer models (skip eval to avoid OOM)
  export EVAL_PERIOD_OVERRIDE=999999

  local mag_models=(
    "nodpth_ref"
    "depthnorm_on"
  )
  for v in "${mag_models[@]}"; do
    log "Stream A: magformer ${v}" | tee -a "${LOG_A}"
    bash "${SCRIPTS}/run_0831_1k_20ep_1024_revisit_magformer.sh" \
      ${COMMON} --variant "${v}" --ddp --num-gpus 2 \
      2>&1 | tee -a "${LOG_A}"
    log "Stream A: magformer ${v} done (rc=$?)" | tee -a "${LOG_A}"
  done

  local lightdepth_models=(
    "convnextlite_spatialgate_edge_validhole"
    "mobilenetv3_sagate_edge_validhole"
    "mobilenetv3_spatialgate_edge_validhole"
  )
  for v in "${lightdepth_models[@]}"; do
    log "Stream A: lightdepth ${v}" | tee -a "${LOG_A}"
    bash "${SCRIPTS}/run_0831_1k_20ep_1024_lightdepth_stage_a_magformer.sh" \
      ${COMMON} --variant "${v}" --ddp --num-gpus 2 \
      2>&1 | tee -a "${LOG_A}"
    log "Stream A: lightdepth ${v} done (rc=$?)" | tee -a "${LOG_A}"
  done

  unset EVAL_PERIOD_OVERRIDE

  # YOLOv8 models
  local yolo_sizes=("n" "s" "m" "l" "x")
  for sz in "${yolo_sizes[@]}"; do
    log "Stream A: yolov8_seg_${sz}" | tee -a "${LOG_A}"
    bash "${SCRIPTS}/run_0831_1k_20ep_scratch_yolov8_seg.sh" \
      ${COMMON} --model-size "${sz}" --pretrained --device 0,1 \
      2>&1 | tee -a "${LOG_A}"
    log "Stream A: yolov8_seg_${sz} done (rc=$?)" | tee -a "${LOG_A}"
  done

  log "Stream A: ALL DONE" | tee -a "${LOG_A}"
}

# ============================================================
# Stream B: GPU 6 — 6 single-GPU models
# ============================================================
run_stream_b() {
  source "${CONDA_SH}" 2>/dev/null || true
  export CUDA_VISIBLE_DEVICES=6

  local LOG_B="${LOG_DIR}/stream_b.log"
  log "Stream B starting on GPU 6" | tee -a "${LOG_B}"

  # 1. mgm_mask2former_nodpth_ref
  log "Stream B: mgm_mask2former_nodpth_ref" | tee -a "${LOG_B}"
  bash "${SCRIPTS}/run_0831_1k_20ep_1024_revisit_mgm_mask2former.sh" \
    ${COMMON} --variant nodpth_ref --num-gpus 1 \
    2>&1 | tee -a "${LOG_B}"
  log "Stream B: mgm_mask2former_nodpth_ref done (rc=$?)" | tee -a "${LOG_B}"

  # 2. mask2former
  log "Stream B: mask2former" | tee -a "${LOG_B}"
  bash "${SCRIPTS}/run_0831_1k_20ep_scratch_official_mask2former.sh" \
    ${COMMON} --pretrained \
    2>&1 | tee -a "${LOG_B}"
  log "Stream B: mask2former done (rc=$?)" | tee -a "${LOG_B}"

  # 3. msmformer
  log "Stream B: msmformer" | tee -a "${LOG_B}"
  bash "${SCRIPTS}/run_0831_1k_20ep_scratch_msmformer.sh" \
    ${COMMON} \
    2>&1 | tee -a "${LOG_B}"
  log "Stream B: msmformer done (rc=$?)" | tee -a "${LOG_B}"

  # 4. uoais
  log "Stream B: uoais" | tee -a "${LOG_B}"
  bash "${SCRIPTS}/run_0831_1k_20ep_scratch_uoais.sh" \
    ${COMMON} \
    2>&1 | tee -a "${LOG_B}"
  log "Stream B: uoais done (rc=$?)" | tee -a "${LOG_B}"

  # 5. iaunet
  log "Stream B: iaunet" | tee -a "${LOG_B}"
  bash "${SCRIPTS}/run_0831_1k_20ep_1024_revisit_iaunet_inst.sh" \
    ${COMMON} \
    2>&1 | tee -a "${LOG_B}"
  log "Stream B: iaunet done (rc=$?)" | tee -a "${LOG_B}"

  # 6. cellpose
  log "Stream B: cellpose" | tee -a "${LOG_B}"
  bash "${SCRIPTS}/run_0831_1k_20ep_1024_revisit_cellpose_inst.sh" \
    ${COMMON} \
    2>&1 | tee -a "${LOG_B}"
  log "Stream B: cellpose done (rc=$?)" | tee -a "${LOG_B}"

  log "Stream B: ALL DONE" | tee -a "${LOG_B}"
}

# ============================================================
# Stream C: GPU 7 — 6 single-GPU models
# ============================================================
run_stream_c() {
  source "${CONDA_SH}" 2>/dev/null || true
  export CUDA_VISIBLE_DEVICES=7

  local LOG_C="${LOG_DIR}/stream_c.log"
  log "Stream C starting on GPU 7" | tee -a "${LOG_C}"

  # 1. mgm_mask2former_depthnorm_on
  log "Stream C: mgm_mask2former_depthnorm_on" | tee -a "${LOG_C}"
  bash "${SCRIPTS}/run_0831_1k_20ep_1024_revisit_mgm_mask2former.sh" \
    ${COMMON} --variant depthnorm_on --num-gpus 1 \
    2>&1 | tee -a "${LOG_C}"
  log "Stream C: mgm_mask2former_depthnorm_on done (rc=$?)" | tee -a "${LOG_C}"

  # 2. maskrcnn
  log "Stream C: maskrcnn" | tee -a "${LOG_C}"
  bash "${SCRIPTS}/run_0831_1k_20ep_scratch_maskrcnn.sh" \
    ${COMMON} --pretrained \
    2>&1 | tee -a "${LOG_C}"
  log "Stream C: maskrcnn done (rc=$?)" | tee -a "${LOG_C}"

  # 3. stardist
  log "Stream C: stardist" | tee -a "${LOG_C}"
  bash "${SCRIPTS}/run_0831_1k_20ep_1024_revisit_stardist_inst.sh" \
    ${COMMON} \
    2>&1 | tee -a "${LOG_C}"
  log "Stream C: stardist done (rc=$?)" | tee -a "${LOG_C}"

  # 4. unet_semantic_inst
  log "Stream C: unet_semantic_inst" | tee -a "${LOG_C}"
  bash "${SCRIPTS}/run_0831_1k_20ep_1024_revisit_unet_semantic_inst.sh" \
    ${COMMON} \
    2>&1 | tee -a "${LOG_C}"
  log "Stream C: unet_semantic_inst done (rc=$?)" | tee -a "${LOG_C}"

  # 5. unet_boundary_inst
  log "Stream C: unet_boundary_inst" | tee -a "${LOG_C}"
  bash "${SCRIPTS}/run_0831_1k_20ep_1024_revisit_unet_boundary_inst.sh" \
    ${COMMON} \
    2>&1 | tee -a "${LOG_C}"
  log "Stream C: unet_boundary_inst done (rc=$?)" | tee -a "${LOG_C}"

  # 6. unetpp_boundary_inst
  log "Stream C: unetpp_boundary_inst" | tee -a "${LOG_C}"
  bash "${SCRIPTS}/run_0831_1k_20ep_1024_revisit_unetpp_boundary_inst.sh" \
    ${COMMON} \
    2>&1 | tee -a "${LOG_C}"
  log "Stream C: unetpp_boundary_inst done (rc=$?)" | tee -a "${LOG_C}"

  log "Stream C: ALL DONE" | tee -a "${LOG_C}"
}

# ============================================================
# Launch all 3 streams in parallel
# ============================================================
log "Starting parallel full19 training on GPUs 4,5,6,7"
log "Stream A (GPUs 4,5): 5 magformer + 5 yolov8"
log "Stream B (GPU 6): 6 single-GPU models"
log "Stream C (GPU 7): 6 single-GPU models"

run_stream_a &
PID_A=$!
run_stream_b &
PID_B=$!
run_stream_c &
PID_C=$!

log "PIDs: A=${PID_A} B=${PID_B} C=${PID_C}"

wait ${PID_A}
RC_A=$?
log "Stream A finished (rc=${RC_A})"

wait ${PID_B}
RC_B=$?
log "Stream B finished (rc=${RC_B})"

wait ${PID_C}
RC_C=$?
log "Stream C finished (rc=${RC_C})"

log "ALL STREAMS DONE. RC: A=${RC_A} B=${RC_B} C=${RC_C}"
