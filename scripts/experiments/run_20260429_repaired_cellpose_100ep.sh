#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/common_runner.sh"
source "${SCRIPT_DIR}/ecc_common.sh"

MODE="run"
REGISTER="20260318_1K_1566"
DATASET_ROOT=""
OUTPUT_BASE="${REPO_ROOT}/output/experiments"
QUEUE_TAG="20260429_repaired_cellpose_100ep"
GPU="${CUDA_VISIBLE_DEVICES:-1}"
EPOCHS=100
NUM_WORKERS=4
WAIT_FREE_MB=20000
WAIT_SLEEP_SEC=30
MIN_RAM_MB=50000
MAX_SWAP_USED_MB=12000
BATCH_512=32
BATCH_1024=16
INFERENCE_BATCH_512=8
INFERENCE_BATCH_1024=4
MAX_TRAIN_STEPS=0
MAX_VAL_IMAGES=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --register) REGISTER="$2"; shift 2 ;;
    --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
    --output-base) OUTPUT_BASE="$2"; shift 2 ;;
    --queue-tag) QUEUE_TAG="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    --wait-free-mb) WAIT_FREE_MB="$2"; shift 2 ;;
    --wait-sleep-sec) WAIT_SLEEP_SEC="$2"; shift 2 ;;
    --min-ram-mb) MIN_RAM_MB="$2"; shift 2 ;;
    --max-swap-used-mb) MAX_SWAP_USED_MB="$2"; shift 2 ;;
    --batch-512) BATCH_512="$2"; shift 2 ;;
    --batch-1024) BATCH_1024="$2"; shift 2 ;;
    --max-train-steps) MAX_TRAIN_STEPS="$2"; shift 2 ;;
    --max-val-images) MAX_VAL_IMAGES="$2"; shift 2 ;;
    --run) MODE="run"; shift ;;
    --dry-run) MODE="dry-run"; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "${DATASET_ROOT}" ]]; then
  DATASET_ROOT="$(ecc_default_dataset_root "${REGISTER}")"
fi
if [[ -d "${DATASET_ROOT}" ]]; then
  DATASET_ROOT="$(cd "${DATASET_ROOT}" && pwd)"
elif [[ "${MODE}" == "run" ]]; then
  echo "Dataset root does not exist: ${DATASET_ROOT}" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
export OPENCV_FOR_THREADS_NUM="${OPENCV_FOR_THREADS_NUM:-4}"
export TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-4}"
export TF_NUM_INTEROP_THREADS="${TF_NUM_INTEROP_THREADS:-2}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"

OUTPUT_ROOT_512="${OUTPUT_BASE}/20260429_repaired_unet_100ep_512_full19"
OUTPUT_ROOT_1024="${OUTPUT_BASE}/20260429_repaired_unet_100ep_1024_full19"
RUN_LOG="$(runner_setup_log "${OUTPUT_BASE}/${QUEUE_TAG}" "${MODE}")"

runner_log "${MODE}" "${RUN_LOG}" "[repaired-cellpose] CUDA_VISIBLE_DEVICES=${GPU}"
runner_log "${MODE}" "${RUN_LOG}" "[repaired-cellpose] dataset_root=${DATASET_ROOT}"
runner_log "${MODE}" "${RUN_LOG}" "[repaired-cellpose] output_root_512=${OUTPUT_ROOT_512}"
runner_log "${MODE}" "${RUN_LOG}" "[repaired-cellpose] output_root_1024=${OUTPUT_ROOT_1024}"
runner_log_launch_guard_snapshot "${MODE}" "${RUN_LOG}" "repaired-cellpose"

run_cellpose_if_missing() {
  local image_size="$1"
  local output_root="$2"
  local batch="$3"
  local inference_batch="$4"
  local done_marker="${output_root}/cellpose/metrics.cocoeval.json"
  local label="cellpose_${image_size}_100ep"
  local target_cache_dir="${output_root}/cellpose/target_cache/${image_size}_official-cellpose-3.1.1.1-diffusion"

  runner_log "${MODE}" "${RUN_LOG}" "[repaired-cellpose] plan ${label}"
  if runner_json_file_valid "${done_marker}"; then
    runner_log "${MODE}" "${RUN_LOG}" "[repaired-cellpose] skip ${label}: ${done_marker}"
    return 0
  fi
  runner_wait_for_free_gpu_mb "${MODE}" "${RUN_LOG}" "${WAIT_FREE_MB}" "${WAIT_SLEEP_SEC}" "${label}"
  runner_wait_for_system_resources "${MODE}" "${RUN_LOG}" "${MIN_RAM_MB}" "${MAX_SWAP_USED_MB}" "${WAIT_SLEEP_SEC}" "${label}"
  runner_exec_gpu_locked "${MODE}" "${RUN_LOG}" "${GPU}" "${done_marker}.lock" "${label}" \
    "cd '${REPO_ROOT}' && CUDA_VISIBLE_DEVICES=${GPU} CELLPOSE_NUM_WORKERS=${NUM_WORKERS} CELLPOSE_INFERENCE_BATCH=${inference_batch} CELLPOSE_TARGET_CACHE_DIR='${target_cache_dir}' bash '${REPO_ROOT}/scripts/experiments/run_0831_1k_20ep_1024_revisit_cellpose_inst.sh' --register '${REGISTER}' --dataset-root '${DATASET_ROOT}' --output-root '${output_root}' --image-size ${image_size} --epochs ${EPOCHS} --batch ${batch} --num-workers ${NUM_WORKERS} --inference-batch ${inference_batch} --max-train-steps ${MAX_TRAIN_STEPS} --max-val-images ${MAX_VAL_IMAGES} --${MODE}"
}

run_cellpose_if_missing 512 "${OUTPUT_ROOT_512}" "${BATCH_512}" "${INFERENCE_BATCH_512}"
run_cellpose_if_missing 1024 "${OUTPUT_ROOT_1024}" "${BATCH_1024}" "${INFERENCE_BATCH_1024}"

runner_log "${MODE}" "${RUN_LOG}" "[repaired-cellpose] done"
