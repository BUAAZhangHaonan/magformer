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
QUEUE_TAG="20260429_repaired_iaunet_100ep"
GPU="${CUDA_VISIBLE_DEVICES:-1}"
EPOCHS=100
NUM_WORKERS=4
NUM_QUERIES=100
EVAL_EVERY=20
WAIT_FREE_MB=60000
WAIT_SLEEP_SEC=30
MIN_RAM_MB=50000
MAX_SWAP_USED_MB=12000
BATCH_512=16
BATCH_1024=8
VAL_BATCH_512=4
VAL_BATCH_1024=4
GRAD_ACCUM_512=1
GRAD_ACCUM_1024=1
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
    --num-queries) NUM_QUERIES="$2"; shift 2 ;;
    --eval-every) EVAL_EVERY="$2"; shift 2 ;;
    --wait-free-mb) WAIT_FREE_MB="$2"; shift 2 ;;
    --wait-sleep-sec) WAIT_SLEEP_SEC="$2"; shift 2 ;;
    --min-ram-mb) MIN_RAM_MB="$2"; shift 2 ;;
    --max-swap-used-mb) MAX_SWAP_USED_MB="$2"; shift 2 ;;
    --batch-512) BATCH_512="$2"; shift 2 ;;
    --batch-1024) BATCH_1024="$2"; shift 2 ;;
    --val-batch-512) VAL_BATCH_512="$2"; shift 2 ;;
    --val-batch-1024) VAL_BATCH_1024="$2"; shift 2 ;;
    --grad-accum-512) GRAD_ACCUM_512="$2"; shift 2 ;;
    --grad-accum-1024) GRAD_ACCUM_1024="$2"; shift 2 ;;
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
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"

OUTPUT_ROOT_512="${OUTPUT_BASE}/20260429_repaired_unet_100ep_512_full19"
OUTPUT_ROOT_1024="${OUTPUT_BASE}/20260429_repaired_unet_100ep_1024_full19"
RUN_LOG="$(runner_setup_log "${OUTPUT_BASE}/${QUEUE_TAG}" "${MODE}")"

runner_log "${MODE}" "${RUN_LOG}" "[repaired-iaunet] CUDA_VISIBLE_DEVICES=${GPU}"
runner_log "${MODE}" "${RUN_LOG}" "[repaired-iaunet] dataset_root=${DATASET_ROOT}"
runner_log "${MODE}" "${RUN_LOG}" "[repaired-iaunet] output_root_512=${OUTPUT_ROOT_512}"
runner_log "${MODE}" "${RUN_LOG}" "[repaired-iaunet] output_root_1024=${OUTPUT_ROOT_1024}"
runner_log_launch_guard_snapshot "${MODE}" "${RUN_LOG}" "repaired-iaunet"

run_iaunet_if_missing() {
  local image_size="$1"
  local output_root="$2"
  local batch="$3"
  local val_batch="$4"
  local grad_accum="$5"
  local done_marker="${output_root}/iaunet/metrics.cocoeval.json"
  local label="iaunet_${image_size}_100ep"

  runner_log "${MODE}" "${RUN_LOG}" "[repaired-iaunet] plan ${label}"
  if [[ -f "${done_marker}" ]]; then
    runner_log "${MODE}" "${RUN_LOG}" "[repaired-iaunet] skip ${label}: ${done_marker}"
    return 0
  fi
  runner_wait_for_free_gpu_mb "${MODE}" "${RUN_LOG}" "${WAIT_FREE_MB}" "${WAIT_SLEEP_SEC}" "${label}"
  runner_wait_for_system_resources "${MODE}" "${RUN_LOG}" "${MIN_RAM_MB}" "${MAX_SWAP_USED_MB}" "${WAIT_SLEEP_SEC}" "${label}"
  runner_exec_locked "${MODE}" "${RUN_LOG}" "${done_marker}.lock" "${label}" \
    "cd '${REPO_ROOT}' && CUDA_VISIBLE_DEVICES=${GPU} bash '${REPO_ROOT}/scripts/experiments/run_0831_1k_20ep_1024_revisit_iaunet_inst.sh' --register '${REGISTER}' --dataset-root '${DATASET_ROOT}' --output-root '${output_root}' --image-size ${image_size} --epochs ${EPOCHS} --batch ${batch} --val-batch ${val_batch} --num-workers ${NUM_WORKERS} --num-queries ${NUM_QUERIES} --eval-every ${EVAL_EVERY} --grad-accum-steps ${grad_accum} --max-train-steps ${MAX_TRAIN_STEPS} --max-val-images ${MAX_VAL_IMAGES} --${MODE}"
}

run_iaunet_if_missing 512 "${OUTPUT_ROOT_512}" "${BATCH_512}" "${VAL_BATCH_512}" "${GRAD_ACCUM_512}"
run_iaunet_if_missing 1024 "${OUTPUT_ROOT_1024}" "${BATCH_1024}" "${VAL_BATCH_1024}" "${GRAD_ACCUM_1024}"

runner_log "${MODE}" "${RUN_LOG}" "[repaired-iaunet] done"
