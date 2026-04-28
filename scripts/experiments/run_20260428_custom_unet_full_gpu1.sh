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
QUEUE_TAG="20260428_custom_unet_full_gpu1"
GPU="1"
WAIT_FREE_MB=78000
WAIT_SLEEP_SEC=30
MIN_RAM_MB=50000
MAX_SWAP_USED_MB=12000

while [[ $# -gt 0 ]]; do
  case "$1" in
    --register)
      REGISTER="$2"
      shift 2
      ;;
    --dataset-root)
      DATASET_ROOT="$2"
      shift 2
      ;;
    --output-base)
      OUTPUT_BASE="$2"
      shift 2
      ;;
    --queue-tag)
      QUEUE_TAG="$2"
      shift 2
      ;;
    --gpu)
      GPU="$2"
      shift 2
      ;;
    --wait-free-mb)
      WAIT_FREE_MB="$2"
      shift 2
      ;;
    --wait-sleep-sec)
      WAIT_SLEEP_SEC="$2"
      shift 2
      ;;
    --min-ram-mb)
      MIN_RAM_MB="$2"
      shift 2
      ;;
    --max-swap-used-mb)
      MAX_SWAP_USED_MB="$2"
      shift 2
      ;;
    --run)
      MODE="run"
      shift
      ;;
    --dry-run)
      MODE="dry-run"
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
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

OUTPUT_ROOT_512="${OUTPUT_BASE}/20260406_1k_1566_20ep_512_full19"
OUTPUT_ROOT_1024="${OUTPUT_BASE}/20260406_1k_1566_20ep_1024_full19"
RUN_LOG="$(runner_setup_log "${OUTPUT_BASE}/${QUEUE_TAG}" "${MODE}")"

runner_log "${MODE}" "${RUN_LOG}" "[custom-unet-full] CUDA_VISIBLE_DEVICES=${GPU}"
runner_log "${MODE}" "${RUN_LOG}" "[custom-unet-full] dataset_root=${DATASET_ROOT}"
runner_log "${MODE}" "${RUN_LOG}" "[custom-unet-full] queue_tag=${QUEUE_TAG}"
runner_log "${MODE}" "${RUN_LOG}" "[custom-unet-full] output_root_512=${OUTPUT_ROOT_512}"
runner_log "${MODE}" "${RUN_LOG}" "[custom-unet-full] output_root_1024=${OUTPUT_ROOT_1024}"
runner_log "${MODE}" "${RUN_LOG}" "[custom-unet-full] wait_free_mb=${WAIT_FREE_MB} min_ram_mb=${MIN_RAM_MB} max_swap_used_mb=${MAX_SWAP_USED_MB}"

run_full_if_missing() {
  local label="$1"
  local done_marker="$2"
  shift 2
  runner_log "${MODE}" "${RUN_LOG}" "[custom-unet-full] plan ${label}"
  if [[ -f "${done_marker}" ]]; then
    runner_log "${MODE}" "${RUN_LOG}" "[custom-unet-full] skip ${label}: ${done_marker}"
    return 0
  fi
  runner_wait_for_free_gpu_mb "${MODE}" "${RUN_LOG}" "${WAIT_FREE_MB}" "${WAIT_SLEEP_SEC}" "${label}"
  runner_wait_for_system_resources "${MODE}" "${RUN_LOG}" "${MIN_RAM_MB}" "${MAX_SWAP_USED_MB}" "${WAIT_SLEEP_SEC}" "${label}"
  if [[ -f "${done_marker}" ]]; then
    runner_log "${MODE}" "${RUN_LOG}" "[custom-unet-full] skip ${label} after wait: ${done_marker}"
    return 0
  fi
  runner_exec_locked "${MODE}" "${RUN_LOG}" "${done_marker}.lock" "${label}" "$*"
}

run_full_if_missing \
  "cellpose_512_full" \
  "${OUTPUT_ROOT_512}/cellpose/metrics.cocoeval.json" \
  "cd '${REPO_ROOT}' && CUDA_VISIBLE_DEVICES=${GPU} CELLPOSE_NUM_WORKERS=4 CELLPOSE_INFERENCE_BATCH=8 bash '${REPO_ROOT}/scripts/experiments/run_0831_1k_20ep_1024_revisit_cellpose_inst.sh' --register '${REGISTER}' --dataset-root '${DATASET_ROOT}' --output-root '${OUTPUT_ROOT_512}' --image-size 512 --${MODE}"

run_full_if_missing \
  "cellpose_1024_full" \
  "${OUTPUT_ROOT_1024}/cellpose/metrics.cocoeval.json" \
  "cd '${REPO_ROOT}' && CUDA_VISIBLE_DEVICES=${GPU} CELLPOSE_NUM_WORKERS=4 CELLPOSE_INFERENCE_BATCH=4 bash '${REPO_ROOT}/scripts/experiments/run_0831_1k_20ep_1024_revisit_cellpose_inst.sh' --register '${REGISTER}' --dataset-root '${DATASET_ROOT}' --output-root '${OUTPUT_ROOT_1024}' --image-size 1024 --${MODE}"

run_full_if_missing \
  "iaunet_512_full" \
  "${OUTPUT_ROOT_512}/iaunet/metrics.cocoeval.json" \
  "cd '${REPO_ROOT}' && CUDA_VISIBLE_DEVICES=${GPU} bash '${REPO_ROOT}/scripts/experiments/run_0831_1k_20ep_1024_revisit_iaunet_inst.sh' --register '${REGISTER}' --dataset-root '${DATASET_ROOT}' --output-root '${OUTPUT_ROOT_512}' --image-size 512 --${MODE}"

run_full_if_missing \
  "iaunet_1024_full" \
  "${OUTPUT_ROOT_1024}/iaunet/metrics.cocoeval.json" \
  "cd '${REPO_ROOT}' && CUDA_VISIBLE_DEVICES=${GPU} bash '${REPO_ROOT}/scripts/experiments/run_0831_1k_20ep_1024_revisit_iaunet_inst.sh' --register '${REGISTER}' --dataset-root '${DATASET_ROOT}' --output-root '${OUTPUT_ROOT_1024}' --image-size 1024 --${MODE}"

runner_log "${MODE}" "${RUN_LOG}" "[custom-unet-full] done"
