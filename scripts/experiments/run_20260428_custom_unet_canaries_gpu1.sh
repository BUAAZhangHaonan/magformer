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
QUEUE_TAG="20260428_custom_unet_canaries_gpu1"
GPU="1"
WAIT_FREE_MB=78000
WAIT_SLEEP_SEC=30
MIN_RAM_MB=50000
MAX_SWAP_USED_MB=1024
MAX_TRAIN_STEPS=50
MAX_VAL_IMAGES=32

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
    --max-train-steps)
      MAX_TRAIN_STEPS="$2"
      shift 2
      ;;
    --max-val-images)
      MAX_VAL_IMAGES="$2"
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

CANARY_ROOT="${OUTPUT_BASE}/${QUEUE_TAG}"
RUN_LOG="$(runner_setup_log "${CANARY_ROOT}" "${MODE}")"
if [[ -d "${DATASET_ROOT}" ]]; then
  DATASET_ROOT="$(cd "${DATASET_ROOT}" && pwd)"
elif [[ "${MODE}" == "run" ]]; then
  echo "Dataset root does not exist: ${DATASET_ROOT}" >&2
  exit 1
fi

runner_log "${MODE}" "${RUN_LOG}" "[custom-unet-canary] CUDA_VISIBLE_DEVICES=${GPU}"
runner_log "${MODE}" "${RUN_LOG}" "[custom-unet-canary] dataset_root=${DATASET_ROOT}"
runner_log "${MODE}" "${RUN_LOG}" "[custom-unet-canary] canary_root=${CANARY_ROOT}"
runner_log "${MODE}" "${RUN_LOG}" "[custom-unet-canary] max_train_steps=${MAX_TRAIN_STEPS} max_val_images=${MAX_VAL_IMAGES}"
runner_log "${MODE}" "${RUN_LOG}" "[custom-unet-canary] wait_free_mb=${WAIT_FREE_MB} min_ram_mb=${MIN_RAM_MB} max_swap_used_mb=${MAX_SWAP_USED_MB}"

run_canary_if_missing() {
  local label="$1"
  local done_marker="$2"
  shift 2
  runner_log "${MODE}" "${RUN_LOG}" "[custom-unet-canary] plan ${label}"
  if runner_json_file_valid "${done_marker}"; then
    runner_log "${MODE}" "${RUN_LOG}" "[custom-unet-canary] skip ${label}: ${done_marker}"
    return 0
  fi
  runner_wait_for_free_gpu_mb "${MODE}" "${RUN_LOG}" "${WAIT_FREE_MB}" "${WAIT_SLEEP_SEC}" "${label}"
  runner_wait_for_system_resources "${MODE}" "${RUN_LOG}" "${MIN_RAM_MB}" "${MAX_SWAP_USED_MB}" "${WAIT_SLEEP_SEC}" "${label}"
  if runner_json_file_valid "${done_marker}"; then
    runner_log "${MODE}" "${RUN_LOG}" "[custom-unet-canary] skip ${label} after wait: ${done_marker}"
    return 0
  fi
  runner_exec_locked "${MODE}" "${RUN_LOG}" "${done_marker}.lock" "${label}" "$*"
}

run_canary_if_missing \
  "cellpose_512_canary" \
  "${CANARY_ROOT}/cellpose_512/metrics.cocoeval.json" \
  "cd '${REPO_ROOT}' && CUDA_VISIBLE_DEVICES=${GPU} conda run -n magformer python baselines/run_cellpose_instance_ecc.py --dataset-root '${DATASET_ROOT}' --output-dir '${CANARY_ROOT}/cellpose_512' --image-size 512 --epochs 2 --batch 32 --num-workers 4 --device cuda --target-cache-dir '${CANARY_ROOT}/cellpose_512/target_cache/512_official-cellpose-3.1.1.1-diffusion' --log-every 10 --inference-batch 8 --max-train-steps ${MAX_TRAIN_STEPS} --max-val-images ${MAX_VAL_IMAGES}"

run_canary_if_missing \
  "cellpose_1024_canary" \
  "${CANARY_ROOT}/cellpose_1024/metrics.cocoeval.json" \
  "cd '${REPO_ROOT}' && CUDA_VISIBLE_DEVICES=${GPU} conda run -n magformer python baselines/run_cellpose_instance_ecc.py --dataset-root '${DATASET_ROOT}' --output-dir '${CANARY_ROOT}/cellpose_1024' --image-size 1024 --epochs 2 --batch 16 --num-workers 4 --device cuda --target-cache-dir '${CANARY_ROOT}/cellpose_1024/target_cache/1024_official-cellpose-3.1.1.1-diffusion' --log-every 10 --inference-batch 4 --max-train-steps ${MAX_TRAIN_STEPS} --max-val-images ${MAX_VAL_IMAGES}"

run_canary_if_missing \
  "iaunet_512_canary" \
  "${CANARY_ROOT}/iaunet_512/metrics.cocoeval.json" \
  "cd '${REPO_ROOT}' && CUDA_VISIBLE_DEVICES=${GPU} conda run -n magformer python baselines/run_iaunet_instance_ecc.py --dataset-root '${DATASET_ROOT}' --output-dir '${CANARY_ROOT}/iaunet_512' --image-size 512 --epochs 2 --batch 16 --val-batch 4 --num-workers 4 --num-queries 100 --eval-every 1 --device cuda --amp --max-train-steps ${MAX_TRAIN_STEPS} --max-val-images ${MAX_VAL_IMAGES}"

run_canary_if_missing \
  "iaunet_1024_canary" \
  "${CANARY_ROOT}/iaunet_1024/metrics.cocoeval.json" \
  "cd '${REPO_ROOT}' && CUDA_VISIBLE_DEVICES=${GPU} conda run -n magformer python baselines/run_iaunet_instance_ecc.py --dataset-root '${DATASET_ROOT}' --output-dir '${CANARY_ROOT}/iaunet_1024' --image-size 1024 --epochs 2 --batch 8 --val-batch 4 --num-workers 4 --num-queries 100 --eval-every 1 --device cuda --amp --max-train-steps ${MAX_TRAIN_STEPS} --max-val-images ${MAX_VAL_IMAGES}"

runner_log "${MODE}" "${RUN_LOG}" "[custom-unet-canary] done"
