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
QUEUE_TAG="20260429_repaired_unet_launcher"
GPU0=0
GPU1=1
FREE_THRESHOLD_MB=60000
SESSION_PREFIX="repaired-unet-20260429"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --register) REGISTER="$2"; shift 2 ;;
    --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
    --output-base) OUTPUT_BASE="$2"; shift 2 ;;
    --queue-tag) QUEUE_TAG="$2"; shift 2 ;;
    --gpu0) GPU0="$2"; shift 2 ;;
    --gpu1) GPU1="$2"; shift 2 ;;
    --free-threshold-mb) FREE_THRESHOLD_MB="$2"; shift 2 ;;
    --session-prefix) SESSION_PREFIX="$2"; shift 2 ;;
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

RUN_LOG="$(runner_setup_log "${OUTPUT_BASE}/${QUEUE_TAG}" "${MODE}")"

gpu_free_mb_by_id() {
  local gpu_id="$1"
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "0"
    return 0
  fi
  nvidia-smi --id="${gpu_id}" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d ' ' || echo "0"
}

start_tmux_job() {
  local session="$1"
  local gpu="$2"
  local script="$3"
  local label="$4"
  local cmd
  cmd="cd '${REPO_ROOT}' && export PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=${gpu} OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 OPENCV_FOR_THREADS_NUM=4 MALLOC_ARENA_MAX=2 && bash '${script}' --register '${REGISTER}' --dataset-root '${DATASET_ROOT}' --output-base '${OUTPUT_BASE}' --gpu '${gpu}' --run"
  runner_log "${MODE}" "${RUN_LOG}" "[repaired-launch] ${label}: session=${session} gpu=${gpu}"
  runner_log "${MODE}" "${RUN_LOG}" "+ tmux new-session -d -s ${session} ${cmd}"
  if [[ "${MODE}" == "run" ]]; then
    if tmux has-session -t "${session}" 2>/dev/null; then
      runner_log "${MODE}" "${RUN_LOG}" "[repaired-launch] skip existing session ${session}"
    else
      tmux new-session -d -s "${session}" "${cmd}"
    fi
  fi
}

GPU0_FREE="$(gpu_free_mb_by_id "${GPU0}")"
GPU1_FREE="$(gpu_free_mb_by_id "${GPU1}")"
runner_log "${MODE}" "${RUN_LOG}" "[repaired-launch] gpu0=${GPU0} free_mb=${GPU0_FREE} gpu1=${GPU1} free_mb=${GPU1_FREE} threshold_mb=${FREE_THRESHOLD_MB}"
runner_log_launch_guard_snapshot "${MODE}" "${RUN_LOG}" "repaired-launcher"

CELLPOSE_GPU="${GPU0}"
IAUNET_GPU="${GPU0}"
if [[ "${GPU1_FREE}" -ge "${FREE_THRESHOLD_MB}" && "${GPU0_FREE}" -ge "${FREE_THRESHOLD_MB}" ]]; then
  CELLPOSE_GPU="${GPU0}"
  IAUNET_GPU="${GPU1}"
elif [[ "${GPU1_FREE}" -ge "${FREE_THRESHOLD_MB}" ]]; then
  CELLPOSE_GPU="${GPU1}"
  IAUNET_GPU="${GPU1}"
else
  CELLPOSE_GPU="${GPU0}"
  IAUNET_GPU="${GPU0}"
fi

start_tmux_job "${SESSION_PREFIX}-cellpose-gpu${CELLPOSE_GPU}" "${CELLPOSE_GPU}" "${REPO_ROOT}/scripts/experiments/run_20260429_repaired_cellpose_100ep.sh" "cellpose"
start_tmux_job "${SESSION_PREFIX}-iaunet-gpu${IAUNET_GPU}" "${IAUNET_GPU}" "${REPO_ROOT}/scripts/experiments/run_20260429_repaired_iaunet_100ep.sh" "iaunet"

runner_log "${MODE}" "${RUN_LOG}" "[repaired-launch] done"
