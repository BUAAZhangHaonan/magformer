#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
source "${SCRIPT_DIR}/common_runner.sh"

DATASET_ROOT_DEFAULT="${PROJECT_ROOT}/magformer_datasets/0831_1K"
OUTPUT_ROOT_DEFAULT_RUN="${REPO_ROOT}/output/experiments/0831_1k_5k_scratch8"
OUTPUT_ROOT_DEFAULT_SMOKE="${REPO_ROOT}/output/experiments/0831_1k_5k_scratch8_smoke"

DATASET_ROOT="${DATASET_ROOT_DEFAULT}"
OUTPUT_ROOT="${OUTPUT_ROOT_DEFAULT_RUN}"
MODE="run"
SMOKE=0
OUTPUT_ROOT_SET=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-root)
      DATASET_ROOT="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      OUTPUT_ROOT_SET=1
      shift 2
      ;;
    --smoke)
      SMOKE=1
      shift
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

if [[ "${SMOKE}" == "1" && "${OUTPUT_ROOT_SET}" == "0" ]]; then
  OUTPUT_ROOT="${OUTPUT_ROOT_DEFAULT_SMOKE}"
fi

mkdir -p "${OUTPUT_ROOT}"
RUN_ALL_LOG="${OUTPUT_ROOT}/run_all.log"
if [[ "${MODE}" == "run" ]]; then
  : > "${RUN_ALL_LOG}"
fi

runner_log "${MODE}" "${RUN_ALL_LOG}" "[0831-1k-5k-scratch8-all] mode=${MODE}"
runner_log "${MODE}" "${RUN_ALL_LOG}" "[0831-1k-5k-scratch8-all] smoke=${SMOKE}"
runner_log "${MODE}" "${RUN_ALL_LOG}" "[0831-1k-5k-scratch8-all] dataset_root=${DATASET_ROOT}"
runner_log "${MODE}" "${RUN_ALL_LOG}" "[0831-1k-5k-scratch8-all] output_root=${OUTPUT_ROOT}"

run_model() {
  local model_id="$1"
  local runner="$2"
  local model_out="${OUTPUT_ROOT}/${model_id}"
  local done_marker="${model_out}/metrics.cocoeval.json"
  local smoke_flag=""
  if [[ "${SMOKE}" == "1" ]]; then
    smoke_flag="--smoke"
  fi
  if [[ -f "${done_marker}" ]]; then
    runner_log "${MODE}" "${RUN_ALL_LOG}" "[0831-1k-5k-scratch8-all] SKIP ${model_id} (already has metrics.cocoeval.json)"
    return 0
  fi
  if [[ "${MODE}" == "run" ]]; then
    rm -rf "${model_out}"
  fi
  runner_log "${MODE}" "${RUN_ALL_LOG}" "[0831-1k-5k-scratch8-all] START ${model_id}"
  runner_exec "${MODE}" "${RUN_ALL_LOG}" "bash '${SCRIPT_DIR}/${runner}' --dataset-root '${DATASET_ROOT}' --output-root '${OUTPUT_ROOT}' ${smoke_flag} --${MODE}"
  runner_exec "${MODE}" "${RUN_ALL_LOG}" "cd '${REPO_ROOT}' && python scripts/experiments/summarize_0831_1k_5k_scratch8.py --output-root '${OUTPUT_ROOT}' --write"
  runner_exec "${MODE}" "${RUN_ALL_LOG}" "cd '${REPO_ROOT}' && conda run -n magformer python scripts/experiments/visualize_0831_1k_5k_scratch8.py --dataset-root '${DATASET_ROOT}' --output-root '${OUTPUT_ROOT}' --num-images 50 --score-threshold 0.5"
  runner_log "${MODE}" "${RUN_ALL_LOG}" "[0831-1k-5k-scratch8-all] END ${model_id}"
}

run_model "magformer_scratch" "run_0831_1k_5k_scratch_magformer.sh"
run_model "mgm_mask2former_scratch" "run_0831_1k_5k_scratch_mgm_mask2former.sh"
run_model "msmformer_scratch" "run_0831_1k_5k_scratch_msmformer.sh"
run_model "uoais_scratch" "run_0831_1k_5k_scratch_uoais.sh"
run_model "ucn_scratch" "run_0831_1k_5k_scratch_ucn.sh"
run_model "official_mask2former_scratch" "run_0831_1k_5k_scratch_official_mask2former.sh"
run_model "maskrcnn_scratch" "run_0831_1k_5k_scratch_maskrcnn.sh"
run_model "yolov8_seg_scratch" "run_0831_1k_5k_scratch_yolov8_seg.sh"

runner_exec "${MODE}" "${RUN_ALL_LOG}" "cd '${REPO_ROOT}' && python scripts/experiments/summarize_0831_1k_5k_scratch8.py --output-root '${OUTPUT_ROOT}' --write"
runner_exec "${MODE}" "${RUN_ALL_LOG}" "cd '${REPO_ROOT}' && conda run -n magformer python scripts/experiments/visualize_0831_1k_5k_scratch8.py --dataset-root '${DATASET_ROOT}' --output-root '${OUTPUT_ROOT}' --num-images 50 --score-threshold 0.5"

runner_log "${MODE}" "${RUN_ALL_LOG}" "[0831-1k-5k-scratch8-all] done"
