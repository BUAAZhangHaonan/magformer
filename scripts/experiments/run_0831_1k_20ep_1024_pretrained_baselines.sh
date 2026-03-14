#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
source "${SCRIPT_DIR}/common_runner.sh"

DATASET_ROOT_DEFAULT="${PROJECT_ROOT}/magformer_datasets/0831_1K"
OUTPUT_ROOT_DEFAULT="${REPO_ROOT}/output/experiments/0831_1k_20ep_1024_depth_revisit"

DATASET_ROOT="${DATASET_ROOT_DEFAULT}"
OUTPUT_ROOT="${OUTPUT_ROOT_DEFAULT}"
MODE="run"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-root)
      DATASET_ROOT="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
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

mkdir -p "${OUTPUT_ROOT}"
RUN_LOG="${OUTPUT_ROOT}/run_pretrained_baselines.log"
if [[ "${MODE}" == "run" ]]; then
  : > "${RUN_LOG}"
fi

run_summary() {
  runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && python scripts/experiments/summarize_suite.py --output-root '${OUTPUT_ROOT}' --write"
}

run_cmd() {
  local tag="$1"
  local cmd="$2"
  runner_log "${MODE}" "${RUN_LOG}" "[0831-1k-20ep-1024-pretrained] START ${tag}"
  runner_exec "${MODE}" "${RUN_LOG}" "${cmd}"
  runner_log "${MODE}" "${RUN_LOG}" "[0831-1k-20ep-1024-pretrained] END ${tag}"
  run_summary
}

run_cmd "maskrcnn_pretrained_1024" "bash '${SCRIPT_DIR}/run_0831_1k_20ep_scratch_maskrcnn.sh' --dataset-root '${DATASET_ROOT}' --output-root '${OUTPUT_ROOT}' --candidate-id C1 --run-tag final --image-size 1024 --pretrained --${MODE}"
run_cmd "official_mask2former_pretrained_1024" "bash '${SCRIPT_DIR}/run_0831_1k_20ep_scratch_official_mask2former.sh' --dataset-root '${DATASET_ROOT}' --output-root '${OUTPUT_ROOT}' --candidate-id C1 --run-tag final --image-size 1024 --pretrained --${MODE}"
run_cmd "yolov8_seg_pretrained_1024" "bash '${SCRIPT_DIR}/run_0831_1k_20ep_scratch_yolov8_seg.sh' --dataset-root '${DATASET_ROOT}' --output-root '${OUTPUT_ROOT}' --candidate-id C1 --run-tag final --image-size 1024 --pretrained --${MODE}"

run_summary
runner_log "${MODE}" "${RUN_LOG}" "[0831-1k-20ep-1024-pretrained] done"
