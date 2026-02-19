#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"

DATASET_ROOT_DEFAULT="${PROJECT_ROOT}/magformer_datasets/0831_1K"
OUTPUT_ROOT_DEFAULT="${REPO_ROOT}/output/experiments/0831_1k_5k_scratch8"

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

run_cmd() {
  echo "+ $*"
  if [[ "${MODE}" == "run" ]]; then
    eval "$@"
  fi
}

echo "[0831-1k-5k-scratch8-all] mode=${MODE}"
echo "[0831-1k-5k-scratch8-all] dataset_root=${DATASET_ROOT}"
echo "[0831-1k-5k-scratch8-all] output_root=${OUTPUT_ROOT}"

run_cmd "bash '${SCRIPT_DIR}/run_0831_1k_5k_scratch_magformer.sh' --dataset-root '${DATASET_ROOT}' --output-root '${OUTPUT_ROOT}' --${MODE}"
run_cmd "bash '${SCRIPT_DIR}/run_0831_1k_5k_scratch_mgm_mask2former.sh' --dataset-root '${DATASET_ROOT}' --output-root '${OUTPUT_ROOT}' --${MODE}"
run_cmd "bash '${SCRIPT_DIR}/run_0831_1k_5k_scratch_msmformer.sh' --dataset-root '${DATASET_ROOT}' --output-root '${OUTPUT_ROOT}' --${MODE}"
run_cmd "bash '${SCRIPT_DIR}/run_0831_1k_5k_scratch_uoais.sh' --dataset-root '${DATASET_ROOT}' --output-root '${OUTPUT_ROOT}' --${MODE}"
run_cmd "bash '${SCRIPT_DIR}/run_0831_1k_5k_scratch_ucn.sh' --dataset-root '${DATASET_ROOT}' --output-root '${OUTPUT_ROOT}' --${MODE}"
run_cmd "bash '${SCRIPT_DIR}/run_0831_1k_5k_scratch_official_mask2former.sh' --dataset-root '${DATASET_ROOT}' --output-root '${OUTPUT_ROOT}' --${MODE}"
run_cmd "bash '${SCRIPT_DIR}/run_0831_1k_5k_scratch_maskrcnn.sh' --dataset-root '${DATASET_ROOT}' --output-root '${OUTPUT_ROOT}' --${MODE}"
run_cmd "bash '${SCRIPT_DIR}/run_0831_1k_5k_scratch_yolov8_seg.sh' --dataset-root '${DATASET_ROOT}' --output-root '${OUTPUT_ROOT}' --${MODE}"

run_cmd "cd '${REPO_ROOT}' && python scripts/experiments/summarize_0831_1k_5k_scratch8.py --output-root '${OUTPUT_ROOT}' --write"

echo "[0831-1k-5k-scratch8-all] done"
