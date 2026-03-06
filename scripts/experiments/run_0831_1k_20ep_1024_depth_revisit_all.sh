#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
source "${SCRIPT_DIR}/common_runner.sh"

DATASET_ROOT_DEFAULT="${PROJECT_ROOT}/magformer_datasets/0831_1K"
OUTPUT_ROOT_DEFAULT_RUN="${REPO_ROOT}/output/experiments/0831_1k_20ep_1024_depth_revisit"
OUTPUT_ROOT_DEFAULT_SMOKE="${REPO_ROOT}/output/experiments/0831_1k_20ep_1024_depth_revisit_smoke"

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
    --run)
      MODE="run"
      shift
      ;;
    --dry-run)
      MODE="dry-run"
      shift
      ;;
    --smoke)
      SMOKE=1
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

runner_log "${MODE}" "${RUN_ALL_LOG}" "[0831-1k-20ep-1024-depth-revisit-all] mode=${MODE}"
runner_log "${MODE}" "${RUN_ALL_LOG}" "[0831-1k-20ep-1024-depth-revisit-all] smoke=${SMOKE}"
runner_log "${MODE}" "${RUN_ALL_LOG}" "[0831-1k-20ep-1024-depth-revisit-all] dataset_root=${DATASET_ROOT}"
runner_log "${MODE}" "${RUN_ALL_LOG}" "[0831-1k-20ep-1024-depth-revisit-all] output_root=${OUTPUT_ROOT}"

run_summary() {
  runner_exec "${MODE}" "${RUN_ALL_LOG}" "cd '${REPO_ROOT}' && python scripts/experiments/summarize_suite.py --output-root '${OUTPUT_ROOT}' --write"
}

run_visuals() {
  local summary_json="${OUTPUT_ROOT}/summary_$(basename "${OUTPUT_ROOT}").json"
  runner_exec "${MODE}" "${RUN_ALL_LOG}" "cd '${REPO_ROOT}' && conda run -n magformer python scripts/experiments/visualize_suite.py --dataset-root '${DATASET_ROOT}' --output-root '${OUTPUT_ROOT}' --summary '${summary_json}' --num-images 50 --score-threshold 0.5"
  runner_exec "${MODE}" "${RUN_ALL_LOG}" "cd '${REPO_ROOT}' && conda run -n magformer python scripts/analysis/write_revisit_comparison.py --summary '${summary_json}'"
}

run_cmd() {
  local tag="$1"
  local cmd="$2"
  runner_log "${MODE}" "${RUN_ALL_LOG}" "[0831-1k-20ep-1024-depth-revisit-all] START ${tag}"
  runner_exec "${MODE}" "${RUN_ALL_LOG}" "${cmd}"
  runner_log "${MODE}" "${RUN_ALL_LOG}" "[0831-1k-20ep-1024-depth-revisit-all] END ${tag}"
  run_summary
}

SMOKE_FLAG=""
if [[ "${SMOKE}" == "1" ]]; then
  SMOKE_FLAG="--smoke"
fi

run_cmd "magformer_nodpth_ref" "bash '${SCRIPT_DIR}/run_0831_1k_20ep_1024_revisit_magformer.sh' --dataset-root '${DATASET_ROOT}' --output-root '${OUTPUT_ROOT}' --variant nodpth_ref ${SMOKE_FLAG} --${MODE}"
run_cmd "magformer_depthnorm_on" "bash '${SCRIPT_DIR}/run_0831_1k_20ep_1024_revisit_magformer.sh' --dataset-root '${DATASET_ROOT}' --output-root '${OUTPUT_ROOT}' --variant depthnorm_on ${SMOKE_FLAG} --${MODE}"
run_cmd "mgm_mask2former_nodpth_ref" "bash '${SCRIPT_DIR}/run_0831_1k_20ep_1024_revisit_mgm_mask2former.sh' --dataset-root '${DATASET_ROOT}' --output-root '${OUTPUT_ROOT}' --variant nodpth_ref ${SMOKE_FLAG} --${MODE}"
run_cmd "mgm_mask2former_depthnorm_on" "bash '${SCRIPT_DIR}/run_0831_1k_20ep_1024_revisit_mgm_mask2former.sh' --dataset-root '${DATASET_ROOT}' --output-root '${OUTPUT_ROOT}' --variant depthnorm_on ${SMOKE_FLAG} --${MODE}"

run_cmd "official_mask2former_1024" "bash '${SCRIPT_DIR}/run_0831_1k_20ep_scratch_official_mask2former.sh' --dataset-root '${DATASET_ROOT}' --output-root '${OUTPUT_ROOT}' --candidate-id C1 --run-tag final --image-size 1024 ${SMOKE_FLAG} --${MODE}"
run_cmd "maskrcnn_1024" "bash '${SCRIPT_DIR}/run_0831_1k_20ep_scratch_maskrcnn.sh' --dataset-root '${DATASET_ROOT}' --output-root '${OUTPUT_ROOT}' --candidate-id C1 --run-tag final --image-size 1024 ${SMOKE_FLAG} --${MODE}"
run_cmd "msmformer_1024" "bash '${SCRIPT_DIR}/run_0831_1k_20ep_scratch_msmformer.sh' --dataset-root '${DATASET_ROOT}' --output-root '${OUTPUT_ROOT}' --candidate-id C1 --run-tag final --image-size 1024 ${SMOKE_FLAG} --${MODE}"
run_cmd "ucn_1024" "bash '${SCRIPT_DIR}/run_0831_1k_20ep_scratch_ucn.sh' --dataset-root '${DATASET_ROOT}' --output-root '${OUTPUT_ROOT}' --candidate-id C1 --run-tag final --image-size 1024 ${SMOKE_FLAG} --${MODE}"
run_cmd "uoais_1024" "bash '${SCRIPT_DIR}/run_0831_1k_20ep_scratch_uoais.sh' --dataset-root '${DATASET_ROOT}' --output-root '${OUTPUT_ROOT}' --candidate-id C1 --run-tag final --image-size 1024 ${SMOKE_FLAG} --${MODE}"
run_cmd "yolov8_seg_1024" "bash '${SCRIPT_DIR}/run_0831_1k_20ep_scratch_yolov8_seg.sh' --dataset-root '${DATASET_ROOT}' --output-root '${OUTPUT_ROOT}' --candidate-id C1 --run-tag final --image-size 1024 ${SMOKE_FLAG} --${MODE}"

run_cmd "unet_boundary_inst" "bash '${SCRIPT_DIR}/run_0831_1k_20ep_1024_revisit_unet_boundary_inst.sh' --dataset-root '${DATASET_ROOT}' --output-root '${OUTPUT_ROOT}' ${SMOKE_FLAG} --${MODE}"
run_cmd "unetpp_boundary_inst" "bash '${SCRIPT_DIR}/run_0831_1k_20ep_1024_revisit_unetpp_boundary_inst.sh' --dataset-root '${DATASET_ROOT}' --output-root '${OUTPUT_ROOT}' ${SMOKE_FLAG} --${MODE}"
run_cmd "unet_distance_inst" "bash '${SCRIPT_DIR}/run_0831_1k_20ep_1024_revisit_unet_distance_inst.sh' --dataset-root '${DATASET_ROOT}' --output-root '${OUTPUT_ROOT}' ${SMOKE_FLAG} --${MODE}"

run_summary
run_visuals
runner_log "${MODE}" "${RUN_ALL_LOG}" "[0831-1k-20ep-1024-depth-revisit-all] done"
