#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
source "${SCRIPT_DIR}/common_runner.sh"

DATASET_ROOT="${PROJECT_ROOT}/magformer_datasets/0831_1K"
OUTPUT_ROOT="${REPO_ROOT}/output/experiments/0831_1k_20ep_1024_lightdepth_stage_b"
MODE="run"
SMOKE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --smoke) SMOKE=1; shift ;;
    --run) MODE="run"; shift ;;
    --dry-run) MODE="dry-run"; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

mkdir -p "${OUTPUT_ROOT}"
RUN_ALL_LOG="${OUTPUT_ROOT}/run_all.log"
if [[ "${MODE}" == "run" ]]; then
  : > "${RUN_ALL_LOG}"
fi

runner_log "${MODE}" "${RUN_ALL_LOG}" "[0831-1k-20ep-1024-lightdepth-stage-b-all] mode=${MODE}"
runner_log "${MODE}" "${RUN_ALL_LOG}" "[0831-1k-20ep-1024-lightdepth-stage-b-all] smoke=${SMOKE}"
runner_log "${MODE}" "${RUN_ALL_LOG}" "[0831-1k-20ep-1024-lightdepth-stage-b-all] dataset_root=${DATASET_ROOT}"
runner_log "${MODE}" "${RUN_ALL_LOG}" "[0831-1k-20ep-1024-lightdepth-stage-b-all] output_root=${OUTPUT_ROOT}"

SMOKE_FLAG=""
if [[ "${SMOKE}" == "1" ]]; then
  SMOKE_FLAG="--smoke"
fi

run_summary() {
  runner_exec "${MODE}" "${RUN_ALL_LOG}" "cd '${REPO_ROOT}' && python scripts/experiments/summarize_suite.py --output-root '${OUTPUT_ROOT}' --write"
}

run_visuals() {
  local summary_json="${OUTPUT_ROOT}/summary_$(basename "${OUTPUT_ROOT}").json"
  runner_exec "${MODE}" "${RUN_ALL_LOG}" "cd '${REPO_ROOT}' && conda run -n magformer python scripts/experiments/visualize_suite.py --dataset-root '${DATASET_ROOT}' --output-root '${OUTPUT_ROOT}' --summary '${summary_json}' --num-images 20 --score-threshold 0.5"
}

ensure_stage_a_reference() {
  local src_best="${REPO_ROOT}/output/experiments/0831_1k_20ep_1024_lightdepth_stage_a/magformer_lightdepth_mobilenetv3_directadd_edge"
  local src_rgb="${REPO_ROOT}/output/experiments/0831_1k_20ep_1024_depth_revisit/magformer_nodpth_ref"
  local dst_best="${OUTPUT_ROOT}/magformer_lightdepth_mobilenetv3_directadd_edge"
  local dst_rgb="${OUTPUT_ROOT}/magformer_nodpth_ref"
  if [[ "${MODE}" != "run" ]]; then
    runner_log "${MODE}" "${RUN_ALL_LOG}" "+ ln -sfn '${src_best}' '${dst_best}'"
    runner_log "${MODE}" "${RUN_ALL_LOG}" "+ ln -sfn '${src_rgb}' '${dst_rgb}'"
    return 0
  fi
  [[ -d "${src_best}" ]] || { echo "Missing Stage A best source: ${src_best}" >&2; exit 1; }
  [[ -d "${src_rgb}" ]] || { echo "Missing RGB-only source: ${src_rgb}" >&2; exit 1; }
  ln -sfn "${src_best}" "${dst_best}"
  ln -sfn "${src_rgb}" "${dst_rgb}"
}

run_candidate() {
  local variant="$1"
  runner_log "${MODE}" "${RUN_ALL_LOG}" "[0831-1k-20ep-1024-lightdepth-stage-b-all] START ${variant}"
  runner_exec "${MODE}" "${RUN_ALL_LOG}" "bash '${SCRIPT_DIR}/run_0831_1k_20ep_1024_lightdepth_stage_b_magformer.sh' --dataset-root '${DATASET_ROOT}' --output-root '${OUTPUT_ROOT}' --variant '${variant}' ${SMOKE_FLAG} --${MODE}"
  runner_log "${MODE}" "${RUN_ALL_LOG}" "[0831-1k-20ep-1024-lightdepth-stage-b-all] END ${variant}"
  run_summary
}

ensure_stage_a_reference
run_summary

run_candidate "mobilenetv3_crossattn_edge"
run_candidate "mobilenetv3_crossattn_edge_validhole"
run_candidate "mobilenetv3_crossattn_edge_validhole_variance"

run_visuals
runner_log "${MODE}" "${RUN_ALL_LOG}" "[0831-1k-20ep-1024-lightdepth-stage-b-all] done"
