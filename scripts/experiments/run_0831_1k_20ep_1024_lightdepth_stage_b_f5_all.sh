#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
source "${SCRIPT_DIR}/common_runner.sh"

DATASET_ROOT="${PROJECT_ROOT}/magformer_datasets/0831_1K"
OUTPUT_ROOT="${REPO_ROOT}/output/experiments/0831_1k_20ep_1024_lightdepth_stage_b_f5"
MODE="run"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
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

runner_log "${MODE}" "${RUN_ALL_LOG}" "[0831-1k-20ep-1024-lightdepth-stage-b-f5-all] mode=${MODE}"
runner_log "${MODE}" "${RUN_ALL_LOG}" "[0831-1k-20ep-1024-lightdepth-stage-b-f5-all] dataset_root=${DATASET_ROOT}"
runner_log "${MODE}" "${RUN_ALL_LOG}" "[0831-1k-20ep-1024-lightdepth-stage-b-f5-all] output_root=${OUTPUT_ROOT}"

run_summary() {
  runner_exec "${MODE}" "${RUN_ALL_LOG}" "cd '${REPO_ROOT}' && python scripts/experiments/summarize_suite.py --output-root '${OUTPUT_ROOT}' --write"
}

ensure_reference_anchors() {
  local src_rgb="${REPO_ROOT}/output/experiments/0831_1k_20ep_1024_depth_revisit/magformer_nodpth_ref"
  local src_convnext="${REPO_ROOT}/output/experiments/0831_1k_20ep_1024_lightdepth_stage_a/magformer_lightdepth_convnextlite_spatialgate_edge_validhole"
  local src_mobilenet="${REPO_ROOT}/output/experiments/0831_1k_20ep_1024_lightdepth_stage_a/magformer_lightdepth_mobilenetv3_spatialgate_edge_validhole"
  local dst_rgb="${OUTPUT_ROOT}/magformer_nodpth_ref"
  local dst_convnext="${OUTPUT_ROOT}/magformer_lightdepth_convnextlite_spatialgate_edge_validhole"
  local dst_mobilenet="${OUTPUT_ROOT}/magformer_lightdepth_mobilenetv3_spatialgate_edge_validhole"
  if [[ "${MODE}" != "run" ]]; then
    runner_log "${MODE}" "${RUN_ALL_LOG}" "+ ln -sfn '${src_rgb}' '${dst_rgb}'"
    runner_log "${MODE}" "${RUN_ALL_LOG}" "+ ln -sfn '${src_convnext}' '${dst_convnext}'"
    runner_log "${MODE}" "${RUN_ALL_LOG}" "+ ln -sfn '${src_mobilenet}' '${dst_mobilenet}'"
    return 0
  fi
  [[ -d "${src_rgb}" ]] || { echo "Missing RGB-only source: ${src_rgb}" >&2; exit 1; }
  [[ -d "${src_convnext}" ]] || { echo "Missing ConvNeXt-lite anchor source: ${src_convnext}" >&2; exit 1; }
  [[ -d "${src_mobilenet}" ]] || { echo "Missing MobileNetV3 anchor source: ${src_mobilenet}" >&2; exit 1; }
  ln -sfn "${src_rgb}" "${dst_rgb}"
  ln -sfn "${src_convnext}" "${dst_convnext}"
  ln -sfn "${src_mobilenet}" "${dst_mobilenet}"
}

run_candidate() {
  local variant="$1"
  runner_log "${MODE}" "${RUN_ALL_LOG}" "[0831-1k-20ep-1024-lightdepth-stage-b-f5-all] START ${variant}"
  runner_exec "${MODE}" "${RUN_ALL_LOG}" "bash '${SCRIPT_DIR}/run_0831_1k_20ep_1024_lightdepth_stage_b_f5_magformer.sh' --dataset-root '${DATASET_ROOT}' --output-root '${OUTPUT_ROOT}' --variant '${variant}' --${MODE}"
  runner_log "${MODE}" "${RUN_ALL_LOG}" "[0831-1k-20ep-1024-lightdepth-stage-b-f5-all] END ${variant}"
  run_summary
}

ensure_reference_anchors
run_summary

run_candidate "mobilenetv3_priorguidedcrossattn_edge"
run_candidate "mobilenetv3_priorguidedcrossattn_edge_validhole"
run_candidate "mobilenetv3_priorguidedcrossattn_edge_validhole_variance"
run_candidate "convnextlite_priorguidedcrossattn_edge"
run_candidate "convnextlite_priorguidedcrossattn_edge_validhole"
run_candidate "convnextlite_priorguidedcrossattn_edge_validhole_variance"

runner_log "${MODE}" "${RUN_ALL_LOG}" "[0831-1k-20ep-1024-lightdepth-stage-b-f5-all] done"
