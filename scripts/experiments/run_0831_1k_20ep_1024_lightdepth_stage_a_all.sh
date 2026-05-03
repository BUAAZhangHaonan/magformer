#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
source "${SCRIPT_DIR}/common_runner.sh"

DATASET_ROOT="${PROJECT_ROOT}/magformer_datasets/0831_1K"
OUTPUT_ROOT="${REPO_ROOT}/output/experiments/0831_1k_20ep_1024_lightdepth_stage_a"
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
RUN_ALL_LOG="${OUTPUT_ROOT}/run_all.log"
if [[ "${MODE}" == "run" ]]; then
  : > "${RUN_ALL_LOG}"
fi

runner_log "${MODE}" "${RUN_ALL_LOG}" "[0831-1k-20ep-1024-lightdepth-stage-a-all] mode=${MODE}"
runner_log "${MODE}" "${RUN_ALL_LOG}" "[0831-1k-20ep-1024-lightdepth-stage-a-all] dataset_root=${DATASET_ROOT}"
runner_log "${MODE}" "${RUN_ALL_LOG}" "[0831-1k-20ep-1024-lightdepth-stage-a-all] output_root=${OUTPUT_ROOT}"

run_summary() {
  runner_exec "${MODE}" "${RUN_ALL_LOG}" "cd '${REPO_ROOT}' && python scripts/experiments/summarize_suite.py --output-root '${OUTPUT_ROOT}' --write"
}

run_visuals() {
  local summary_json="${OUTPUT_ROOT}/summary_$(basename "${OUTPUT_ROOT}").json"
  runner_exec "${MODE}" "${RUN_ALL_LOG}" "cd '${REPO_ROOT}' && conda run -n magformer python scripts/experiments/visualize_suite.py --dataset-root '${DATASET_ROOT}' --output-root '${OUTPUT_ROOT}' --summary '${summary_json}' --num-images 20 --score-threshold 0.5"
}

ensure_rgb_only_baseline() {
  local src="${REPO_ROOT}/output/experiments/0831_1k_20ep_1024_depth_revisit/magformer_nodpth_ref"
  local dst="${OUTPUT_ROOT}/magformer_nodpth_ref"
  runner_log "${MODE}" "${RUN_ALL_LOG}" "[0831-1k-20ep-1024-lightdepth-stage-a-all] baseline_src=${src}"
  if [[ "${MODE}" != "run" ]]; then
    runner_log "${MODE}" "${RUN_ALL_LOG}" "+ ln -sfn '${src}' '${dst}'"
    return 0
  fi
  if [[ ! -d "${src}" ]]; then
    echo "Baseline source missing: ${src}" >&2
    exit 1
  fi
  ln -sfn "${src}" "${dst}"
}

run_candidate() {
  local variant="$1"
  runner_log "${MODE}" "${RUN_ALL_LOG}" "[0831-1k-20ep-1024-lightdepth-stage-a-all] START ${variant}"
  runner_exec "${MODE}" "${RUN_ALL_LOG}" "bash '${SCRIPT_DIR}/run_0831_1k_20ep_1024_lightdepth_stage_a_magformer.sh' --dataset-root '${DATASET_ROOT}' --output-root '${OUTPUT_ROOT}' --variant '${variant}' --${MODE}"
  runner_log "${MODE}" "${RUN_ALL_LOG}" "[0831-1k-20ep-1024-lightdepth-stage-a-all] END ${variant}"
  run_summary
}

ensure_rgb_only_baseline
run_summary

while IFS= read -r variant; do
  [[ -n "${variant}" ]] || continue
  run_candidate "${variant}"
done < <(cd "${REPO_ROOT}" && python scripts/experiments/lightdepth_roster.py --status active --field variant)

run_visuals
runner_log "${MODE}" "${RUN_ALL_LOG}" "[0831-1k-20ep-1024-lightdepth-stage-a-all] done"
