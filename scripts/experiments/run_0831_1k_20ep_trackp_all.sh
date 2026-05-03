#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
source "${SCRIPT_DIR}/common_runner.sh"

REGISTER="0831"

DATASET_ROOT_DEFAULT="${PROJECT_ROOT}/magformer_datasets/0831_1K"
OUTPUT_ROOT_DEFAULT_RUN="${REPO_ROOT}/output/experiments/0831_1k_20ep_trackp"

DATASET_ROOT="${DATASET_ROOT_DEFAULT}"
OUTPUT_ROOT="${OUTPUT_ROOT_DEFAULT_RUN}"
MODE="run"
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

runner_log "${MODE}" "${RUN_ALL_LOG}" "[0831-1k-20ep-trackp-all] mode=${MODE}"
runner_log "${MODE}" "${RUN_ALL_LOG}" "[0831-1k-20ep-trackp-all] dataset_root=${DATASET_ROOT}"
runner_log "${MODE}" "${RUN_ALL_LOG}" "[0831-1k-20ep-trackp-all] output_root=${OUTPUT_ROOT}"

MODELS=(
  "magformer:run_ecc_20ep_trackp_magformer.sh"
  "mgm_mask2former:run_ecc_20ep_trackp_mgm_mask2former.sh"
  "msmformer:run_ecc_20ep_trackp_msmformer.sh"
  "uoais:run_ecc_20ep_trackp_uoais.sh"
  "ucn:run_ecc_20ep_trackp_ucn.sh"
  "official_mask2former:run_ecc_20ep_trackp_official_mask2former.sh"
  "maskrcnn:run_ecc_20ep_trackp_maskrcnn.sh"
  "yolov8_seg:run_ecc_20ep_trackp_yolov8_seg.sh"
)
CANDIDATES=("C1" "C2" "C3" "C4")

run_summary() {
  runner_exec "${MODE}" "${RUN_ALL_LOG}" "cd '${REPO_ROOT}' && python scripts/experiments/summarize_suite.py --output-root '${OUTPUT_ROOT}' --write"
}

run_model_once() {
  local model_id="$1"
  local runner="$2"
  local candidate="$3"
  local run_tag="$4"

  local done_marker
  if [[ "${run_tag}" == "final" ]]; then
    done_marker="${OUTPUT_ROOT}/${model_id}/metrics.cocoeval.json"
  else
    done_marker="${OUTPUT_ROOT}/_tuning/${model_id}/${candidate}/metrics.cocoeval.json"
  fi

  if [[ -f "${done_marker}" ]]; then
    runner_log "${MODE}" "${RUN_ALL_LOG}" "[0831-1k-20ep-trackp-all] SKIP ${model_id} ${run_tag} ${candidate} (already done)"
    return 0
  fi

  runner_log "${MODE}" "${RUN_ALL_LOG}" "[0831-1k-20ep-trackp-all] START ${model_id} tag=${run_tag} candidate=${candidate}"
  runner_exec "${MODE}" "${RUN_ALL_LOG}" "bash '${SCRIPT_DIR}/${runner}' --register '${REGISTER}' --dataset-root '${DATASET_ROOT}' --output-root '${OUTPUT_ROOT}' --candidate-id '${candidate}' --run-tag '${run_tag}' --${MODE}"
  runner_log "${MODE}" "${RUN_ALL_LOG}" "[0831-1k-20ep-trackp-all] END ${model_id} tag=${run_tag} candidate=${candidate}"
}

select_candidate() {
  local model_id="$1"
  runner_exec "${MODE}" "${RUN_ALL_LOG}" "cd '${REPO_ROOT}' && conda run -n magformer python scripts/analysis/select_best_candidate.py --output-root '${OUTPUT_ROOT}' --model-id '${model_id}' --candidates 'C1,C2,C3,C4' --write"
}

read_best_candidate() {
  local model_id="$1"
  local p="${OUTPUT_ROOT}/_tuning/${model_id}/best_candidate.json"
  if [[ ! -f "${p}" ]]; then
    echo "C1"
    return 0
  fi
  python3 -c "import json;print(json.load(open('${p}','r',encoding='utf-8')).get('best_candidate','C1'))"
}


# Stage 1: sweep 4 candidates (5ep)
for item in "${MODELS[@]}"; do
  model_id="${item%%:*}"
  runner="${item##*:}"
  for candidate in "${CANDIDATES[@]}"; do
    run_model_once "${model_id}" "${runner}" "${candidate}" "sweep"
  done
  select_candidate "${model_id}"
  run_summary
done

# Stage 2: final 20ep with best candidate
for item in "${MODELS[@]}"; do
  model_id="${item%%:*}"
  runner="${item##*:}"
  best_candidate="$(read_best_candidate "${model_id}")"
  runner_log "${MODE}" "${RUN_ALL_LOG}" "[0831-1k-20ep-trackp-all] BEST ${model_id}=${best_candidate}"
  run_model_once "${model_id}" "${runner}" "${best_candidate}" "final"
  run_summary
done

run_summary

SUMMARY_JSON="${OUTPUT_ROOT}/summary_$(basename "${OUTPUT_ROOT}").json"
runner_exec "${MODE}" "${RUN_ALL_LOG}" "cd '${REPO_ROOT}' && conda run -n magformer python scripts/analysis/check_trackp_constraints.py '${SUMMARY_JSON}'"

runner_log "${MODE}" "${RUN_ALL_LOG}" "[0831-1k-20ep-trackp-all] done"
