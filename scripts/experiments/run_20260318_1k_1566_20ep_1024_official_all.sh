#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
source "${SCRIPT_DIR}/common_runner.sh"

REGISTER="20260318_1K_1566"
DATASET_ROOT="${PROJECT_ROOT}/magformer_datasets/20260318_1K_1566"
OUTPUT_ROOT="${REPO_ROOT}/output/experiments/20260318_1k_1566_20ep_1024_official"
MODE="run"
WAIT_FREE_GPU_MB="${WAIT_FREE_GPU_MB:-45000}"
WAIT_CHECK_SEC="${WAIT_CHECK_SEC:-300}"

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

TRACK_NAME="$(basename "${OUTPUT_ROOT}")"
SUMMARY_JSON="${OUTPUT_ROOT}/summary_${TRACK_NAME}.json"
EXTENDED_JSON="${OUTPUT_ROOT}/extended_metrics_table.json"
EXTENDED_CSV="${OUTPUT_ROOT}/extended_metrics_table.csv"
EXTENDED_MD="${OUTPUT_ROOT}/extended_metrics_table.md"

runner_log "${MODE}" "${RUN_ALL_LOG}" "[20260318-1k-1566-20ep-1024-official-all] mode=${MODE}"
runner_log "${MODE}" "${RUN_ALL_LOG}" "[20260318-1k-1566-20ep-1024-official-all] register=${REGISTER}"
runner_log "${MODE}" "${RUN_ALL_LOG}" "[20260318-1k-1566-20ep-1024-official-all] dataset_root=${DATASET_ROOT}"
runner_log "${MODE}" "${RUN_ALL_LOG}" "[20260318-1k-1566-20ep-1024-official-all] output_root=${OUTPUT_ROOT}"
runner_log "${MODE}" "${RUN_ALL_LOG}" "[20260318-1k-1566-20ep-1024-official-all] wait_free_gpu_mb=${WAIT_FREE_GPU_MB}"

run_summary() {
  runner_exec "${MODE}" "${RUN_ALL_LOG}" "cd '${REPO_ROOT}' && python scripts/experiments/summarize_suite.py --output-root '${OUTPUT_ROOT}' --write"
}

run_benchmarks() {
  runner_exec "${MODE}" "${RUN_ALL_LOG}" "cd '${REPO_ROOT}' && conda run -n magformer python scripts/analysis/benchmark_inference_suite.py --output-root '${OUTPUT_ROOT}' --dataset-root '${DATASET_ROOT}' --summary '${SUMMARY_JSON}' --continue-on-error"
}

run_visualizations() {
  runner_exec "${MODE}" "${RUN_ALL_LOG}" "cd '${REPO_ROOT}' && conda run -n magformer python scripts/experiments/visualize_suite.py --output-root '${OUTPUT_ROOT}' --dataset-root '${DATASET_ROOT}' --summary '${SUMMARY_JSON}' --num-images 50"
}

write_extended_metrics() {
  runner_exec "${MODE}" "${RUN_ALL_LOG}" "cd '${REPO_ROOT}' && conda run -n magformer python scripts/analysis/write_extended_metrics_table.py --summary '${SUMMARY_JSON}' --out-json '${EXTENDED_JSON}' --out-csv '${EXTENDED_CSV}' --out-md '${EXTENDED_MD}'"
}

while IFS=$'\t' read -r MODEL_ID CMD; do
  DONE_MARKER="${OUTPUT_ROOT}/${MODEL_ID}/metrics.cocoeval.json"
  if [[ -f "${DONE_MARKER}" ]]; then
    runner_log "${MODE}" "${RUN_ALL_LOG}" "[20260318-1k-1566-20ep-1024-official-all] SKIP ${MODEL_ID} (already done)"
    continue
  fi

  runner_wait_for_free_gpu_mb "${MODE}" "${RUN_ALL_LOG}" "${WAIT_FREE_GPU_MB}" "${WAIT_CHECK_SEC}" "${MODEL_ID}"
  runner_log "${MODE}" "${RUN_ALL_LOG}" "[20260318-1k-1566-20ep-1024-official-all] START ${MODEL_ID}"
  runner_exec "${MODE}" "${RUN_ALL_LOG}" "${CMD}"
  runner_log "${MODE}" "${RUN_ALL_LOG}" "[20260318-1k-1566-20ep-1024-official-all] END ${MODEL_ID}"
done < <(
  python "${SCRIPT_DIR}/official_roster.py" \
    --format commands \
    --register "${REGISTER}" \
    --dataset-root "${DATASET_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --mode "${MODE}"
)

run_summary
run_benchmarks
run_summary
run_visualizations
write_extended_metrics

runner_log "${MODE}" "${RUN_ALL_LOG}" "[20260318-1k-1566-20ep-1024-official-all] done"
