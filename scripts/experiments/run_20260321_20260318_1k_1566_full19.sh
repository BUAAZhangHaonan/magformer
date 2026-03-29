#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
source "${SCRIPT_DIR}/common_runner.sh"

REGISTER="20260318_1K_1566"
DATASET_ROOT="${PROJECT_ROOT}/magformer_datasets/20260318_1K_1566"
OUTPUT_ROOT="${REPO_ROOT}/output/experiments/20260318_1k_1566_20ep_1024_full19"
MODE="run"
IMAGE_SIZE=""
SINGLE_GPU=0
DIRECT_PYTHON=0
WAIT_FREE_GPU_MB="${WAIT_FREE_GPU_MB:-45000}"
WAIT_CHECK_SEC="${WAIT_CHECK_SEC:-300}"
STAGING_ROOT="${OUTPUT_ROOT}/_staging"

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
      STAGING_ROOT="${OUTPUT_ROOT}/_staging"
      shift 2
      ;;
    --image-size)
      IMAGE_SIZE="$2"
      shift 2
      ;;
    --single-gpu)
      SINGLE_GPU=1
      shift
      ;;
    --direct-python)
      DIRECT_PYTHON=1
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

mkdir -p "${OUTPUT_ROOT}" "${STAGING_ROOT}"
RUN_ALL_LOG="${OUTPUT_ROOT}/run_all.log"
if [[ "${MODE}" == "run" ]]; then
  : > "${RUN_ALL_LOG}"
fi

TRACK_NAME="$(basename "${OUTPUT_ROOT}")"
SUMMARY_JSON="${OUTPUT_ROOT}/summary_${TRACK_NAME}.json"
EXTENDED_JSON="${OUTPUT_ROOT}/extended_metrics_table.json"
EXTENDED_CSV="${OUTPUT_ROOT}/extended_metrics_table.csv"
EXTENDED_MD="${OUTPUT_ROOT}/extended_metrics_table.md"
MODELS_MANIFEST_JSON="${OUTPUT_ROOT}/models_manifest.json"
ROSTER_TSV="${OUTPUT_ROOT}/.full19_roster.tsv"

runner_log "${MODE}" "${RUN_ALL_LOG}" "[20260318-full19] mode=${MODE}"
runner_log "${MODE}" "${RUN_ALL_LOG}" "[20260318-full19] register=${REGISTER}"
runner_log "${MODE}" "${RUN_ALL_LOG}" "[20260318-full19] dataset_root=${DATASET_ROOT}"
runner_log "${MODE}" "${RUN_ALL_LOG}" "[20260318-full19] output_root=${OUTPUT_ROOT}"
if [[ -n "${IMAGE_SIZE}" ]]; then
  runner_log "${MODE}" "${RUN_ALL_LOG}" "[20260318-full19] image_size=${IMAGE_SIZE}"
fi
runner_log "${MODE}" "${RUN_ALL_LOG}" "[20260318-full19] single_gpu=${SINGLE_GPU}"
runner_log "${MODE}" "${RUN_ALL_LOG}" "[20260318-full19] direct_python=${DIRECT_PYTHON}"

setup_direct_python_shim() {
  if [[ "${DIRECT_PYTHON}" != "1" ]]; then
    return 0
  fi
  local shim_dir="${OUTPUT_ROOT}/_conda_shim"
  local real_conda
  local env_python
  real_conda="$(command -v conda)"
  env_python="$("${real_conda}" env list --json | python3 -c 'import json, sys; payload=json.load(sys.stdin); print(next(f"{p}/bin/python" for p in payload.get("envs", []) if p.rstrip("/").endswith("/magformer")))' )"
  mkdir -p "${shim_dir}"
  cat > "${shim_dir}/conda" <<EOS
#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "run" ]]; then
  shift
fi
if [[ "${1:-}" == "-n" ]]; then
  shift 2
fi
if [[ "${1:-}" == "python" ]]; then
  shift
  exec ${env_python} "\$@"
fi
exec ${real_conda} "\$@"
EOS
  chmod +x "${shim_dir}/conda"
  export PATH="${shim_dir}:${PATH}"
}

setup_direct_python_shim

recover_completed_staging() {
  local model_id="$1"
  local source_name="$2"
  local final_dir="$3"
  local src_dir="${STAGING_ROOT}/${source_name}"
  local final_done_marker="${final_dir}/metrics.cocoeval.json"
  local staged_done_marker="${src_dir}/metrics.cocoeval.json"

  if [[ -f "${final_done_marker}" ]]; then
    return 0
  fi
  if [[ ! -f "${staged_done_marker}" ]]; then
    return 0
  fi
  if [[ "${MODE}" != "run" ]]; then
    runner_log "${MODE}" "${RUN_ALL_LOG}" "[20260318-full19] RECOVER ${model_id} (dry-run)"
    return 0
  fi

  runner_log "${MODE}" "${RUN_ALL_LOG}" "[20260318-full19] RECOVER ${model_id} from ${src_dir}"
  rm -rf "${final_dir}"
  mv "${src_dir}" "${final_dir}"
}

prepare_model_staging() {
  local model_id="$1"
  local source_name="$2"
  local final_dir="$3"
  local src_dir="${STAGING_ROOT}/${source_name}"
  local final_done_marker="${final_dir}/metrics.cocoeval.json"
  local staged_done_marker="${src_dir}/metrics.cocoeval.json"

  if [[ "${MODE}" != "run" ]]; then
    return 0
  fi
  if [[ -f "${final_done_marker}" ]]; then
    return 0
  fi
  if [[ ! -d "${src_dir}" ]]; then
    return 0
  fi
  if [[ -f "${staged_done_marker}" ]]; then
    return 0
  fi

  runner_log "${MODE}" "${RUN_ALL_LOG}" "[20260318-full19] PURGE stale staging for ${model_id}: ${src_dir}"
  rm -rf "${src_dir}"
}

finalize_model_dir() {
  local model_id="$1"
  local source_name="$2"
  local final_dir="$3"
  local src_dir="${STAGING_ROOT}/${source_name}"
  local staged_done_marker="${src_dir}/metrics.cocoeval.json"
  local final_done_marker="${final_dir}/metrics.cocoeval.json"

  if [[ "${MODE}" != "run" ]]; then
    return 0
  fi
  if [[ ! -d "${src_dir}" ]]; then
    runner_log "${MODE}" "${RUN_ALL_LOG}" "[20260318-full19] FAILED missing staging dir for ${model_id}: ${src_dir}"
    exit 1
  fi
  if [[ ! -f "${staged_done_marker}" ]]; then
    runner_log "${MODE}" "${RUN_ALL_LOG}" "[20260318-full19] FAILED missing staged metrics for ${model_id}: ${staged_done_marker}"
    exit 1
  fi

  runner_log "${MODE}" "${RUN_ALL_LOG}" "[20260318-full19] ARCHIVE ${model_id} ${src_dir} -> ${final_dir}"
  rm -rf "${final_dir}"
  mv "${src_dir}" "${final_dir}"
  if [[ ! -f "${final_done_marker}" ]]; then
    runner_log "${MODE}" "${RUN_ALL_LOG}" "[20260318-full19] FAILED archive verification for ${model_id}: ${final_done_marker}"
    exit 1
  fi
}

run_summary() {
  runner_exec "${MODE}" "${RUN_ALL_LOG}" "cd '${REPO_ROOT}' && python3 scripts/experiments/full19_roster.py --format manifest > '${MODELS_MANIFEST_JSON}'"
  runner_exec "${MODE}" "${RUN_ALL_LOG}" "cd '${REPO_ROOT}' && python3 scripts/experiments/summarize_suite.py --output-root '${OUTPUT_ROOT}' --models-manifest '${MODELS_MANIFEST_JSON}' --write"
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

ROSTER_CMD=(
  python3 "${SCRIPT_DIR}/full19_roster.py"
  --format commands
  --register "${REGISTER}"
  --dataset-root "${DATASET_ROOT}"
  --output-root "${STAGING_ROOT}"
  --mode "${MODE}"
)
if [[ -n "${IMAGE_SIZE}" ]]; then
  ROSTER_CMD+=(--image-size "${IMAGE_SIZE}")
fi
if [[ "${SINGLE_GPU}" == "1" ]]; then
  ROSTER_CMD+=(--single-gpu)
fi
"${ROSTER_CMD[@]}" > "${ROSTER_TSV}"

FAILED_MODELS_FILE="${OUTPUT_ROOT}/failed_models.tsv"
FAILED_MODEL_COUNT=0
if [[ "${MODE}" == "run" ]]; then
  : > "${FAILED_MODELS_FILE}"
fi
while IFS=$'\t' read -r MODEL_ID SOURCE_NAME CMD; do
  FINAL_DIR="${OUTPUT_ROOT}/${MODEL_ID}"
  DONE_MARKER="${FINAL_DIR}/metrics.cocoeval.json"
  recover_completed_staging "${MODEL_ID}" "${SOURCE_NAME}" "${FINAL_DIR}"
  if [[ -f "${DONE_MARKER}" ]]; then
    runner_log "${MODE}" "${RUN_ALL_LOG}" "[20260318-full19] SKIP ${MODEL_ID} (already done)"
    continue
  fi

  prepare_model_staging "${MODEL_ID}" "${SOURCE_NAME}" "${FINAL_DIR}"
  runner_wait_for_free_gpu_mb "${MODE}" "${RUN_ALL_LOG}" "${WAIT_FREE_GPU_MB}" "${WAIT_CHECK_SEC}" "${MODEL_ID}"
  runner_log "${MODE}" "${RUN_ALL_LOG}" "[20260318-full19] START ${MODEL_ID}"
  if [[ "${MODE}" != "run" ]]; then
    runner_exec "${MODE}" "${RUN_ALL_LOG}" "${CMD} </dev/null"
    finalize_model_dir "${MODEL_ID}" "${SOURCE_NAME}" "${FINAL_DIR}"
    runner_log "${MODE}" "${RUN_ALL_LOG}" "[20260318-full19] END ${MODEL_ID}"
    continue
  fi
  runner_log "${MODE}" "${RUN_ALL_LOG}" "+ ${CMD}"
  set +e
  eval "${CMD} </dev/null" 2>&1 | tee -a "${RUN_ALL_LOG}"
  RC=${PIPESTATUS[0]}
  set -e
  if [[ ${RC} -ne 0 ]]; then
    runner_log "${MODE}" "${RUN_ALL_LOG}" "[20260318-full19] FAILED ${MODEL_ID} rc=${RC}"
    printf "%s\t%s\t%s\n" "${MODEL_ID}" "${SOURCE_NAME}" "${RC}" >> "${FAILED_MODELS_FILE}"
    FAILED_MODEL_COUNT=$((FAILED_MODEL_COUNT+1))
    continue
  fi
  finalize_model_dir "${MODEL_ID}" "${SOURCE_NAME}" "${FINAL_DIR}"
  runner_log "${MODE}" "${RUN_ALL_LOG}" "[20260318-full19] END ${MODEL_ID}"
done < "${ROSTER_TSV}"

if [[ "${MODE}" == "run" && ${FAILED_MODEL_COUNT} -gt 0 ]]; then
  runner_log "${MODE}" "${RUN_ALL_LOG}" "[20260318-full19] completed with ${FAILED_MODEL_COUNT} failures (see ${FAILED_MODELS_FILE})"
  EXIT_CODE=1
else
  EXIT_CODE=0
fi

# Run summary/benchmark/visualization once after handling failures.
run_summary
run_benchmarks
run_summary
run_visualizations
write_extended_metrics

runner_log "${MODE}" "${RUN_ALL_LOG}" "[20260318-full19] done"
exit "${EXIT_CODE}"
