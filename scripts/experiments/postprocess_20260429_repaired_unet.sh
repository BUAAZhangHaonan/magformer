#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/common_runner.sh"
source "${SCRIPT_DIR}/ecc_common.sh"

MODE="run"
REGISTER="20260318_1K_1566"
DATASET_ROOT=""
OUTPUT_BASE="${REPO_ROOT}/output/experiments"
QUEUE_TAG="20260429_repaired_unet_postprocess"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
NUM_IMAGES=50
SCORE_THRESHOLD=0.5
WARMUP=5
TIMED_IMAGES=50

while [[ $# -gt 0 ]]; do
  case "$1" in
    --register) REGISTER="$2"; shift 2 ;;
    --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
    --output-base) OUTPUT_BASE="$2"; shift 2 ;;
    --queue-tag) QUEUE_TAG="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --num-images) NUM_IMAGES="$2"; shift 2 ;;
    --score-threshold) SCORE_THRESHOLD="$2"; shift 2 ;;
    --warmup) WARMUP="$2"; shift 2 ;;
    --timed-images) TIMED_IMAGES="$2"; shift 2 ;;
    --run) MODE="run"; shift ;;
    --dry-run) MODE="dry-run"; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "${DATASET_ROOT}" ]]; then
  DATASET_ROOT="$(ecc_default_dataset_root "${REGISTER}")"
fi
if [[ -d "${DATASET_ROOT}" ]]; then
  DATASET_ROOT="$(cd "${DATASET_ROOT}" && pwd)"
elif [[ "${MODE}" == "run" ]]; then
  echo "Dataset root does not exist: ${DATASET_ROOT}" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
export OPENCV_FOR_THREADS_NUM="${OPENCV_FOR_THREADS_NUM:-4}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"

OUTPUT_ROOT_512="${OUTPUT_BASE}/20260429_repaired_unet_100ep_512_full19"
OUTPUT_ROOT_1024="${OUTPUT_BASE}/20260429_repaired_unet_100ep_1024_full19"
POST_ROOT="${OUTPUT_BASE}/${QUEUE_TAG}"
RUN_LOG="$(runner_setup_log "${POST_ROOT}" "${MODE}")"
ROSTER_MANIFEST="${POST_ROOT}/full19_roster_manifest.json"
LIVE_MANIFEST="${POST_ROOT}/full19_live_metrics_manifest.json"
EXTENDED_JSON="${POST_ROOT}/full19_extended_metrics.json"
EXTENDED_CSV="${POST_ROOT}/full19_extended_metrics.csv"
EXTENDED_MD="${POST_ROOT}/full19_extended_metrics.md"

SUMMARY_512="${OUTPUT_ROOT_512}/summary_$(basename "${OUTPUT_ROOT_512}").json"
SUMMARY_1024="${OUTPUT_ROOT_1024}/summary_$(basename "${OUTPUT_ROOT_1024}").json"

runner_log "${MODE}" "${RUN_LOG}" "[repaired-postprocess] CUDA_VISIBLE_DEVICES=${GPU}"
runner_log "${MODE}" "${RUN_LOG}" "[repaired-postprocess] dataset_root=${DATASET_ROOT}"
runner_log "${MODE}" "${RUN_LOG}" "[repaired-postprocess] output_root_512=${OUTPUT_ROOT_512}"
runner_log "${MODE}" "${RUN_LOG}" "[repaired-postprocess] output_root_1024=${OUTPUT_ROOT_1024}"

postprocess_root() {
  local label_prefix="$1"
  local output_root="$2"
  local summary_json="$3"

  runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && python3 scripts/experiments/full19_roster.py --format manifest > '${ROSTER_MANIFEST}'"
  runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && conda run -n magformer python scripts/experiments/summarize_suite.py --output-root '${output_root}' --models-manifest '${ROSTER_MANIFEST}' --write"
  runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && CUDA_VISIBLE_DEVICES=${GPU} conda run -n magformer python scripts/analysis/benchmark_inference_suite.py --output-root '${output_root}' --dataset-root '${DATASET_ROOT}' --summary '${summary_json}' --device cuda --warmup '${WARMUP}' --timed-images '${TIMED_IMAGES}' --continue-on-error"
  runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && conda run -n magformer python scripts/experiments/summarize_suite.py --output-root '${output_root}' --models-manifest '${ROSTER_MANIFEST}' --write"
  runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && conda run -n magformer python scripts/experiments/visualize_suite.py --output-root '${output_root}' --dataset-root '${DATASET_ROOT}' --summary '${summary_json}' --num-images '${NUM_IMAGES}' --score-threshold '${SCORE_THRESHOLD}'"
}

postprocess_root "512" "${OUTPUT_ROOT_512}" "${SUMMARY_512}"
postprocess_root "1024" "${OUTPUT_ROOT_1024}" "${SUMMARY_1024}"

runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && conda run -n magformer python scripts/analysis/build_full19_live_metrics_manifest.py --output '${LIVE_MANIFEST}'"
runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && conda run -n magformer python scripts/analysis/write_extended_metrics_table.py --manifest '${LIVE_MANIFEST}' --out-json '${EXTENDED_JSON}' --out-csv '${EXTENDED_CSV}' --out-md '${EXTENDED_MD}'"

runner_log "${MODE}" "${RUN_LOG}" "[repaired-postprocess] done"
