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
QUEUE_TAG="20260502_repaired_stardist_gpu1_100ep"
GPU="${CUDA_VISIBLE_DEVICES:-1}"
EPOCHS=100
NUM_WORKERS=0
WAIT_FREE_MB=60000
WAIT_SLEEP_SEC=30
MAX_RAM_USED_PCT=50
MAX_SWAP_USED_MB=1024
BATCH_512=4
BATCH_1024=1
MAX_TRAIN_IMAGES=0
MAX_VAL_IMAGES=0
STARDIST_ENV="${STARDIST_ENV:-stardist}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --register) REGISTER="$2"; shift 2 ;;
    --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
    --output-base) OUTPUT_BASE="$2"; shift 2 ;;
    --queue-tag) QUEUE_TAG="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    --wait-free-mb) WAIT_FREE_MB="$2"; shift 2 ;;
    --wait-sleep-sec) WAIT_SLEEP_SEC="$2"; shift 2 ;;
    --max-ram-used-pct) MAX_RAM_USED_PCT="$2"; shift 2 ;;
    --max-swap-used-mb) MAX_SWAP_USED_MB="$2"; shift 2 ;;
    --batch-512) BATCH_512="$2"; shift 2 ;;
    --batch-1024) BATCH_1024="$2"; shift 2 ;;
    --max-train-images) MAX_TRAIN_IMAGES="$2"; shift 2 ;;
    --max-val-images) MAX_VAL_IMAGES="$2"; shift 2 ;;
    --stardist-env) STARDIST_ENV="$2"; shift 2 ;;
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
export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-1}"
export TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-4}"
export TF_NUM_INTEROP_THREADS="${TF_NUM_INTEROP_THREADS:-2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
export OPENCV_FOR_THREADS_NUM="${OPENCV_FOR_THREADS_NUM:-4}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"

OUTPUT_ROOT_512="${OUTPUT_BASE}/20260429_repaired_unet_100ep_512_full19"
OUTPUT_ROOT_1024="${OUTPUT_BASE}/20260429_repaired_unet_100ep_1024_full19"
RUN_LOG="$(runner_setup_log "${OUTPUT_BASE}/${QUEUE_TAG}" "${MODE}")"

min_ram_mb_for_pct() {
  local max_used_pct="$1"
  local total_mb
  total_mb="$(runner_mem_total_mb)"
  if [[ -z "${total_mb}" || "${total_mb}" -le 0 ]]; then
    echo "65000"
    return 0
  fi
  awk -v total="${total_mb}" -v max_used="${max_used_pct}" 'BEGIN { printf "%d\n", int(total * (100 - max_used) / 100) }'
}

stardist_completed_artifacts_valid() {
  local metrics_path="$1"
  local results_path="$2"
  if [[ ! -s "${metrics_path}" || ! -s "${results_path}" ]]; then
    return 1
  fi

  python3 - "${metrics_path}" "${results_path}" <<'PY'
import json
import sys

metrics_path, results_path = sys.argv[1], sys.argv[2]
try:
    with open(metrics_path, "r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    with open(results_path, "r", encoding="utf-8") as handle:
        rows = json.load(handle)
except Exception:
    raise SystemExit(1)

required_metric_keys = {"segm/AP", "segm/AP50", "segm/AP75", "bbox/AP", "bbox/AP50", "bbox/AP75"}
if not isinstance(metrics, dict) or not required_metric_keys.issubset(metrics):
    raise SystemExit(1)
if not isinstance(rows, list) or not rows:
    raise SystemExit(1)
required_row_keys = {"image_id", "category_id", "score", "segmentation"}
for row in rows[: min(len(rows), 25)]:
    if not isinstance(row, dict) or not required_row_keys.issubset(row):
        raise SystemExit(1)
raise SystemExit(0)
PY
}

run_stardist_if_missing() {
  local image_size="$1"
  local output_root="$2"
  local batch="$3"
  local out_dir="${output_root}/stardist"
  local metrics_marker="${out_dir}/metrics.cocoeval.json"
  local results_marker="${out_dir}/coco_instances_results.json"
  local label="stardist_${image_size}_100ep"
  local min_ram_mb
  min_ram_mb="$(min_ram_mb_for_pct "${MAX_RAM_USED_PCT}")"

  runner_log "${MODE}" "${RUN_LOG}" "[repaired-stardist] plan ${label}: batch=${batch} min_ram_mb=${min_ram_mb} max_ram_used_pct=${MAX_RAM_USED_PCT}"
  if stardist_completed_artifacts_valid "${metrics_marker}" "${results_marker}"; then
    runner_log "${MODE}" "${RUN_LOG}" "[repaired-stardist] skip ${label}: ${metrics_marker} and ${results_marker}"
    return 0
  fi

  runner_wait_for_free_gpu_mb "${MODE}" "${RUN_LOG}" "${WAIT_FREE_MB}" "${WAIT_SLEEP_SEC}" "${label}"
  runner_wait_for_system_resources "${MODE}" "${RUN_LOG}" "${min_ram_mb}" "${MAX_SWAP_USED_MB}" "${WAIT_SLEEP_SEC}" "${label}"
  runner_exec_gpu_locked "${MODE}" "${RUN_LOG}" "${GPU}" "${metrics_marker}.lock" "${label}" \
    "cd '${REPO_ROOT}' && STARDIST_ENV='${STARDIST_ENV}' CUDA_VISIBLE_DEVICES=${GPU} bash '${REPO_ROOT}/scripts/experiments/run_0831_1k_20ep_1024_revisit_stardist_inst.sh' --register '${REGISTER}' --dataset-root '${DATASET_ROOT}' --output-root '${output_root}' --image-size ${image_size} --epochs ${EPOCHS} --batch ${batch} --num-workers ${NUM_WORKERS} --ram-limit-pct ${MAX_RAM_USED_PCT} --max-train-images ${MAX_TRAIN_IMAGES} --max-val-images ${MAX_VAL_IMAGES} --${MODE}"
}

runner_log "${MODE}" "${RUN_LOG}" "[repaired-stardist] CUDA_VISIBLE_DEVICES=${GPU}"
runner_log "${MODE}" "${RUN_LOG}" "[repaired-stardist] STARDIST_ENV=${STARDIST_ENV}"
runner_log "${MODE}" "${RUN_LOG}" "[repaired-stardist] dataset_root=${DATASET_ROOT}"
runner_log "${MODE}" "${RUN_LOG}" "[repaired-stardist] output_root_512=${OUTPUT_ROOT_512}"
runner_log "${MODE}" "${RUN_LOG}" "[repaired-stardist] output_root_1024=${OUTPUT_ROOT_1024}"
if [[ "${MODE}" == "run" ]]; then
  runner_log_launch_guard_snapshot "${MODE}" "${RUN_LOG}" "repaired-stardist"
fi

run_stardist_if_missing 512 "${OUTPUT_ROOT_512}" "${BATCH_512}"
run_stardist_if_missing 1024 "${OUTPUT_ROOT_1024}" "${BATCH_1024}"

runner_log "${MODE}" "${RUN_LOG}" "[repaired-stardist] done"
