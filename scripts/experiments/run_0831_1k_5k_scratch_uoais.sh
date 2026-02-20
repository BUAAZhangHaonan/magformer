#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
source "${SCRIPT_DIR}/common_runner.sh"

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

UOAIS_ROOT="${REPO_ROOT}/baselines/icra_2026/uoais"
CFG="${REPO_ROOT}/configs/baselines/uoais_0831_1k_5k_scratch.yaml"
OUT="${OUTPUT_ROOT}/uoais_scratch"

mkdir -p "${OUT}"
RUN_LOG="$(runner_setup_log "${OUT}" "${MODE}")"

runner_log "${MODE}" "${RUN_LOG}" "[uoais-0831-1k-5k-scratch] mode=${MODE}"
runner_log "${MODE}" "${RUN_LOG}" "[uoais-0831-1k-5k-scratch] dataset_root=${DATASET_ROOT}"
runner_log "${MODE}" "${RUN_LOG}" "[uoais-0831-1k-5k-scratch] output_dir=${OUT}"
runner_log "${MODE}" "${RUN_LOG}" "[uoais-0831-1k-5k-scratch] config=${CFG}"

SECONDS=0
runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && conda run -n magformer python baselines/run_uoais_0831_1k.py \
  --dataset-root '${DATASET_ROOT}' \
  --uoais-root '${UOAIS_ROOT}' \
  -- \
  --num-gpus 1 \
  --config-file '${CFG}' \
  OUTPUT_DIR '${OUT}'"
echo "${SECONDS}" > "${OUT}/wall_time_sec.txt"

if [[ "${MODE}" == "run" ]]; then
  runner_exec "${MODE}" "${RUN_LOG}" "conda run -n magformer python '${REPO_ROOT}/scripts/analysis/write_params_from_detectron2_ckpt.py' --out-dir '${OUT}'"
fi

runner_log "${MODE}" "${RUN_LOG}" "[uoais-0831-1k-5k-scratch] done"
