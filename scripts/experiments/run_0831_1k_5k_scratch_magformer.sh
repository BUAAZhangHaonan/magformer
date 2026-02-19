#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"

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

CFG="${REPO_ROOT}/configs/magformer_0831_1k_5k_scratch.yaml"
OUT="${OUTPUT_ROOT}/magformer_scratch"

mkdir -p "${OUT}"

run_cmd() {
  echo "+ $*"
  if [[ "${MODE}" == "run" ]]; then
    eval "$@"
  fi
}

echo "[magformer-0831-1k-5k-scratch] mode=${MODE}"
echo "[magformer-0831-1k-5k-scratch] dataset_root=${DATASET_ROOT}"
echo "[magformer-0831-1k-5k-scratch] output_dir=${OUT}"

SECONDS=0
run_cmd "cd '${REPO_ROOT}' && conda run -n magformer python tools/train.py --config '${CFG}' --dataset-root '${DATASET_ROOT}' --output-dir '${OUT}' --num-workers 4"
echo "${SECONDS}" > "${OUT}/wall_time_sec.txt"

if [[ "${MODE}" == "run" ]]; then
  conda run -n magformer python scripts/analysis/write_params_from_magformer_ckpt.py --out-dir "${OUT}"
fi

echo "[magformer-0831-1k-5k-scratch] done"
