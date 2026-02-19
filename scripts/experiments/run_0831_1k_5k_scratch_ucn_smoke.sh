#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"

DATASET_ROOT_DEFAULT="${PROJECT_ROOT}/magformer_datasets/0831_1K"
OUTPUT_ROOT_DEFAULT="${REPO_ROOT}/output/experiments/0831_1k_5k_scratch8_smoke"

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

OUT="${OUTPUT_ROOT}/ucn_scratch"

mkdir -p "${OUT}"

run_cmd() {
  echo "+ $*"
  if [[ "${MODE}" == "run" ]]; then
    eval "$@"
  fi
}

echo "[ucn-0831-1k-smoke] mode=${MODE}"
echo "[ucn-0831-1k-smoke] dataset_root=${DATASET_ROOT}"
echo "[ucn-0831-1k-smoke] output_dir=${OUT}"

EPOCHS=1

SECONDS=0
run_cmd "cd '${REPO_ROOT}' && conda run -n magformer python baselines/run_ucn_0831_1k.py \
  --dataset-root '${DATASET_ROOT}' \
  --output-dir '${OUT}' \
  --epochs ${EPOCHS} \
  --batch 2 \
  --img-size 512"
echo "${SECONDS}" > "${OUT}/wall_time_sec.txt"

echo "[ucn-0831-1k-smoke] done"

