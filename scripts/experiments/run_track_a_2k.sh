#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"

DATASET_ROOT_DEFAULT="${PROJECT_ROOT}/magformer_datasets/0909_512_0.12K"
OUTPUT_ROOT_DEFAULT="${REPO_ROOT}/output/experiments/track_a_2k"

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

MAGFORMER_DIR="${REPO_ROOT}"
MASK2FORMER_DIR="${PROJECT_ROOT}/mask2former/MGM_Mask2Former"

MAGFORMER_CONFIG="${MAGFORMER_DIR}/configs/magformer_track_a_2k.yaml"
MASK2FORMER_CONFIG="${MASK2FORMER_DIR}/configs/mgm_aligned_comparison.yaml"

MAGFORMER_OUT="${OUTPUT_ROOT}/magformer"
MASK2FORMER_OUT="${OUTPUT_ROOT}/mask2former"
SUMMARY_FILE="${OUTPUT_ROOT}/track_a_2k_summary.json"

mkdir -p "${MAGFORMER_OUT}" "${MASK2FORMER_OUT}"

run_cmd() {
  echo "+ $*"
  if [[ "${MODE}" == "run" ]]; then
    eval "$@"
  fi
}

echo "[track-a-2k] mode=${MODE}"
echo "[track-a-2k] dataset_root=${DATASET_ROOT}"
echo "[track-a-2k] output_root=${OUTPUT_ROOT}"

run_cmd "cd '${MAGFORMER_DIR}' && conda run -n magformer python tools/train.py --config '${MAGFORMER_CONFIG}' --dataset-root '${DATASET_ROOT}' --output-dir '${MAGFORMER_OUT}' --num-workers 4"

run_cmd "cd '${MASK2FORMER_DIR}' && conda run -n magformer python train_net_mgm_0831.py --num-gpus 1 --config-file '${MASK2FORMER_CONFIG}' INPUT.DATASET_ROOT '${DATASET_ROOT}' OUTPUT_DIR '${MASK2FORMER_OUT}' MODEL.FINETUNE_WEIGHTS '' MODEL.WEIGHTS ''"

run_cmd "cd '${REPO_ROOT}' && conda run -n magformer python scripts/analysis/summarize_track_a_2k.py \
  --magformer-metrics '${MAGFORMER_OUT}/metrics_log.jsonl' \
  --mask2former-metrics '${MASK2FORMER_OUT}/metrics.json' \
  --summary-file '${SUMMARY_FILE}'"

echo "[track-a-2k] done"
