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
DATE_TAG="20260406"
VARIANT="nodpth_ref"
IMAGE_SIZE=512
NUM_GPUS=1

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
    --output-base)
      OUTPUT_BASE="$2"
      shift 2
      ;;
    --date-tag)
      DATE_TAG="$2"
      shift 2
      ;;
    --variant)
      VARIANT="$2"
      shift 2
      ;;
    --image-size)
      IMAGE_SIZE="$2"
      shift 2
      ;;
    --num-gpus)
      NUM_GPUS="$2"
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

if [[ -z "${DATASET_ROOT}" ]]; then
  DATASET_ROOT="$(ecc_default_dataset_root "${REGISTER}")"
fi

RUN_LOG="$(runner_setup_log "${OUTPUT_BASE}/${DATE_TAG}_gpu1_resume_mgm" "${MODE}")"
runner_log "${MODE}" "${RUN_LOG}" "[gpu1-resume-mgm] register=${REGISTER}"
runner_log "${MODE}" "${RUN_LOG}" "[gpu1-resume-mgm] dataset_root=${DATASET_ROOT}"
runner_log "${MODE}" "${RUN_LOG}" "[gpu1-resume-mgm] output_base=${OUTPUT_BASE}"

runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && CUDA_VISIBLE_DEVICES=1 bash scripts/experiments/run_0831_1k_20ep_1024_revisit_mgm_mask2former.sh \
  --register '${REGISTER}' \
  --dataset-root '${DATASET_ROOT}' \
  --output-root '${OUTPUT_BASE}/${DATE_TAG}_1k_1566_20ep_512_full19' \
  --variant '${VARIANT}' \
  --image-size ${IMAGE_SIZE} \
  --num-gpus ${NUM_GPUS} \
  --resume \
  --run"

runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && CUDA_VISIBLE_DEVICES=1 bash scripts/experiments/run_20260406_training_campaign_gpu1.sh \
  --register '${REGISTER}' \
  --dataset-root '${DATASET_ROOT}' \
  --output-base '${OUTPUT_BASE}' \
  --run"

runner_log "${MODE}" "${RUN_LOG}" "[gpu1-resume-mgm] done"
