#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/common_runner.sh"

MODE="run"
REGISTER="20260318_1K_1566"
DATASET_ROOT="${REPO_ROOT}/magformer_datasets/20260318_1K_1566"
OUTPUT_ROOT_512="${REPO_ROOT}/output/experiments/20260406_1k_1566_20ep_512_full19"

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
    --output-root-512)
      OUTPUT_ROOT_512="$2"
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

mkdir -p "${OUTPUT_ROOT_512}"
RUN_LOG="$(runner_setup_log "${OUTPUT_ROOT_512}" "${MODE}")"

runner_log "${MODE}" "${RUN_LOG}" "[baseline-repair-gpu1] register=${REGISTER}"
runner_log "${MODE}" "${RUN_LOG}" "[baseline-repair-gpu1] dataset_root=${DATASET_ROOT}"
runner_log "${MODE}" "${RUN_LOG}" "[baseline-repair-gpu1] output_root_512=${OUTPUT_ROOT_512}"

runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && bash scripts/experiments/run_0831_1k_20ep_scratch_official_mask2former.sh --register '${REGISTER}' --dataset-root '${DATASET_ROOT}' --output-root '${OUTPUT_ROOT_512}' --candidate-id C1 --run-tag final --image-size 512 --pretrained --run"
runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && bash scripts/experiments/run_0831_1k_20ep_scratch_maskrcnn.sh --register '${REGISTER}' --dataset-root '${DATASET_ROOT}' --output-root '${OUTPUT_ROOT_512}' --candidate-id C1 --run-tag final --image-size 512 --pretrained --run"
runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && bash scripts/experiments/run_0831_1k_20ep_1024_revisit_mgm_mask2former.sh --register '${REGISTER}' --dataset-root '${DATASET_ROOT}' --output-root '${OUTPUT_ROOT_512}' --variant nodpth_ref --image-size 512 --run"
runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && bash scripts/experiments/run_0831_1k_20ep_scratch_yolov8_seg.sh --register '${REGISTER}' --dataset-root '${DATASET_ROOT}' --output-root '${OUTPUT_ROOT_512}' --candidate-id C1 --run-tag final --image-size 512 --model-size x --pretrained --run"

runner_log "${MODE}" "${RUN_LOG}" "[baseline-repair-gpu1] done"
