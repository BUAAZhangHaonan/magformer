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
NUM_WORKERS=4
CHECKPOINT=""

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
    --num-workers)
      NUM_WORKERS="$2"
      shift 2
      ;;
    --checkpoint)
      CHECKPOINT="$2"
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

OUT="${OUTPUT_BASE}/${DATE_TAG}_1k_1566_20ep_1024_full19/magformer_nodpth_ref_fair"
if [[ -z "${CHECKPOINT}" ]]; then
  CHECKPOINT="$(ls "${OUT}"/checkpoint_iter_*.pth 2>/dev/null | sort | tail -n 1)"
fi
if [[ -z "${CHECKPOINT}" ]]; then
  echo "No checkpoint found under ${OUT}" >&2
  exit 1
fi

RUN_LOG="$(runner_setup_log "${OUTPUT_BASE}/${DATE_TAG}_gpu0_resume_fair" "${MODE}")"
runner_log "${MODE}" "${RUN_LOG}" "[gpu0-resume-fair] checkpoint=${CHECKPOINT}"
runner_log "${MODE}" "${RUN_LOG}" "[gpu0-resume-fair] dataset_root=${DATASET_ROOT}"
runner_log "${MODE}" "${RUN_LOG}" "[gpu0-resume-fair] out_dir=${OUT}"

runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && CUDA_VISIBLE_DEVICES=0 conda run -n magformer python tools/train.py \
  --config '${OUT}/magformer_runtime_config.yaml' \
  --dataset-root '${DATASET_ROOT}' \
  --output-dir '${OUT}' \
  --num-workers ${NUM_WORKERS} \
  --resume '${CHECKPOINT}'"

runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && bash scripts/experiments/run_20260407_finalize_fair_gpu0_and_continue.sh \
  --skip-wait \
  --register '${REGISTER}' \
  --dataset-root '${DATASET_ROOT}' \
  --output-base '${OUTPUT_BASE}' \
  --run"

runner_log "${MODE}" "${RUN_LOG}" "[gpu0-resume-fair] done"
