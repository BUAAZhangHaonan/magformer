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
GPU="1"
WAIT_FREE_MB=78000
WAIT_SLEEP_SEC=30
MIN_RAM_MB=50000
MAX_SWAP_USED_MB=1024

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
    --gpu)
      GPU="$2"
      shift 2
      ;;
    --wait-free-mb)
      WAIT_FREE_MB="$2"
      shift 2
      ;;
    --wait-sleep-sec)
      WAIT_SLEEP_SEC="$2"
      shift 2
      ;;
    --min-ram-mb)
      MIN_RAM_MB="$2"
      shift 2
      ;;
    --max-swap-used-mb)
      MAX_SWAP_USED_MB="$2"
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
export CUDA_VISIBLE_DEVICES="${GPU}"

OUTPUT_ROOT_512="${OUTPUT_BASE}/${DATE_TAG}_1k_1566_20ep_512_full19"
RUN_LOG="$(runner_setup_log "${OUTPUT_BASE}/${DATE_TAG}_gpu1_campaign" "${MODE}")"

runner_log "${MODE}" "${RUN_LOG}" "[campaign-gpu1] CUDA_VISIBLE_DEVICES=${GPU}"
runner_log "${MODE}" "${RUN_LOG}" "[campaign-gpu1] dataset_root=${DATASET_ROOT}"
runner_log "${MODE}" "${RUN_LOG}" "[campaign-gpu1] output_root_512=${OUTPUT_ROOT_512}"
runner_log "${MODE}" "${RUN_LOG}" "[campaign-gpu1] wait_free_mb=${WAIT_FREE_MB} wait_sleep_sec=${WAIT_SLEEP_SEC}"
runner_log "${MODE}" "${RUN_LOG}" "[campaign-gpu1] min_ram_mb=${MIN_RAM_MB} max_swap_used_mb=${MAX_SWAP_USED_MB}"

run_if_missing() {
  local label="$1"
  local done_marker="$2"
  shift 2
  runner_log "${MODE}" "${RUN_LOG}" "[campaign-gpu1] plan ${label}"
  if [[ -f "${done_marker}" ]]; then
    runner_log "${MODE}" "${RUN_LOG}" "[campaign-gpu1] skip ${label}: ${done_marker}"
    return 0
  fi
  runner_wait_for_free_gpu_mb "${MODE}" "${RUN_LOG}" "${WAIT_FREE_MB}" "${WAIT_SLEEP_SEC}" "${label}"
  runner_wait_for_system_resources "${MODE}" "${RUN_LOG}" "${MIN_RAM_MB}" "${MAX_SWAP_USED_MB}" "${WAIT_SLEEP_SEC}" "${label}"
  if [[ -f "${done_marker}" ]]; then
    runner_log "${MODE}" "${RUN_LOG}" "[campaign-gpu1] skip ${label} after wait: ${done_marker}"
    return 0
  fi
  runner_exec "${MODE}" "${RUN_LOG}" "$*"
}

run_if_missing \
  "official_mask2former_pretrained_512" \
  "${OUTPUT_ROOT_512}/official_mask2former_pretrained/metrics.cocoeval.json" \
  "cd '${REPO_ROOT}' && CUDA_VISIBLE_DEVICES=${GPU} bash '${REPO_ROOT}/scripts/experiments/run_0831_1k_20ep_scratch_official_mask2former.sh' --register '${REGISTER}' --dataset-root '${DATASET_ROOT}' --output-root '${OUTPUT_ROOT_512}' --candidate-id C1 --run-tag final --image-size 512 --pretrained --${MODE}"

run_if_missing \
  "maskrcnn_pretrained_512" \
  "${OUTPUT_ROOT_512}/maskrcnn_pretrained/metrics.cocoeval.json" \
  "cd '${REPO_ROOT}' && CUDA_VISIBLE_DEVICES=${GPU} bash '${REPO_ROOT}/scripts/experiments/run_0831_1k_20ep_scratch_maskrcnn.sh' --register '${REGISTER}' --dataset-root '${DATASET_ROOT}' --output-root '${OUTPUT_ROOT_512}' --candidate-id C1 --run-tag final --image-size 512 --pretrained --${MODE}"

run_if_missing \
  "mgm_mask2former_nodpth_ref_512" \
  "${OUTPUT_ROOT_512}/mgm_mask2former_nodpth_ref/metrics.cocoeval.json" \
  "cd '${REPO_ROOT}' && CUDA_VISIBLE_DEVICES=${GPU} bash '${REPO_ROOT}/scripts/experiments/run_0831_1k_20ep_1024_revisit_mgm_mask2former.sh' --register '${REGISTER}' --dataset-root '${DATASET_ROOT}' --output-root '${OUTPUT_ROOT_512}' --variant nodpth_ref --image-size 512 --num-gpus 1 --${MODE}"

run_if_missing \
  "yolov8_seg_x_pretrained_512" \
  "${OUTPUT_ROOT_512}/yolov8_seg_x_pretrained/metrics.cocoeval.json" \
  "cd '${REPO_ROOT}' && CUDA_VISIBLE_DEVICES=${GPU} bash '${REPO_ROOT}/scripts/experiments/run_0831_1k_20ep_scratch_yolov8_seg.sh' --register '${REGISTER}' --dataset-root '${DATASET_ROOT}' --output-root '${OUTPUT_ROOT_512}' --candidate-id C1 --run-tag final --image-size 512 --model-size x --pretrained --device 0 --${MODE}"

run_if_missing \
  "iaunet_512" \
  "${OUTPUT_ROOT_512}/iaunet/metrics.cocoeval.json" \
  "cd '${REPO_ROOT}' && CUDA_VISIBLE_DEVICES=${GPU} bash '${REPO_ROOT}/scripts/experiments/run_0831_1k_20ep_1024_revisit_iaunet_inst.sh' --register '${REGISTER}' --dataset-root '${DATASET_ROOT}' --output-root '${OUTPUT_ROOT_512}' --image-size 512 --${MODE}"

run_if_missing \
  "cellpose_512" \
  "${OUTPUT_ROOT_512}/cellpose/metrics.cocoeval.json" \
  "cd '${REPO_ROOT}' && CUDA_VISIBLE_DEVICES=${GPU} bash '${REPO_ROOT}/scripts/experiments/run_0831_1k_20ep_1024_revisit_cellpose_inst.sh' --register '${REGISTER}' --dataset-root '${DATASET_ROOT}' --output-root '${OUTPUT_ROOT_512}' --image-size 512 --${MODE}"

run_if_missing \
  "stardist_512" \
  "${OUTPUT_ROOT_512}/stardist/metrics.cocoeval.json" \
  "cd '${REPO_ROOT}' && CUDA_VISIBLE_DEVICES=${GPU} bash '${REPO_ROOT}/scripts/experiments/run_0831_1k_20ep_1024_revisit_stardist_inst.sh' --register '${REGISTER}' --dataset-root '${DATASET_ROOT}' --output-root '${OUTPUT_ROOT_512}' --image-size 512 --${MODE}"

runner_log "${MODE}" "${RUN_LOG}" "[campaign-gpu1] done"
