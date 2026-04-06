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
GPU="0"

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

OUTPUT_ROOT_1024="${OUTPUT_BASE}/${DATE_TAG}_1k_1566_20ep_1024_full19"
OUTPUT_ROOT_512="${OUTPUT_BASE}/${DATE_TAG}_1k_1566_20ep_512_full19"
RUN_LOG="$(runner_setup_log "${OUTPUT_BASE}/${DATE_TAG}_gpu0_campaign" "${MODE}")"

runner_log "${MODE}" "${RUN_LOG}" "[campaign-gpu0] CUDA_VISIBLE_DEVICES=${GPU}"
runner_log "${MODE}" "${RUN_LOG}" "[campaign-gpu0] dataset_root=${DATASET_ROOT}"
runner_log "${MODE}" "${RUN_LOG}" "[campaign-gpu0] output_root_1024=${OUTPUT_ROOT_1024}"
runner_log "${MODE}" "${RUN_LOG}" "[campaign-gpu0] output_root_512=${OUTPUT_ROOT_512}"

run_if_missing() {
  local label="$1"
  local done_marker="$2"
  shift 2
  runner_log "${MODE}" "${RUN_LOG}" "[campaign-gpu0] plan ${label}"
  if [[ -f "${done_marker}" ]]; then
    runner_log "${MODE}" "${RUN_LOG}" "[campaign-gpu0] skip ${label}: ${done_marker}"
    return 0
  fi
  runner_exec "${MODE}" "${RUN_LOG}" "$*"
}

run_if_missing \
  "magformer_nodpth_ref_fair_1024" \
  "${OUTPUT_ROOT_1024}/magformer_nodpth_ref_fair/metrics.cocoeval.json" \
  "cd '${REPO_ROOT}' && CUDA_VISIBLE_DEVICES=${GPU} bash '${REPO_ROOT}/scripts/experiments/run_0831_1k_20ep_1024_revisit_magformer.sh' --register '${REGISTER}' --dataset-root '${DATASET_ROOT}' --output-root '${OUTPUT_ROOT_1024}' --variant nodpth_ref_fair --image-size 1024 --${MODE}"

run_if_missing \
  "magformer_depthnorm_on_512" \
  "${OUTPUT_ROOT_512}/magformer_depthnorm_on/metrics.cocoeval.json" \
  "cd '${REPO_ROOT}' && CUDA_VISIBLE_DEVICES=${GPU} bash '${REPO_ROOT}/scripts/experiments/run_0831_1k_20ep_1024_revisit_magformer.sh' --register '${REGISTER}' --dataset-root '${DATASET_ROOT}' --output-root '${OUTPUT_ROOT_512}' --variant depthnorm_on --image-size 512 --${MODE}"

run_if_missing \
  "magformer_nodpth_ref_fair_512" \
  "${OUTPUT_ROOT_512}/magformer_nodpth_ref_fair/metrics.cocoeval.json" \
  "cd '${REPO_ROOT}' && CUDA_VISIBLE_DEVICES=${GPU} bash '${REPO_ROOT}/scripts/experiments/run_0831_1k_20ep_1024_revisit_magformer.sh' --register '${REGISTER}' --dataset-root '${DATASET_ROOT}' --output-root '${OUTPUT_ROOT_512}' --variant nodpth_ref_fair --image-size 512 --${MODE}"

run_if_missing \
  "magformer_lightdepth_convnextlite_spatialgate_edge_validhole_512" \
  "${OUTPUT_ROOT_512}/magformer_lightdepth_convnextlite_spatialgate_edge_validhole/metrics.cocoeval.json" \
  "cd '${REPO_ROOT}' && CUDA_VISIBLE_DEVICES=${GPU} bash '${REPO_ROOT}/scripts/experiments/run_0831_1k_20ep_1024_lightdepth_stage_a_magformer.sh' --register '${REGISTER}' --dataset-root '${DATASET_ROOT}' --output-root '${OUTPUT_ROOT_512}' --variant convnextlite_spatialgate_edge_validhole --image-size 512 --${MODE}"

run_if_missing \
  "mgm_mask2former_depthnorm_on_512" \
  "${OUTPUT_ROOT_512}/mgm_mask2former_depthnorm_on/metrics.cocoeval.json" \
  "cd '${REPO_ROOT}' && CUDA_VISIBLE_DEVICES=${GPU} bash '${REPO_ROOT}/scripts/experiments/run_0831_1k_20ep_1024_revisit_mgm_mask2former.sh' --register '${REGISTER}' --dataset-root '${DATASET_ROOT}' --output-root '${OUTPUT_ROOT_512}' --variant depthnorm_on --image-size 512 --num-gpus 1 --${MODE}"

runner_log "${MODE}" "${RUN_LOG}" "[campaign-gpu0] done"
