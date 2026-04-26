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
QUEUE_TAG="20260409_gpu1_non256_backfill"
GPU="1"
WAIT_FREE_MB=78000
WAIT_SLEEP_SEC=30

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
    --queue-tag)
      QUEUE_TAG="$2"
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
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"
export OPENCV_FOR_THREADS_NUM="${OPENCV_FOR_THREADS_NUM:-8}"
export TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-8}"
export TF_NUM_INTEROP_THREADS="${TF_NUM_INTEROP_THREADS:-2}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"

OUTPUT_ROOT_512="${OUTPUT_BASE}/20260406_1k_1566_20ep_512_full19"
OUTPUT_ROOT_1024="${OUTPUT_BASE}/20260406_1k_1566_20ep_1024_full19"
RUN_LOG="$(runner_setup_log "${OUTPUT_BASE}/${QUEUE_TAG}" "${MODE}")"

runner_log "${MODE}" "${RUN_LOG}" "[non256-gpu1] CUDA_VISIBLE_DEVICES=${GPU}"
runner_log "${MODE}" "${RUN_LOG}" "[non256-gpu1] dataset_root=${DATASET_ROOT}"
runner_log "${MODE}" "${RUN_LOG}" "[non256-gpu1] output_root_512=${OUTPUT_ROOT_512}"
runner_log "${MODE}" "${RUN_LOG}" "[non256-gpu1] output_root_1024=${OUTPUT_ROOT_1024}"
runner_log "${MODE}" "${RUN_LOG}" "[non256-gpu1] wait_free_mb=${WAIT_FREE_MB} wait_sleep_sec=${WAIT_SLEEP_SEC}"
runner_log "${MODE}" "${RUN_LOG}" "[non256-gpu1] thread_caps OMP=${OMP_NUM_THREADS} MKL=${MKL_NUM_THREADS} OPENBLAS=${OPENBLAS_NUM_THREADS} TF_INTRA=${TF_NUM_INTRAOP_THREADS} TF_INTER=${TF_NUM_INTEROP_THREADS}"

run_if_missing() {
  local label="$1"
  local done_marker="$2"
  shift 2
  runner_log "${MODE}" "${RUN_LOG}" "[non256-gpu1] plan ${label}"
  if [[ -f "${done_marker}" ]]; then
    runner_log "${MODE}" "${RUN_LOG}" "[non256-gpu1] skip ${label}: ${done_marker}"
    return 0
  fi
  runner_wait_for_free_gpu_mb "${MODE}" "${RUN_LOG}" "${WAIT_FREE_MB}" "${WAIT_SLEEP_SEC}" "${label}"
  if [[ -f "${done_marker}" ]]; then
    runner_log "${MODE}" "${RUN_LOG}" "[non256-gpu1] skip ${label} after wait: ${done_marker}"
    return 0
  fi
  runner_exec "${MODE}" "${RUN_LOG}" "$*"
}

run_if_missing \
  "magformer_lightdepth_mobilenetv3_spatialgate_edge_validhole_512" \
  "${OUTPUT_ROOT_512}/magformer_lightdepth_mobilenetv3_spatialgate_edge_validhole/metrics.cocoeval.json" \
  "cd '${REPO_ROOT}' && CUDA_VISIBLE_DEVICES=${GPU} bash '${REPO_ROOT}/scripts/experiments/run_0831_1k_20ep_1024_lightdepth_stage_a_magformer.sh' --register '${REGISTER}' --dataset-root '${DATASET_ROOT}' --output-root '${OUTPUT_ROOT_512}' --variant mobilenetv3_spatialgate_edge_validhole --num-gpus 1 --image-size 512 --${MODE}"

run_if_missing \
  "magformer_lightdepth_mobilenetv3_sagate_edge_validhole_512" \
  "${OUTPUT_ROOT_512}/magformer_lightdepth_mobilenetv3_sagate_edge_validhole/metrics.cocoeval.json" \
  "cd '${REPO_ROOT}' && CUDA_VISIBLE_DEVICES=${GPU} bash '${REPO_ROOT}/scripts/experiments/run_0831_1k_20ep_1024_lightdepth_stage_a_magformer.sh' --register '${REGISTER}' --dataset-root '${DATASET_ROOT}' --output-root '${OUTPUT_ROOT_512}' --variant mobilenetv3_sagate_edge_validhole --num-gpus 1 --image-size 512 --${MODE}"

run_if_missing \
  "yolov8_seg_n_pretrained_512" \
  "${OUTPUT_ROOT_512}/yolov8_seg_n_pretrained/metrics.cocoeval.json" \
  "cd '${REPO_ROOT}' && CUDA_VISIBLE_DEVICES=${GPU} bash '${REPO_ROOT}/scripts/experiments/run_0831_1k_20ep_scratch_yolov8_seg.sh' --register '${REGISTER}' --dataset-root '${DATASET_ROOT}' --output-root '${OUTPUT_ROOT_512}' --image-size 512 --model-size n --pretrained --device 0 --${MODE}"

run_if_missing \
  "yolov8_seg_s_pretrained_512" \
  "${OUTPUT_ROOT_512}/yolov8_seg_s_pretrained/metrics.cocoeval.json" \
  "cd '${REPO_ROOT}' && CUDA_VISIBLE_DEVICES=${GPU} bash '${REPO_ROOT}/scripts/experiments/run_0831_1k_20ep_scratch_yolov8_seg.sh' --register '${REGISTER}' --dataset-root '${DATASET_ROOT}' --output-root '${OUTPUT_ROOT_512}' --image-size 512 --model-size s --pretrained --device 0 --${MODE}"

run_if_missing \
  "yolov8_seg_m_pretrained_512" \
  "${OUTPUT_ROOT_512}/yolov8_seg_m_pretrained/metrics.cocoeval.json" \
  "cd '${REPO_ROOT}' && CUDA_VISIBLE_DEVICES=${GPU} bash '${REPO_ROOT}/scripts/experiments/run_0831_1k_20ep_scratch_yolov8_seg.sh' --register '${REGISTER}' --dataset-root '${DATASET_ROOT}' --output-root '${OUTPUT_ROOT_512}' --image-size 512 --model-size m --pretrained --device 0 --${MODE}"

run_if_missing \
  "yolov8_seg_l_pretrained_512" \
  "${OUTPUT_ROOT_512}/yolov8_seg_l_pretrained/metrics.cocoeval.json" \
  "cd '${REPO_ROOT}' && CUDA_VISIBLE_DEVICES=${GPU} bash '${REPO_ROOT}/scripts/experiments/run_0831_1k_20ep_scratch_yolov8_seg.sh' --register '${REGISTER}' --dataset-root '${DATASET_ROOT}' --output-root '${OUTPUT_ROOT_512}' --image-size 512 --model-size l --pretrained --device 0 --${MODE}"

run_if_missing \
  "uoais_512" \
  "${OUTPUT_ROOT_512}/uoais_scratch/metrics.cocoeval.json" \
  "cd '${REPO_ROOT}' && CUDA_VISIBLE_DEVICES=${GPU} bash '${REPO_ROOT}/scripts/experiments/run_0831_1k_20ep_scratch_uoais.sh' --register '${REGISTER}' --dataset-root '${DATASET_ROOT}' --output-root '${OUTPUT_ROOT_512}' --image-size 512 --${MODE}"

run_if_missing \
  "unet_semantic_inst_512" \
  "${OUTPUT_ROOT_512}/unet_semantic_inst/metrics.cocoeval.json" \
  "cd '${REPO_ROOT}' && CUDA_VISIBLE_DEVICES=${GPU} bash '${REPO_ROOT}/scripts/experiments/run_0831_1k_20ep_1024_revisit_unet_semantic_inst.sh' --register '${REGISTER}' --dataset-root '${DATASET_ROOT}' --output-root '${OUTPUT_ROOT_512}' --image-size 512 --${MODE}"

run_if_missing \
  "unet_boundary_inst_512" \
  "${OUTPUT_ROOT_512}/unet_boundary_inst/metrics.cocoeval.json" \
  "cd '${REPO_ROOT}' && CUDA_VISIBLE_DEVICES=${GPU} bash '${REPO_ROOT}/scripts/experiments/run_0831_1k_20ep_1024_revisit_unet_boundary_inst.sh' --register '${REGISTER}' --dataset-root '${DATASET_ROOT}' --output-root '${OUTPUT_ROOT_512}' --image-size 512 --${MODE}"

run_if_missing \
  "unetpp_boundary_inst_512" \
  "${OUTPUT_ROOT_512}/unetpp_boundary_inst/metrics.cocoeval.json" \
  "cd '${REPO_ROOT}' && CUDA_VISIBLE_DEVICES=${GPU} bash '${REPO_ROOT}/scripts/experiments/run_0831_1k_20ep_1024_revisit_unetpp_boundary_inst.sh' --register '${REGISTER}' --dataset-root '${DATASET_ROOT}' --output-root '${OUTPUT_ROOT_512}' --image-size 512 --${MODE}"

run_if_missing \
  "msmformer_512" \
  "${OUTPUT_ROOT_512}/msmformer/metrics.cocoeval.json" \
  "cd '${REPO_ROOT}' && CUDA_VISIBLE_DEVICES=${GPU} bash '${REPO_ROOT}/scripts/experiments/run_0831_1k_20ep_scratch_msmformer.sh' --register '${REGISTER}' --dataset-root '${DATASET_ROOT}' --output-root '${OUTPUT_ROOT_512}' --image-size 512 --${MODE}"

run_if_missing \
  "ucn_512" \
  "${OUTPUT_ROOT_512}/ucn/metrics.cocoeval.json" \
  "cd '${REPO_ROOT}' && CUDA_VISIBLE_DEVICES=${GPU} bash '${REPO_ROOT}/scripts/experiments/run_0831_1k_20ep_scratch_ucn.sh' --register '${REGISTER}' --dataset-root '${DATASET_ROOT}' --output-root '${OUTPUT_ROOT_512}' --image-size 512 --${MODE}"

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

run_if_missing \
  "ucn_1024" \
  "${OUTPUT_ROOT_1024}/ucn/metrics.cocoeval.json" \
  "cd '${REPO_ROOT}' && CUDA_VISIBLE_DEVICES=${GPU} bash '${REPO_ROOT}/scripts/experiments/run_0831_1k_20ep_scratch_ucn.sh' --register '${REGISTER}' --dataset-root '${DATASET_ROOT}' --output-root '${OUTPUT_ROOT_1024}' --image-size 1024 --${MODE}"

run_if_missing \
  "iaunet_1024" \
  "${OUTPUT_ROOT_1024}/iaunet/metrics.cocoeval.json" \
  "cd '${REPO_ROOT}' && CUDA_VISIBLE_DEVICES=${GPU} bash '${REPO_ROOT}/scripts/experiments/run_0831_1k_20ep_1024_revisit_iaunet_inst.sh' --register '${REGISTER}' --dataset-root '${DATASET_ROOT}' --output-root '${OUTPUT_ROOT_1024}' --image-size 1024 --${MODE}"

run_if_missing \
  "cellpose_1024" \
  "${OUTPUT_ROOT_1024}/cellpose/metrics.cocoeval.json" \
  "cd '${REPO_ROOT}' && CUDA_VISIBLE_DEVICES=${GPU} bash '${REPO_ROOT}/scripts/experiments/run_0831_1k_20ep_1024_revisit_cellpose_inst.sh' --register '${REGISTER}' --dataset-root '${DATASET_ROOT}' --output-root '${OUTPUT_ROOT_1024}' --image-size 1024 --${MODE}"

run_if_missing \
  "stardist_1024" \
  "${OUTPUT_ROOT_1024}/stardist/metrics.cocoeval.json" \
  "cd '${REPO_ROOT}' && CUDA_VISIBLE_DEVICES=${GPU} bash '${REPO_ROOT}/scripts/experiments/run_0831_1k_20ep_1024_revisit_stardist_inst.sh' --register '${REGISTER}' --dataset-root '${DATASET_ROOT}' --output-root '${OUTPUT_ROOT_1024}' --image-size 1024 --${MODE}"

runner_log "${MODE}" "${RUN_LOG}" "[non256-gpu1] done"
