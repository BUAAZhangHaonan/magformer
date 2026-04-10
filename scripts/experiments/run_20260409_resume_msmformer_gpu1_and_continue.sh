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
QUEUE_TAG="20260409_gpu1_resume_msmformer_non256"
IMAGE_SIZE=512

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
    --queue-tag)
      QUEUE_TAG="$2"
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

OUT="${OUTPUT_BASE}/${DATE_TAG}_1k_1566_20ep_512_full19/msmformer"
DONE_MARKER="${OUT}/metrics.cocoeval.json"
LAST_CHECKPOINT_FILE="${OUT}/last_checkpoint"
RUN_LOG="$(runner_setup_log "${OUTPUT_BASE}/${QUEUE_TAG}" "${MODE}")"

MSMFORMER_ROOT="${REPO_ROOT}/baselines/msmformer/MSMFormer"
CFG="${REPO_ROOT}/configs/baselines/msmformer_0831_1k_tracks.yaml"
LOCAL_PRETRAINED="${REPO_ROOT}/output/pretrained/norm_RGBD_pretrained.pth"
PRETRAINED_ARG=""
if [[ -f "${LOCAL_PRETRAINED}" ]]; then
  PRETRAINED_ARG="--pretrained '${LOCAL_PRETRAINED}'"
fi

if [[ -d "${DATASET_ROOT}/images/train" ]]; then
  read -r DATASET_NAME_TRAIN DATASET_NAME_VAL < <(ecc_dataset_names_coco_rgbd "${REGISTER}" "${DATASET_ROOT}")
  read -r PIXEL_MEAN PIXEL_STD < <(ecc_read_rgb_stats_bgr "${REGISTER}" "${DATASET_ROOT}")
else
  DATASET_NAME_TRAIN="ecc20260318_1k_1566_rgbd_train"
  DATASET_NAME_VAL="ecc20260318_1k_1566_rgbd_val"
  PIXEL_MEAN="[113.4361,114.1293,116.7070]"
  PIXEL_STD="[28.8131,28.6797,27.9589]"
fi

MAX_ITER=6320
SOLVER_STEPS="(5056,5688)"
BASE_LR="0.0001"
WARMUP_ITERS=316
IMS_PER_BATCH=4
CHECKPOINT_PERIOD=316
EVAL_PERIOD=316

if [[ ! -f "${DONE_MARKER}" && ! -f "${LAST_CHECKPOINT_FILE}" ]]; then
  LATEST_MODEL="$(ls "${OUT}"/model_*.pth 2>/dev/null | sort | tail -n 1 || true)"
  if [[ -n "${LATEST_MODEL}" ]]; then
    printf '%s\n' "$(basename "${LATEST_MODEL}")" > "${LAST_CHECKPOINT_FILE}"
  fi
fi

LAST_CHECKPOINT_VALUE=""
if [[ -f "${LAST_CHECKPOINT_FILE}" ]]; then
  LAST_CHECKPOINT_VALUE="$(tr -d '\r\n' < "${LAST_CHECKPOINT_FILE}")"
fi

runner_log "${MODE}" "${RUN_LOG}" "[gpu1-resume-msmformer] cuda_visible_devices=${GPU}"
runner_log "${MODE}" "${RUN_LOG}" "[gpu1-resume-msmformer] dataset_root=${DATASET_ROOT}"
runner_log "${MODE}" "${RUN_LOG}" "[gpu1-resume-msmformer] out_dir=${OUT}"
runner_log "${MODE}" "${RUN_LOG}" "[gpu1-resume-msmformer] done_marker=${DONE_MARKER}"
runner_log "${MODE}" "${RUN_LOG}" "[gpu1-resume-msmformer] last_checkpoint=${LAST_CHECKPOINT_VALUE}"
runner_log "${MODE}" "${RUN_LOG}" "[gpu1-resume-msmformer] wait_free_mb=${WAIT_FREE_MB} wait_sleep_sec=${WAIT_SLEEP_SEC}"

if [[ ! -f "${DONE_MARKER}" && -z "${LAST_CHECKPOINT_VALUE}" ]]; then
  echo "No last_checkpoint found under ${OUT}" >&2
  exit 1
fi

if [[ -f "${DONE_MARKER}" ]]; then
  runner_log "${MODE}" "${RUN_LOG}" "[gpu1-resume-msmformer] skip resume: ${DONE_MARKER}"
else
  runner_wait_for_free_gpu_mb "${MODE}" "${RUN_LOG}" "${WAIT_FREE_MB}" "${WAIT_SLEEP_SEC}" "msmformer_512_resume"
  runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && CUDA_VISIBLE_DEVICES=${GPU} conda run -n magformer python baselines/run_msmformer_ecc.py \
    --register '${REGISTER}' \
    --dataset-root '${DATASET_ROOT}' \
    --msmformer-root '${MSMFORMER_ROOT}' \
    ${PRETRAINED_ARG} \
    -- \
    --resume \
    --num-gpus 1 \
    --config-file '${CFG}' \
    DATASETS.TRAIN \"('${DATASET_NAME_TRAIN}',)\" \
    DATASETS.TEST \"('${DATASET_NAME_VAL}',)\" \
    MODEL.PIXEL_MEAN '${PIXEL_MEAN}' \
    MODEL.PIXEL_STD '${PIXEL_STD}' \
    MODEL.BACKBONE.FREEZE_AT 0 \
    SOLVER.MAX_ITER ${MAX_ITER} \
    SOLVER.STEPS '${SOLVER_STEPS}' \
    SOLVER.BASE_LR ${BASE_LR} \
    SOLVER.WARMUP_ITERS ${WARMUP_ITERS} \
    SOLVER.IMS_PER_BATCH ${IMS_PER_BATCH} \
    SOLVER.CHECKPOINT_PERIOD ${CHECKPOINT_PERIOD} \
    TEST.EVAL_PERIOD ${EVAL_PERIOD} \
    INPUT.MIN_SIZE_TRAIN '(${IMAGE_SIZE},)' \
    INPUT.MAX_SIZE_TRAIN ${IMAGE_SIZE} \
    INPUT.MIN_SIZE_TEST ${IMAGE_SIZE} \
    INPUT.MAX_SIZE_TEST ${IMAGE_SIZE} \
    OUTPUT_DIR '${OUT}'"

  runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && conda run -n magformer python scripts/experiments/postprocess_cocoeval.py --dataset-root '${DATASET_ROOT}' --out-dir '${OUT}' --metrics-out 'metrics.cocoeval.json'"
  runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && conda run -n magformer python scripts/analysis/write_metrics_std.py --out-dir '${OUT}' --iters-per-epoch 316"
  runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && conda run -n magformer python scripts/analysis/prune_checkpoints.py --out-dir '${OUT}' --framework detectron2"
  runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && conda run -n magformer python scripts/analysis/write_run_metadata.py --phase end --out-dir '${OUT}'"
fi

runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && bash scripts/experiments/run_20260409_non256_completion_gpu1.sh \
  --register '${REGISTER}' \
  --dataset-root '${DATASET_ROOT}' \
  --output-base '${OUTPUT_BASE}' \
  --gpu ${GPU} \
  --queue-tag 20260409_gpu1_non256_backfill \
  --wait-free-mb ${WAIT_FREE_MB} \
  --wait-sleep-sec ${WAIT_SLEEP_SEC} \
  --${MODE}"

runner_log "${MODE}" "${RUN_LOG}" "[gpu1-resume-msmformer] done"
