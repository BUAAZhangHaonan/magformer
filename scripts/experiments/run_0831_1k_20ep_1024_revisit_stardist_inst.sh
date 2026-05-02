#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
source "${SCRIPT_DIR}/common_runner.sh"
source "${SCRIPT_DIR}/ecc_common.sh"

DATASET_ROOT="${PROJECT_ROOT}/magformer_datasets/20260318_1K_1566"
OUTPUT_ROOT="${REPO_ROOT}/output/experiments/20260318_1k_1566_20ep_1024_full19"
MODE="run"
MODEL_ID="stardist"
REGISTER="20260318_1K_1566"
IMAGE_SIZE=1024
EPOCHS=20
BATCH=4
# StarDist loads the resized train/val arrays in-process before Keras training.
# Keep workers at zero to avoid duplicating that memory on the shared server.
NUM_WORKERS=0
RAM_LIMIT_PCT=50
STARDIST_ENV="${STARDIST_ENV:-stardist}"
TF_ENV_PREFIX="TF_FORCE_GPU_ALLOW_GROWTH=true TF_CPP_MIN_LOG_LEVEL=1 TF_NUM_INTRAOP_THREADS=${TF_NUM_INTRAOP_THREADS:-4} TF_NUM_INTEROP_THREADS=${TF_NUM_INTEROP_THREADS:-2} OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MALLOC_ARENA_MAX=${MALLOC_ARENA_MAX:-2}"
MAX_TRAIN_IMAGES=0
MAX_VAL_IMAGES=0
TRAIN_N_VAL_PATCHES=8

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
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --image-size)
      IMAGE_SIZE="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --batch)
      BATCH="$2"
      shift 2
      ;;
    --num-workers)
      NUM_WORKERS="$2"
      shift 2
      ;;
    --ram-limit-pct)
      RAM_LIMIT_PCT="$2"
      shift 2
      ;;
    --max-train-images)
      MAX_TRAIN_IMAGES="$2"
      shift 2
      ;;
    --max-val-images)
      MAX_VAL_IMAGES="$2"
      shift 2
      ;;
    --train-n-val-patches)
      TRAIN_N_VAL_PATCHES="$2"
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
if [[ "${IMAGE_SIZE}" != "512" && "${IMAGE_SIZE}" != "1024" ]]; then
  echo "Unsupported --image-size ${IMAGE_SIZE}; expected 512 or 1024" >&2
  exit 1
fi

OUT="${OUTPUT_ROOT}/${MODEL_ID}"
mkdir -p "${OUT}" "${OUT}/visualizations"
OUT="$(cd "${OUT}" && pwd)"
DATASET_ROOT="$(cd "${DATASET_ROOT}" && pwd)"
RUN_LOG="$(runner_setup_log "${OUT}" "${MODE}")"
NUM_IMAGES="$(ecc_num_train_images "${DATASET_ROOT}")"
ITERS_PER_EPOCH="$(ecc_iters_per_epoch "${NUM_IMAGES}" "${BATCH}")"
MAX_ITER=$((EPOCHS * ITERS_PER_EPOCH))

METADATA_ARGS=(
  bash
  "$(basename "${BASH_SOURCE[0]}")"
  --register "${REGISTER}"
  --dataset-root "${DATASET_ROOT}"
  --output-root "${OUTPUT_ROOT}"
  --image-size "${IMAGE_SIZE}"
  --epochs "${EPOCHS}"
  --batch "${BATCH}"
  --num-workers "${NUM_WORKERS}"
  --ram-limit-pct "${RAM_LIMIT_PCT}"
  --max-train-images "${MAX_TRAIN_IMAGES}"
  --max-val-images "${MAX_VAL_IMAGES}"
  --train-n-val-patches "${TRAIN_N_VAL_PATCHES}"
)
if [[ "${MODE}" == "run" ]]; then
  METADATA_ARGS+=(--run)
else
  METADATA_ARGS+=(--dry-run)
fi
METADATA_CMD="$(printf "%q " "${METADATA_ARGS[@]}")"
METADATA_CMD="${METADATA_CMD% }"

runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && conda run -n magformer python scripts/analysis/write_run_metadata.py --phase start --out-dir '${OUT}' --track $(basename "${OUTPUT_ROOT}") --register '${REGISTER}' --dataset-root '${DATASET_ROOT}' --model-id '${MODEL_ID}' --candidate-id 'C1' --run-tag 'final' --command \"${METADATA_CMD}\" --iters-per-epoch ${ITERS_PER_EPOCH} --max-iter ${MAX_ITER} --epochs ${EPOCHS} --ims-per-batch ${BATCH}"
if [[ "${MODE}" == "run" ]]; then
  runner_log_launch_guard_snapshot "${MODE}" "${RUN_LOG}" "stardist-${IMAGE_SIZE}"
fi
runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && ${TF_ENV_PREFIX} conda run -n '${STARDIST_ENV}' python baselines/run_stardist_instance_ecc.py --dataset-root '${DATASET_ROOT}' --output-dir '${OUT}' --image-size ${IMAGE_SIZE} --epochs ${EPOCHS} --batch ${BATCH} --num-workers ${NUM_WORKERS} --ram-limit-pct ${RAM_LIMIT_PCT} --max-train-images ${MAX_TRAIN_IMAGES} --max-val-images ${MAX_VAL_IMAGES} --train-n-val-patches ${TRAIN_N_VAL_PATCHES}"
runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && conda run -n magformer python scripts/analysis/write_metrics_std.py --out-dir '${OUT}' --iters-per-epoch ${ITERS_PER_EPOCH}"
runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && conda run -n magformer python scripts/analysis/prune_checkpoints.py --out-dir '${OUT}' --framework auto"
runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && conda run -n magformer python scripts/analysis/write_run_metadata.py --phase end --out-dir '${OUT}'"
runner_log "${MODE}" "${RUN_LOG}" "[stardist-20260318] done"
