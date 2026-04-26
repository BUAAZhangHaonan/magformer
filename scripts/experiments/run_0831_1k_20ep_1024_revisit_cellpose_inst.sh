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
MODEL_ID="cellpose"
REGISTER="20260318_1K_1566"
IMAGE_SIZE=1024
EPOCHS=20
BATCH=4
# CellPose stores decoded ECC masks in each dataset object. Extra DataLoader
# workers duplicate that state and can push the shared GPU-1 server over 50% RAM.
NUM_WORKERS=0
DEVICE="cuda"

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
    --device)
      DEVICE="$2"
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
)
if [[ "${MODE}" == "run" ]]; then
  METADATA_ARGS+=(--run)
else
  METADATA_ARGS+=(--dry-run)
fi
METADATA_CMD="$(printf "%q " "${METADATA_ARGS[@]}")"
METADATA_CMD="${METADATA_CMD% }"

runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && conda run -n magformer python scripts/analysis/write_run_metadata.py --phase start --out-dir '${OUT}' --track $(basename "${OUTPUT_ROOT}") --register '${REGISTER}' --dataset-root '${DATASET_ROOT}' --model-id '${MODEL_ID}' --candidate-id 'C1' --run-tag 'final' --command \"${METADATA_CMD}\" --iters-per-epoch ${ITERS_PER_EPOCH} --max-iter ${MAX_ITER} --epochs ${EPOCHS} --ims-per-batch ${BATCH}"
runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && conda run -n magformer python baselines/run_cellpose_instance_ecc.py --dataset-root '${DATASET_ROOT}' --output-dir '${OUT}' --image-size ${IMAGE_SIZE} --epochs ${EPOCHS} --batch ${BATCH} --num-workers ${NUM_WORKERS} --device '${DEVICE}'"
runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && conda run -n magformer python scripts/analysis/write_metrics_std.py --out-dir '${OUT}' --iters-per-epoch ${ITERS_PER_EPOCH}"
runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && conda run -n magformer python scripts/analysis/prune_checkpoints.py --out-dir '${OUT}' --framework auto"
runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && conda run -n magformer python scripts/analysis/write_run_metadata.py --phase end --out-dir '${OUT}'"
runner_log "${MODE}" "${RUN_LOG}" "[cellpose-20260318] done"
