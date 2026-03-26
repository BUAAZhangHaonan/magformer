#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
source "${SCRIPT_DIR}/common_runner.sh"
source "${SCRIPT_DIR}/ecc_common.sh"

DATASET_ROOT="${PROJECT_ROOT}/magformer_datasets/0831_1K"
OUTPUT_ROOT="${REPO_ROOT}/output/experiments/0831_1k_20ep_1024_depth_revisit"
MODE="run"
MODEL_ID="unet_boundary_inst"
REGISTER="0831"
IMAGE_SIZE=1024

while [[ $# -gt 0 ]]; do
  case "$1" in
    --register) REGISTER="$2"; shift 2 ;;
    --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --image-size) IMAGE_SIZE="$2"; shift 2 ;;
    --run) MODE="run"; shift ;;
    --dry-run) MODE="dry-run"; shift ;;

    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

OUT="${OUTPUT_ROOT}/${MODEL_ID}"
mkdir -p "${OUT}" "${OUT}/visualizations"
OUT="$(cd "${OUT}" && pwd)"
DATASET_ROOT="$(cd "${DATASET_ROOT}" && pwd)"
RUN_LOG="$(runner_setup_log "${OUT}" "${MODE}")"
read -r RGB_MEAN RGB_STD < <(ecc_read_rgb_stats_rgb "${REGISTER}" "${DATASET_ROOT}")
read -r DEPTH_CLIP_MIN DEPTH_CLIP_MAX < <(ecc_read_depth_clip "${REGISTER}" "${DATASET_ROOT}")

EPOCHS=20
BATCH=4
NUM_WORKERS=4
EXTRA_ARGS=()
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
runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && conda run -n magformer python baselines/run_unet_instance_ecc.py --dataset-root '${DATASET_ROOT}' --output-dir '${OUT}' --variant '${MODEL_ID}' --image-size ${IMAGE_SIZE} --epochs ${EPOCHS} --batch ${BATCH} --num-workers ${NUM_WORKERS} --rgb-mean '${RGB_MEAN}' --rgb-std '${RGB_STD}' --depth-clip-min ${DEPTH_CLIP_MIN} --depth-clip-max ${DEPTH_CLIP_MAX} ${EXTRA_ARGS[*]}"
runner_exec "${MODE}" "${RUN_LOG}" "conda run -n magformer python '${REPO_ROOT}/scripts/analysis/write_metrics_std.py' --out-dir '${OUT}' --framework detectron2 --iters-per-epoch ${ITERS_PER_EPOCH}"
runner_exec "${MODE}" "${RUN_LOG}" "conda run -n magformer python '${REPO_ROOT}/scripts/analysis/prune_checkpoints.py' --out-dir '${OUT}' --framework detectron2"
runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && conda run -n magformer python scripts/analysis/write_run_metadata.py --phase end --out-dir '${OUT}'"
