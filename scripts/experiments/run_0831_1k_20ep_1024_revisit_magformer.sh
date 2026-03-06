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
SMOKE=0
VARIANT="depthnorm_on" # depthnorm_on | nodpth_ref
NUM_WORKERS=4

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-root)
      DATASET_ROOT="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --variant)
      VARIANT="$2"
      shift 2
      ;;
    --smoke)
      SMOKE=1
      shift
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

case "${VARIANT}" in
  depthnorm_on)
    MODEL_ID="magformer_depthnorm_on"
    CFG_BASE="${REPO_ROOT}/configs/magformer_0831_1k_20ep_1024_depthnorm_on.yaml"
    ;;
  nodpth_ref)
    MODEL_ID="magformer_nodpth_ref"
    CFG_BASE="${REPO_ROOT}/configs/magformer_0831_1k_20ep_1024_nodpth_ref.yaml"
    ;;
  *)
    echo "Unsupported --variant: ${VARIANT}" >&2
    exit 1
    ;;
esac

OUT="${OUTPUT_ROOT}/${MODEL_ID}"
mkdir -p "${OUT}"
mkdir -p "${OUT}/visualizations"
OUT="$(cd "${OUT}" && pwd)"
DATASET_ROOT="$(cd "${DATASET_ROOT}" && pwd)"
RUN_LOG="$(runner_setup_log "${OUT}" "${MODE}")"

runner_log "${MODE}" "${RUN_LOG}" "[0831-1k-20ep-1024-revisit-magformer] mode=${MODE}"
runner_log "${MODE}" "${RUN_LOG}" "[0831-1k-20ep-1024-revisit-magformer] smoke=${SMOKE}"
runner_log "${MODE}" "${RUN_LOG}" "[0831-1k-20ep-1024-revisit-magformer] variant=${VARIANT}"
runner_log "${MODE}" "${RUN_LOG}" "[0831-1k-20ep-1024-revisit-magformer] dataset_root=${DATASET_ROOT}"
runner_log "${MODE}" "${RUN_LOG}" "[0831-1k-20ep-1024-revisit-magformer] output_dir=${OUT}"

IMS_PER_BATCH=4
EPOCHS=20
if [[ "${SMOKE}" == "1" ]]; then
  IMS_PER_BATCH=1
  EPOCHS=1
  NUM_WORKERS=2
fi

NUM_IMAGES="$(ecc_num_train_images "${DATASET_ROOT}")"
ITERS_PER_EPOCH="$(ecc_iters_per_epoch "${NUM_IMAGES}" "${IMS_PER_BATCH}")"
MAX_ITER=$(( ITERS_PER_EPOCH * EPOCHS ))
STEP1=$(( MAX_ITER * 8 / 10 ))
STEP2=$(( MAX_ITER * 9 / 10 ))
STEPS="${STEP1},${STEP2}"
WARMUP_ITERS="${ITERS_PER_EPOCH}"
EVAL_PERIOD="${ITERS_PER_EPOCH}"
CHECKPOINT_PERIOD="${ITERS_PER_EPOCH}"
BASE_LR="0.00005"

if [[ "${SMOKE}" == "1" ]]; then
  MAX_ITER=20
  STEPS="15,18"
  WARMUP_ITERS=10
  EVAL_PERIOD=10
  CHECKPOINT_PERIOD=10
fi

METADATA_ARGS=(
  bash
  "$(basename "${BASH_SOURCE[0]}")"
  --dataset-root
  "${DATASET_ROOT}"
  --output-root
  "${OUTPUT_ROOT}"
  --variant
  "${VARIANT}"
)
if [[ "${MODE}" == "run" ]]; then
  METADATA_ARGS+=(--run)
else
  METADATA_ARGS+=(--dry-run)
fi
if [[ "${SMOKE}" == "1" ]]; then
  METADATA_ARGS+=(--smoke)
fi
METADATA_CMD="$(printf "%q " "${METADATA_ARGS[@]}")"
METADATA_CMD="${METADATA_CMD% }"

runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && conda run -n magformer python scripts/analysis/write_run_metadata.py \
  --phase start \
  --out-dir '${OUT}' \
  --track 0831_1k_20ep_1024_depth_revisit \
  --register '0831' \
  --dataset-root '${DATASET_ROOT}' \
  --model-id '${MODEL_ID}' \
  --candidate-id '${VARIANT}' \
  --run-tag 'final' \
  --command \"${METADATA_CMD}\" \
  --iters-per-epoch ${ITERS_PER_EPOCH} \
  --max-iter ${MAX_ITER} \
  --epochs ${EPOCHS} \
  --ims-per-batch ${IMS_PER_BATCH}"

RUNTIME_CFG="${OUT}/magformer_runtime_config.yaml"
RUN_NAME="0831_1k_20ep_1024_${VARIANT}"
runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && conda run -n magformer python scripts/analysis/render_magformer_runtime_config.py \
  --base-config '${CFG_BASE}' \
  --out-config '${RUNTIME_CFG}' \
  --output-dir '${OUT}' \
  --run-name '${RUN_NAME}' \
  --base-lr ${BASE_LR} \
  --max-iter ${MAX_ITER} \
  --steps '${STEPS}' \
  --warmup-iters ${WARMUP_ITERS} \
  --ims-per-batch ${IMS_PER_BATCH} \
  --eval-period ${EVAL_PERIOD} \
  --checkpoint-period ${CHECKPOINT_PERIOD} \
  --num-workers ${NUM_WORKERS}"

runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && conda run -n magformer python tools/train.py \
  --config '${RUNTIME_CFG}' \
  --dataset-root '${DATASET_ROOT}' \
  --output-dir '${OUT}' \
  --num-workers ${NUM_WORKERS}"

runner_exec "${MODE}" "${RUN_LOG}" "conda run -n magformer python '${REPO_ROOT}/scripts/analysis/write_params_from_magformer_ckpt.py' --out-dir '${OUT}'"
runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && conda run -n magformer python scripts/experiments/postprocess_cocoeval.py --dataset-root '${DATASET_ROOT}' --out-dir '${OUT}' --metrics-out 'metrics.cocoeval.json'"
runner_exec "${MODE}" "${RUN_LOG}" "conda run -n magformer python '${REPO_ROOT}/scripts/analysis/write_metrics_std.py' --out-dir '${OUT}' --iters-per-epoch ${ITERS_PER_EPOCH}"
runner_exec "${MODE}" "${RUN_LOG}" "conda run -n magformer python '${REPO_ROOT}/scripts/analysis/prune_checkpoints.py' --out-dir '${OUT}' --framework magformer"
runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && conda run -n magformer python scripts/analysis/write_run_metadata.py --phase end --out-dir '${OUT}'"

runner_log "${MODE}" "${RUN_LOG}" "[0831-1k-20ep-1024-revisit-magformer] done"
