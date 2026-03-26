#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
source "${SCRIPT_DIR}/common_runner.sh"
source "${SCRIPT_DIR}/ecc_common.sh"

OUTPUT_ROOT_DEFAULT="${REPO_ROOT}/output/experiments/0831_1k_20ep"

REGISTER="0831"
DATASET_ROOT=""
OUTPUT_ROOT="${OUTPUT_ROOT_DEFAULT}"
MODE="run"
CANDIDATE_ID="C1"
RUN_TAG="final"
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
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --candidate-id)
      CANDIDATE_ID="$2"
      shift 2
      ;;
    --run-tag)
      RUN_TAG="$2"
      shift 2
      ;;
    --image-size)
      IMAGE_SIZE="$2"
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

REGISTER_RAW="${REGISTER}"
REGISTER="$(ecc_normalize_register "${REGISTER}")"
if [[ -z "${DATASET_ROOT}" ]]; then
  DATASET_ROOT="$(ecc_default_dataset_root "${REGISTER_RAW}")"
fi

LOCAL_PRETRAINED="${REPO_ROOT}/output/pretrained/seg_resnet34_8s_embedding_cosine_rgbd_add_sampling_epoch_16.checkpoint.pth"
MODEL_ID="ucn"
if [[ "${RUN_TAG}" == "final" ]]; then
  OUT="${OUTPUT_ROOT}/${MODEL_ID}"
else
  OUT="${OUTPUT_ROOT}/_tuning/${MODEL_ID}/${CANDIDATE_ID}"
fi

mkdir -p "${OUT}"
mkdir -p "${OUT}/visualizations"
OUT="$(cd "${OUT}" && pwd)"
DATASET_ROOT="$(cd "${DATASET_ROOT}" && pwd)"
RUN_LOG="$(runner_setup_log "${OUT}" "${MODE}")"

runner_log "${MODE}" "${RUN_LOG}" "[ucn-0831-1k-20ep] mode=${MODE}"
runner_log "${MODE}" "${RUN_LOG}" "[ucn-0831-1k-20ep] register=${REGISTER_RAW}"
runner_log "${MODE}" "${RUN_LOG}" "[ucn-0831-1k-20ep] run_tag=${RUN_TAG} candidate=${CANDIDATE_ID}"
runner_log "${MODE}" "${RUN_LOG}" "[ucn-0831-1k-20ep] dataset_root=${DATASET_ROOT}"
runner_log "${MODE}" "${RUN_LOG}" "[ucn-0831-1k-20ep] output_dir=${OUT}"
runner_log "${MODE}" "${RUN_LOG}" "[ucn-0831-1k-20ep] image_size=${IMAGE_SIZE}"
PRETRAINED_ARG=""
if [[ -f "${LOCAL_PRETRAINED}" ]]; then
  PRETRAINED_ARG="--pretrained '${LOCAL_PRETRAINED}'"
  runner_log "${MODE}" "${RUN_LOG}" "[ucn-0831-1k-20ep] pretrained=${LOCAL_PRETRAINED}"
else
  runner_log "${MODE}" "${RUN_LOG}" "[ucn-0831-1k-20ep] pretrained=none"
fi

LR="0.00001"
KAPPA="20"
NUM_SEEDS="100"
case "${CANDIDATE_ID}" in
  C1) LR="0.00001"; KAPPA="20"; NUM_SEEDS="100" ;;
  C2) LR="0.00002"; KAPPA="20"; NUM_SEEDS="100" ;;
  C3) LR="0.000005"; KAPPA="20"; NUM_SEEDS="100" ;;
  C4) LR="0.00001"; KAPPA="20"; NUM_SEEDS="150" ;;
  *)
    echo "Unsupported --candidate-id: ${CANDIDATE_ID}" >&2
    exit 1
    ;;
esac

EPOCHS=20
BATCH=16
if [[ "${RUN_TAG}" == "sweep" ]]; then
  EPOCHS=5
fi

NUM_IMAGES="$(ecc_num_train_images "${DATASET_ROOT}")"
ITERS_PER_EPOCH="$(ecc_iters_per_epoch "${NUM_IMAGES}" "${BATCH}")"
MAX_ITER=$(( ITERS_PER_EPOCH * EPOCHS ))

METADATA_ARGS=(
  bash
  "$(basename "${BASH_SOURCE[0]}")"
  --register
  "${REGISTER_RAW}"
  --dataset-root
  "${DATASET_ROOT}"
  --output-root
  "${OUTPUT_ROOT}"
  --candidate-id
  "${CANDIDATE_ID}"
  --run-tag
  "${RUN_TAG}"
  --image-size
  "${IMAGE_SIZE}"
)
if [[ "${MODE}" == "run" ]]; then
  METADATA_ARGS+=(--run)
else
  METADATA_ARGS+=(--dry-run)
fi
METADATA_CMD="$(printf "%q " "${METADATA_ARGS[@]}")"
METADATA_CMD="${METADATA_CMD% }"
runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && conda run -n magformer python scripts/analysis/write_run_metadata.py \
  --phase start \
  --out-dir '${OUT}' \
  --track tracks \
  --register '${REGISTER_RAW}' \
  --dataset-root '${DATASET_ROOT}' \
  --model-id '${MODEL_ID}' \
  --candidate-id '${CANDIDATE_ID}' \
  --run-tag '${RUN_TAG}' \
  --command \"${METADATA_CMD}\" \
  --iters-per-epoch ${ITERS_PER_EPOCH} \
  --max-iter ${MAX_ITER} \
  --epochs ${EPOCHS} \
  --ims-per-batch ${BATCH}"

run_train_cmd() {
  local batch="$1"
  local cmd="cd '${REPO_ROOT}' && conda run -n magformer python baselines/run_ucn_ecc.py \
    --register '${REGISTER_RAW}' \
    --dataset-root '${DATASET_ROOT}' \
    --output-dir '${OUT}' \
    ${PRETRAINED_ARG} \
    --epochs ${EPOCHS} \
    --batch ${batch} \
    --img-size ${IMAGE_SIZE} \
    --lr ${LR} \
    --kappa ${KAPPA} \
    --num-seeds ${NUM_SEEDS}"

  runner_log "${MODE}" "${RUN_LOG}" "+ ${cmd}"
  if [[ "${MODE}" != "run" ]]; then
    return 0
  fi
  set +e
  eval "${cmd}" 2>&1 | tee -a "${RUN_LOG}"
  local rc=${PIPESTATUS[0]}
  set -e
  return "${rc}"
}

SECONDS=0
if run_train_cmd "${BATCH}"; then
  :
else
  if [[ "${MODE}" != "run" ]]; then
    runner_log "${MODE}" "${RUN_LOG}" "FAILED rc=1"
    exit 1
  fi
  if rg -qi "outofmemoryerror|cuda out of memory" "${RUN_LOG}"; then
    runner_log "${MODE}" "${RUN_LOG}" "[ucn-0831-1k-20ep] OOM detected, retry with batch=8 (same epochs)"
    cat > "${OUT}/notes_oom.txt" <<EON
OOM fallback activated for ${MODEL_ID}.
Original: batch=${BATCH} epochs=${EPOCHS}
Fallback: batch=8 epochs=${EPOCHS}
EON
    rm -f "${OUT}"/checkpoint_iter_*.pth "${OUT}"/model_best.pth "${OUT}"/metrics.cocoeval.json "${OUT}"/coco_instances_results.json || true
    if ! run_train_cmd "8"; then
      runner_log "${MODE}" "${RUN_LOG}" "FAILED rc=1 (fallback also failed)"
      exit 1
    fi
  else
    runner_log "${MODE}" "${RUN_LOG}" "FAILED rc=1"
    exit 1
  fi
fi

echo "${SECONDS}" > "${OUT}/wall_time_sec.txt"

if [[ "${MODE}" == "run" ]]; then
  runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && conda run -n magformer python scripts/experiments/postprocess_cocoeval.py --dataset-root '${DATASET_ROOT}' --out-dir '${OUT}' --metrics-out 'metrics.cocoeval.json'"
  runner_exec "${MODE}" "${RUN_LOG}" "conda run -n magformer python '${REPO_ROOT}/scripts/analysis/write_metrics_std.py' --out-dir '${OUT}' --iters-per-epoch ${ITERS_PER_EPOCH}"
  runner_exec "${MODE}" "${RUN_LOG}" "conda run -n magformer python '${REPO_ROOT}/scripts/analysis/prune_checkpoints.py' --out-dir '${OUT}' --framework ucn"
  runner_exec "${MODE}" "${RUN_LOG}" "conda run -n magformer python '${REPO_ROOT}/scripts/analysis/write_run_metadata.py' --phase end --out-dir '${OUT}'"
fi

runner_log "${MODE}" "${RUN_LOG}" "[ucn-0831-1k-20ep] done"
