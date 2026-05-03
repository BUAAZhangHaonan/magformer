#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
source "${SCRIPT_DIR}/common_runner.sh"
source "${SCRIPT_DIR}/ecc_common.sh"
HF_ENV_PREFIX="$(runner_hf_env_prefix)"

OUTPUT_ROOT_DEFAULT="${REPO_ROOT}/output/experiments/0831_1k_20ep_scratch"

REGISTER="0831"
DATASET_ROOT=""
OUTPUT_ROOT="${OUTPUT_ROOT_DEFAULT}"
MODE="run"
CANDIDATE_ID="C1"
RUN_TAG="final"
IMAGE_SIZE=512
PRETRAINED=0
MODEL_SIZE="n"
DEVICE="0"

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
    --model-size)
      MODEL_SIZE="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --pretrained)
      PRETRAINED=1
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

REGISTER_RAW="${REGISTER}"
REGISTER="$(ecc_normalize_register "${REGISTER}")"
if [[ -z "${DATASET_ROOT}" ]]; then
  DATASET_ROOT="$(ecc_default_dataset_root "${REGISTER_RAW}")"
fi
DATASET_PREFIX="$(ecc_dataset_prefix "${REGISTER_RAW}" "${DATASET_ROOT}")"

case "${MODEL_SIZE}" in
  n|s|m|l|x) ;;
  *)
    echo "Unsupported --model-size: ${MODEL_SIZE}" >&2
    exit 1
    ;;
esac

MODEL_ID="yolov8_seg_${MODEL_SIZE}_scratch"
YOLO_MODEL="yolov8${MODEL_SIZE}-seg.yaml"
YOLO_PRETRAINED="False"
if [[ "${PRETRAINED}" == "1" ]]; then
  MODEL_ID="yolov8_seg_${MODEL_SIZE}_pretrained"
  YOLO_LOCAL_MODEL="${REPO_ROOT}/output/pretrained/yolov8${MODEL_SIZE}-seg.pt"
  if [[ -f "${YOLO_LOCAL_MODEL}" ]]; then
    YOLO_MODEL="${YOLO_LOCAL_MODEL}"
  else
    YOLO_MODEL="yolov8${MODEL_SIZE}-seg.pt"
  fi
  YOLO_PRETRAINED="True"
fi
if [[ "${RUN_TAG}" == "final" ]]; then
  OUT="${OUTPUT_ROOT}/${MODEL_ID}"
else
  OUT="${OUTPUT_ROOT}/_tuning/${MODEL_ID}/${CANDIDATE_ID}"
fi

YOLO_DATA_DIR="${OUTPUT_ROOT}/_shared/yolo_${DATASET_PREFIX}"
YOLO_DATA_YAML="${YOLO_DATA_DIR}/dataset.yaml"

mkdir -p "${OUT}"
mkdir -p "${OUT}/visualizations"
OUT="$(cd "${OUT}" && pwd)"
DATASET_ROOT="$(cd "${DATASET_ROOT}" && pwd)"
RUN_LOG="$(runner_setup_log "${OUT}" "${MODE}")"

runner_log "${MODE}" "${RUN_LOG}" "[yolov8-seg-0831-1k-20ep-scratch] mode=${MODE}"
runner_log "${MODE}" "${RUN_LOG}" "[yolov8-seg-0831-1k-20ep-scratch] run_tag=${RUN_TAG} candidate=${CANDIDATE_ID}"
runner_log "${MODE}" "${RUN_LOG}" "[yolov8-seg-0831-1k-20ep-scratch] dataset_root=${DATASET_ROOT}"
runner_log "${MODE}" "${RUN_LOG}" "[yolov8-seg-0831-1k-20ep-scratch] output_dir=${OUT}"
runner_log "${MODE}" "${RUN_LOG}" "[yolov8-seg-0831-1k-20ep-scratch] image_size=${IMAGE_SIZE}"
runner_log "${MODE}" "${RUN_LOG}" "[yolov8-seg-0831-1k-20ep-scratch] pretrained=${PRETRAINED}"
runner_log "${MODE}" "${RUN_LOG}" "[yolov8-seg-0831-1k-20ep-scratch] model_size=${MODEL_SIZE}"
read -r RGB_MEAN RGB_STD < <(ecc_read_rgb_stats_rgb "${REGISTER_RAW}" "${DATASET_ROOT}")

LR0="0.01"
WARMUP_EPOCHS="3"
COS_LR="False"
case "${CANDIDATE_ID}" in
  C1) LR0="0.01"; WARMUP_EPOCHS="3"; COS_LR="False" ;;
  C2) LR0="0.02"; WARMUP_EPOCHS="3"; COS_LR="False" ;;
  C3) LR0="0.005"; WARMUP_EPOCHS="3"; COS_LR="False" ;;
  C4) LR0="0.01"; WARMUP_EPOCHS="0"; COS_LR="True" ;;
  *)
    echo "Unsupported --candidate-id: ${CANDIDATE_ID}" >&2
    exit 1
    ;;
esac

EPOCHS=20
BATCH=8
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
  --model-size
  "${MODEL_SIZE}"
)
if [[ "${PRETRAINED}" == "1" ]]; then
  METADATA_ARGS+=(--pretrained)
fi
METADATA_ARGS+=(--device "${DEVICE}")
if [[ "${MODE}" == "run" ]]; then
  METADATA_ARGS+=(--run)
else
  METADATA_ARGS+=(--dry-run)
fi
METADATA_CMD="$(printf "%q " "${METADATA_ARGS[@]}")"
METADATA_CMD="${METADATA_CMD% }"
runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && ${HF_ENV_PREFIX}conda run -n magformer python scripts/analysis/write_run_metadata.py \
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

if [[ "${MODE}" == "run" ]]; then
  export POLARS_FORCE_PKG=compat

  if [[ ! -f "${YOLO_DATA_YAML}" ]]; then
    runner_exec "${MODE}" "${RUN_LOG}" "${HF_ENV_PREFIX}conda run -n magformer python '${REPO_ROOT}/baselines/ultralytics_tools/convert_coco_to_yolo_seg.py' --dataset-root '${DATASET_ROOT}' --output-root '${YOLO_DATA_DIR}'"
  fi

  if [[ -d "${OUT}/train" ]]; then
    ts="$(date +%Y%m%d_%H%M%S)"
    mv "${OUT}/train" "${OUT}/train_prev_${ts}"
  fi
fi

run_train_cmd() {
  local batch="$1"
  local cmd="cd '${REPO_ROOT}' && ${HF_ENV_PREFIX}conda run -n magformer python baselines/run_yolo_seg_ecc.py \
    --model '${YOLO_MODEL}' \
    --data '${YOLO_DATA_YAML}' \
    --imgsz ${IMAGE_SIZE} \
    --batch ${batch} \
    --epochs ${EPOCHS} \
    --device '${DEVICE}' \
    --workers 4 \
    --pretrained '${YOLO_PRETRAINED}' \
    --lr0 ${LR0} \
    --warmup-epochs ${WARMUP_EPOCHS} \
    --cos-lr '${COS_LR}' \
    --plots=False \
    --project '${OUT}' \
    --name 'train' \
    --rgb-mean '${RGB_MEAN}' \
    --rgb-std '${RGB_STD}'"

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
    runner_log "${MODE}" "${RUN_LOG}" "[yolov8-seg-0831-1k-20ep-scratch] OOM detected, retry with batch=4 (same epochs)"
    cat > "${OUT}/notes_oom.txt" <<EON
OOM fallback activated for ${MODEL_ID}.
Original: batch=${BATCH} epochs=${EPOCHS}
Fallback: batch=4 epochs=${EPOCHS}
EON
    if [[ -d "${OUT}/train" ]]; then
      ts="$(date +%Y%m%d_%H%M%S)"
      mv "${OUT}/train" "${OUT}/train_oom_${ts}"
    fi
    if ! run_train_cmd "4"; then
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
  runner_exec "${MODE}" "${RUN_LOG}" "${HF_ENV_PREFIX}conda run -n magformer python '${REPO_ROOT}/scripts/analysis/write_params_from_yolo_weights.py' --out-dir '${OUT}'"

  ANN_VAL="${DATASET_ROOT}/annotations/instances_val.json"
  runner_exec "${MODE}" "${RUN_LOG}" "${HF_ENV_PREFIX}conda run -n magformer python '${REPO_ROOT}/baselines/yolo_export_coco.py' --dataset-root '${DATASET_ROOT}' --ann-file '${ANN_VAL}' --split val --output-json '${OUT}/coco_instances_results.json'"
  runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && ${HF_ENV_PREFIX}conda run -n magformer python scripts/experiments/postprocess_cocoeval.py --dataset-root '${DATASET_ROOT}' --out-dir '${OUT}' --metrics-out 'metrics.cocoeval.json'"
  runner_exec "${MODE}" "${RUN_LOG}" "${HF_ENV_PREFIX}conda run -n magformer python '${REPO_ROOT}/scripts/analysis/write_metrics_std.py' --out-dir '${OUT}' --iters-per-epoch ${ITERS_PER_EPOCH}"
  runner_exec "${MODE}" "${RUN_LOG}" "${HF_ENV_PREFIX}conda run -n magformer python '${REPO_ROOT}/scripts/analysis/prune_checkpoints.py' --out-dir '${OUT}' --framework yolo"
  runner_exec "${MODE}" "${RUN_LOG}" "${HF_ENV_PREFIX}conda run -n magformer python '${REPO_ROOT}/scripts/analysis/write_run_metadata.py' --phase end --out-dir '${OUT}'"
fi

runner_log "${MODE}" "${RUN_LOG}" "[yolov8-seg-0831-1k-20ep-scratch] done"
