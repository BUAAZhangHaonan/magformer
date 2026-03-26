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
read -r DATASET_NAME_TRAIN DATASET_NAME_VAL < <(ecc_dataset_names_coco "${REGISTER_RAW}" "${DATASET_ROOT}")
read -r PIXEL_MEAN PIXEL_STD < <(ecc_read_rgb_stats_bgr "${REGISTER_RAW}" "${DATASET_ROOT}")

MASK2FORMER_ROOT="${REPO_ROOT}/baselines/Mask2Former"
CFG_REL="configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml"
MODEL_ID="official_mask2former_scratch"
WEIGHTS=""
if [[ "${PRETRAINED}" == "1" ]]; then
  MODEL_ID="official_mask2former_pretrained"
  WEIGHTS_URL="https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/maskformer2_R50_bs16_50ep/model_final_3c8ec9.pkl"
  WEIGHTS_1CLASS="${REPO_ROOT}/output/pretrained/$(basename "${WEIGHTS_URL}" .pkl)_1class_v2_${DATASET_PREFIX}_${IMAGE_SIZE}.pth"
  WEIGHTS="${WEIGHTS_1CLASS}"
fi

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

runner_log "${MODE}" "${RUN_LOG}" "[official-mask2former-0831-1k-20ep-scratch] mode=${MODE}"
runner_log "${MODE}" "${RUN_LOG}" "[official-mask2former-0831-1k-20ep-scratch] run_tag=${RUN_TAG} candidate=${CANDIDATE_ID}"
runner_log "${MODE}" "${RUN_LOG}" "[official-mask2former-0831-1k-20ep-scratch] dataset_root=${DATASET_ROOT}"
runner_log "${MODE}" "${RUN_LOG}" "[official-mask2former-0831-1k-20ep-scratch] output_dir=${OUT}"
runner_log "${MODE}" "${RUN_LOG}" "[official-mask2former-0831-1k-20ep-scratch] image_size=${IMAGE_SIZE}"
runner_log "${MODE}" "${RUN_LOG}" "[official-mask2former-0831-1k-20ep-scratch] pretrained=${PRETRAINED}"

if [[ "${PRETRAINED}" == "1" && "${MODE}" == "run" ]]; then
  runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && ${HF_ENV_PREFIX}conda run -n magformer python scripts/analysis/convert_mask2former_ckpt_1class_detectron2.py --input '${WEIGHTS_URL}' --output '${WEIGHTS_1CLASS}'"
fi

BASE_LR="0.0001"
WARMUP_OVERRIDE=""
case "${CANDIDATE_ID}" in
  C1) BASE_LR="0.0001" ;;
  C2) BASE_LR="0.0002" ;;
  C3) BASE_LR="0.00005" ;;
  C4) BASE_LR="0.0001"; WARMUP_OVERRIDE="0" ;;
  *)
    echo "Unsupported --candidate-id: ${CANDIDATE_ID}" >&2
    exit 1
    ;;
esac

IMS_PER_BATCH=8
EPOCHS=20
MAX_ITER=2220
SOLVER_STEPS="(1776,1998)"
WARMUP_ITERS=111
CHECKPOINT_PERIOD=111
EVAL_PERIOD=111
if [[ "${RUN_TAG}" == "sweep" ]]; then
  EPOCHS=5
  MAX_ITER=555
  SOLVER_STEPS="(444,500)"
  WARMUP_ITERS=55
fi
if [[ -n "${WARMUP_OVERRIDE}" ]]; then
  WARMUP_ITERS="${WARMUP_OVERRIDE}"
fi

NUM_IMAGES="$(ecc_num_train_images "${DATASET_ROOT}")"
ITERS_PER_EPOCH="$(ecc_iters_per_epoch "${NUM_IMAGES}" "${IMS_PER_BATCH}")"

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
if [[ "${PRETRAINED}" == "1" ]]; then
  METADATA_ARGS+=(--pretrained)
fi
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
  --ims-per-batch ${IMS_PER_BATCH}"

run_train_cmd() {
  local max_iter="$1"
  local solver_steps="$2"
  local warmup_iters="$3"
  local ims_per_batch="$4"
  local checkpoint_period="$5"
  local eval_period="$6"

  local cmd="cd '${REPO_ROOT}' && ${HF_ENV_PREFIX}conda run -n magformer python baselines/run_official_mask2former_ecc.py \
    --register '${REGISTER_RAW}' \
    --dataset-root '${DATASET_ROOT}' \
    --mask2former-root '${MASK2FORMER_ROOT}' \
    -- \
    --num-gpus 1 \
    --config-file '${CFG_REL}' \
    OUTPUT_DIR '${OUT}' \
    DATASETS.TRAIN \"('${DATASET_NAME_TRAIN}',)\" \
    DATASETS.TEST \"('${DATASET_NAME_VAL}',)\" \
    SOLVER.IMS_PER_BATCH ${ims_per_batch} \
    SOLVER.BASE_LR ${BASE_LR} \
    SOLVER.WARMUP_ITERS ${warmup_iters} \
    SOLVER.MAX_ITER ${max_iter} \
    SOLVER.STEPS '${solver_steps}' \
    TEST.EVAL_PERIOD ${eval_period} \
    SOLVER.CHECKPOINT_PERIOD ${checkpoint_period} \
    INPUT.MIN_SIZE_TRAIN '(${IMAGE_SIZE},)' \
    INPUT.MAX_SIZE_TRAIN ${IMAGE_SIZE} \
    INPUT.MIN_SIZE_TEST ${IMAGE_SIZE} \
    INPUT.MAX_SIZE_TEST ${IMAGE_SIZE} \
    INPUT.MASK_FORMAT bitmask \
    MODEL.WEIGHTS '${WEIGHTS}' \
    MODEL.PIXEL_MEAN '${PIXEL_MEAN}' \
    MODEL.PIXEL_STD '${PIXEL_STD}' \
    MODEL.SEM_SEG_HEAD.NUM_CLASSES 1 \
    MODEL.MASK_FORMER.NUM_OBJECT_QUERIES 100"

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
if run_train_cmd "${MAX_ITER}" "${SOLVER_STEPS}" "${WARMUP_ITERS}" "${IMS_PER_BATCH}" "${CHECKPOINT_PERIOD}" "${EVAL_PERIOD}"; then
  :
else
  if [[ "${MODE}" != "run" ]]; then
    runner_log "${MODE}" "${RUN_LOG}" "FAILED rc=1"
    exit 1
  fi
  if rg -qi "outofmemoryerror|cuda out of memory" "${RUN_LOG}"; then
    runner_log "${MODE}" "${RUN_LOG}" "[official-mask2former-0831-1k-20ep-scratch] OOM detected, retry with batch=4 and epoch-aligned budget"
    fallback_iter=4440
    fallback_steps="(3552,3996)"
    fallback_warmup=222
    fallback_ckpt=222
    fallback_eval=222
    if [[ "${RUN_TAG}" == "sweep" ]]; then
      fallback_iter=1110
      fallback_steps="(888,1000)"
      fallback_warmup=110
      fallback_ckpt=111
      fallback_eval=111
    fi
    cat > "${OUT}/notes_oom.txt" <<EON
OOM fallback activated for ${MODEL_ID}.
Original: batch=8 max_iter=${MAX_ITER} steps=${SOLVER_STEPS} warmup_iters=${WARMUP_ITERS}
Fallback: batch=4 max_iter=${fallback_iter} steps=${fallback_steps} warmup_iters=${fallback_warmup}
EON
    rm -f "${OUT}"/model_*.pth "${OUT}"/model_final.pth "${OUT}"/last_checkpoint "${OUT}"/metrics.cocoeval.json "${OUT}"/coco_instances_results.json || true
    if ! run_train_cmd "${fallback_iter}" "${fallback_steps}" "${fallback_warmup}" "4" "${fallback_ckpt}" "${fallback_eval}"; then
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
  runner_exec "${MODE}" "${RUN_LOG}" "${HF_ENV_PREFIX}conda run -n magformer python '${REPO_ROOT}/scripts/analysis/write_params_from_detectron2_ckpt.py' --out-dir '${OUT}'"
  runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && ${HF_ENV_PREFIX}conda run -n magformer python scripts/experiments/postprocess_cocoeval.py --dataset-root '${DATASET_ROOT}' --out-dir '${OUT}' --metrics-out 'metrics.cocoeval.json'"
  runner_exec "${MODE}" "${RUN_LOG}" "${HF_ENV_PREFIX}conda run -n magformer python '${REPO_ROOT}/scripts/analysis/write_metrics_std.py' --out-dir '${OUT}' --iters-per-epoch ${ITERS_PER_EPOCH}"
  runner_exec "${MODE}" "${RUN_LOG}" "${HF_ENV_PREFIX}conda run -n magformer python '${REPO_ROOT}/scripts/analysis/prune_checkpoints.py' --out-dir '${OUT}' --framework detectron2"
  runner_exec "${MODE}" "${RUN_LOG}" "${HF_ENV_PREFIX}conda run -n magformer python '${REPO_ROOT}/scripts/analysis/write_run_metadata.py' --phase end --out-dir '${OUT}'"
fi

runner_log "${MODE}" "${RUN_LOG}" "[official-mask2former-0831-1k-20ep-scratch] done"
