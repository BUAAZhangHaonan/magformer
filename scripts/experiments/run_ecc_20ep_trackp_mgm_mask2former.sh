#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
source "${SCRIPT_DIR}/common_runner.sh"
source "${SCRIPT_DIR}/ecc_common.sh"

REGISTER="0831"
DATASET_ROOT=""
OUTPUT_ROOT=""
MODE="run"
SMOKE=0
CANDIDATE_ID="C1"
RUN_TAG="final" # final | sweep

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
    --run)
      MODE="run"
      shift
      ;;
    --dry-run)
      MODE="dry-run"
      shift
      ;;
    --smoke)
      SMOKE=1
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

REGISTER="$(ecc_normalize_register "${REGISTER}")"
if [[ -z "${DATASET_ROOT}" ]]; then
  DATASET_ROOT="$(ecc_default_dataset_root "${REGISTER}")"
fi
if [[ -z "${OUTPUT_ROOT}" ]]; then
  echo "--output-root is required" >&2
  exit 1
fi

MASK2FORMER_DIR="${PROJECT_ROOT}/mask2former/MGM_Mask2Former"
CFG="${MASK2FORMER_DIR}/configs/mgm_swin_convnext_tiny.yaml"
MODEL_ID="mgm_mask2former"

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

runner_log "${MODE}" "${RUN_LOG}" "[mgm-mask2former-ecc-20ep-trackp] mode=${MODE}"
runner_log "${MODE}" "${RUN_LOG}" "[mgm-mask2former-ecc-20ep-trackp] smoke=${SMOKE}"
runner_log "${MODE}" "${RUN_LOG}" "[mgm-mask2former-ecc-20ep-trackp] register=${REGISTER}"
runner_log "${MODE}" "${RUN_LOG}" "[mgm-mask2former-ecc-20ep-trackp] run_tag=${RUN_TAG} candidate=${CANDIDATE_ID}"
runner_log "${MODE}" "${RUN_LOG}" "[mgm-mask2former-ecc-20ep-trackp] dataset_root=${DATASET_ROOT}"
runner_log "${MODE}" "${RUN_LOG}" "[mgm-mask2former-ecc-20ep-trackp] output_dir=${OUT}"

read -r DEPTH_CLIP_MIN DEPTH_CLIP_MAX < <(ecc_read_depth_clip "${REGISTER}")
runner_log "${MODE}" "${RUN_LOG}" "[mgm-mask2former-ecc-20ep-trackp] depth_clip=[${DEPTH_CLIP_MIN}, ${DEPTH_CLIP_MAX}]"

# Track-P uses public COCO/ImageNet pretrained weights; keep ImageNet mean/std (RGB, 0..255)
# to match the pretrained backbone normalization.
PIXEL_MEAN="[123.675,116.280,103.530]"
PIXEL_STD="[58.395,57.120,57.375]"

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

EPOCHS=20
IMS_PER_BATCH=8
NUM_WORKERS=4
if [[ "${RUN_TAG}" == "sweep" ]]; then
  EPOCHS=5
fi
if [[ "${SMOKE}" == "1" ]]; then
  IMS_PER_BATCH=2
  NUM_WORKERS=2
fi

compute_budget() {
  local ims_per_batch="$1"
  local epochs="$2"
  read -r iters_per_epoch max_iter step1 step2 warmup eval_period ckpt_period < <(ecc_detectron2_budget "${DATASET_ROOT}" "${ims_per_batch}" "${epochs}")
  echo "${iters_per_epoch} ${max_iter} (${step1},${step2}) ${warmup} ${eval_period} ${ckpt_period}"
}

ITERS_PER_EPOCH=""
MAX_ITER=""
SOLVER_STEPS=""
WARMUP_ITERS=""
EVAL_PERIOD=""
CHECKPOINT_PERIOD=""
read -r ITERS_PER_EPOCH MAX_ITER SOLVER_STEPS WARMUP_ITERS EVAL_PERIOD CHECKPOINT_PERIOD < <(compute_budget "${IMS_PER_BATCH}" "${EPOCHS}")

if [[ -n "${WARMUP_OVERRIDE}" ]]; then
  WARMUP_ITERS="${WARMUP_OVERRIDE}"
fi

if [[ "${SMOKE}" == "1" ]]; then
  MAX_ITER=20
  SOLVER_STEPS="(15,18)"
  WARMUP_ITERS=10
  EVAL_PERIOD=10
  CHECKPOINT_PERIOD=10
fi

METADATA_CMD="bash $(basename "${BASH_SOURCE[0]}") --register ${REGISTER} --dataset-root ${DATASET_ROOT} --output-root ${OUTPUT_ROOT} --candidate-id ${CANDIDATE_ID} --run-tag ${RUN_TAG} --mode ${MODE} --smoke ${SMOKE}"
runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && conda run -n magformer python scripts/analysis/write_run_metadata.py \
  --phase start \
  --out-dir '${OUT}' \
  --track trackp \
  --register '${REGISTER}' \
  --dataset-root '${DATASET_ROOT}' \
  --model-id '${MODEL_ID}' \
  --candidate-id '${CANDIDATE_ID}' \
  --run-tag '${RUN_TAG}' \
  --command \"${METADATA_CMD}\" \
  --iters-per-epoch ${ITERS_PER_EPOCH} \
  --max-iter ${MAX_ITER} \
  --epochs ${EPOCHS} \
  --ims-per-batch ${IMS_PER_BATCH}"

# Public COCO pretrained weights (Mask2Former Swin-T instance) for warm-start.
# Note: MGM uses Swin-T + ConvNeXt-T backbones; Swin-T Mask2Former weights provide a strong RGB init,
# while depth/fusion params remain randomly initialized (missing keys are expected).
WEIGHTS_URL="https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/maskformer2_swin_tiny_bs16_50ep/model_final_86143f.pkl"
WEIGHTS_1CLASS_MGM="${REPO_ROOT}/output/pretrained/$(basename "${WEIGHTS_URL}" .pkl)_1class_v2_mgm_rgb.pth"
if [[ ! -f "${WEIGHTS_1CLASS_MGM}" ]]; then
  runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && conda run -n magformer python scripts/analysis/convert_mask2former_ckpt_1class_detectron2.py --input '${WEIGHTS_URL}' --output '${WEIGHTS_1CLASS_MGM}' --rename-prefix backbone=rgb_backbone"
fi
WEIGHTS="${WEIGHTS_1CLASS_MGM}"

run_train_cmd() {
  local max_iter="$1"
  local solver_steps="$2"
  local warmup_iters="$3"
  local ims_per_batch="$4"
  local checkpoint_period="$5"
  local eval_period="$6"
  local cmd="cd '${MASK2FORMER_DIR}' && conda run -n magformer python train_net_mgm_0831.py --num-gpus 1 --config-file '${CFG}' \
    INPUT.DATASET_ROOT '${DATASET_ROOT}' \
    OUTPUT_DIR '${OUT}' \
    MODEL.FINETUNE_WEIGHTS '' \
    MODEL.WEIGHTS '${WEIGHTS}' \
    MODEL.PIXEL_MEAN '${PIXEL_MEAN}' \
    MODEL.PIXEL_STD '${PIXEL_STD}' \
    INPUT.DEPTH_SCALE 1.0 \
    INPUT.DEPTH_SHIFT 0.0 \
    INPUT.DEPTH_CLIP_MIN ${DEPTH_CLIP_MIN} \
    INPUT.DEPTH_CLIP_MAX ${DEPTH_CLIP_MAX} \
    INPUT.DEPTH_NOISE.ENABLED False \
    INPUT.DEPTH_NOISE.GAUSSIAN_STD 0.0 \
    INPUT.DEPTH_NOISE.SPECKLE_STD 0.0 \
    INPUT.DEPTH_NOISE.DROP_PROB 0.0 \
    INPUT.DEPTH_NOISE.DROP_VAL 0.0 \
    SOLVER.MAX_ITER ${max_iter} \
    SOLVER.STEPS '${solver_steps}' \
    SOLVER.BASE_LR ${BASE_LR} \
    SOLVER.WARMUP_ITERS ${warmup_iters} \
    SOLVER.IMS_PER_BATCH ${ims_per_batch} \
    SOLVER.CHECKPOINT_PERIOD ${checkpoint_period} \
    TEST.EVAL_PERIOD ${eval_period} \
    DATALOADER.NUM_WORKERS ${NUM_WORKERS} \
    MODEL.MASK_FORMER.NUM_OBJECT_QUERIES 100 \
    MODEL.MGM.PRIOR.COMPUTE_ON 'full'"
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
  if [[ "${MODE}" != "run" || "${SMOKE}" == "1" ]]; then
    runner_log "${MODE}" "${RUN_LOG}" "FAILED rc=1"
    exit 1
  fi
  if rg -qi "outofmemoryerror|cuda out of memory" "${RUN_LOG}"; then
    runner_log "${MODE}" "${RUN_LOG}" "[mgm-mask2former-ecc-20ep-trackp] OOM detected, retry with batch=4 and epoch-aligned budget"
    read -r _ipe fallback_iter fallback_steps fallback_warmup fallback_eval fallback_ckpt < <(compute_budget "4" "${EPOCHS}")
    cat > "${OUT}/notes_oom.txt" <<EON
OOM fallback activated for ${MODEL_ID}.
Original: batch=${IMS_PER_BATCH} max_iter=${MAX_ITER} steps=${SOLVER_STEPS} warmup_iters=${WARMUP_ITERS}
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
  runner_exec "${MODE}" "${RUN_LOG}" "conda run -n magformer python '${REPO_ROOT}/scripts/analysis/write_params_from_detectron2_ckpt.py' --out-dir '${OUT}'"
  runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && conda run -n magformer python scripts/experiments/postprocess_cocoeval.py --dataset-root '${DATASET_ROOT}' --out-dir '${OUT}' --metrics-out 'metrics.cocoeval.json'"
  runner_exec "${MODE}" "${RUN_LOG}" "conda run -n magformer python '${REPO_ROOT}/scripts/analysis/write_metrics_std.py' --out-dir '${OUT}' --iters-per-epoch ${ITERS_PER_EPOCH}"
  runner_exec "${MODE}" "${RUN_LOG}" "conda run -n magformer python '${REPO_ROOT}/scripts/analysis/prune_checkpoints.py' --out-dir '${OUT}' --framework detectron2"
  runner_exec "${MODE}" "${RUN_LOG}" "conda run -n magformer python '${REPO_ROOT}/scripts/analysis/write_run_metadata.py' --phase end --out-dir '${OUT}'"
fi

runner_log "${MODE}" "${RUN_LOG}" "[mgm-mask2former-ecc-20ep-trackp] done"
