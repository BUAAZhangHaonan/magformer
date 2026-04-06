#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
source "${SCRIPT_DIR}/common_runner.sh"
source "${SCRIPT_DIR}/ecc_common.sh"
HF_ENV_PREFIX="$(runner_hf_env_prefix)"

REGISTER="0831"
DATASET_ROOT=""
OUTPUT_ROOT="${REPO_ROOT}/output/experiments/0831_1k_20ep_1024_depth_revisit"
MODE="run"
VARIANT="depthnorm_on" # depthnorm_on | nodpth_ref | nodpth_ref_fair
NUM_WORKERS=4
DDP=0
NUM_GPUS=2
IMAGE_SIZE=1024

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
    --variant)
      VARIANT="$2"
      shift 2
      ;;
    --ddp)
      DDP=1
      shift
      ;;
    --num-gpus)
      NUM_GPUS="$2"
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
if [[ -z "${DATASET_ROOT}" ]]; then
  DATASET_ROOT="$(ecc_default_dataset_root "${REGISTER_RAW}")"
fi

case "${VARIANT}" in
  depthnorm_on)
    MODEL_ID="magformer_depthnorm_on"
    CFG_BASE="${REPO_ROOT}/configs/magformer_0831_1k_20ep_1024_depthnorm_on.yaml"
    ;;
  nodpth_ref)
    MODEL_ID="magformer_nodpth_ref"
    CFG_BASE="${REPO_ROOT}/configs/magformer_0831_1k_20ep_1024_nodpth_ref.yaml"
    ;;
  nodpth_ref_fair)
    MODEL_ID="magformer_nodpth_ref_fair"
    CFG_BASE="${REPO_ROOT}/configs/magformer_0831_1k_20ep_1024_nodpth_ref_fair.yaml"
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
REGISTER="$(ecc_normalize_register "${REGISTER_RAW}")"
TRACK_NAME="$(basename "${OUTPUT_ROOT}")"
RUN_LOG="$(runner_setup_log "${OUT}" "${MODE}")"

runner_log "${MODE}" "${RUN_LOG}" "[0831-1k-20ep-1024-revisit-magformer] mode=${MODE}"
runner_log "${MODE}" "${RUN_LOG}" "[0831-1k-20ep-1024-revisit-magformer] variant=${VARIANT}"
runner_log "${MODE}" "${RUN_LOG}" "[0831-1k-20ep-1024-revisit-magformer] dataset_root=${DATASET_ROOT}"
runner_log "${MODE}" "${RUN_LOG}" "[0831-1k-20ep-1024-revisit-magformer] output_dir=${OUT}"
runner_log "${MODE}" "${RUN_LOG}" "[0831-1k-20ep-1024-revisit-magformer] image_size=${IMAGE_SIZE}"

read -r PIXEL_MEAN PIXEL_STD < <(ecc_read_rgb_stats_rgb "${REGISTER_RAW}" "${DATASET_ROOT}")
read -r DEPTH_CLIP_MIN DEPTH_CLIP_MAX < <(ecc_read_depth_clip "${REGISTER_RAW}" "${DATASET_ROOT}")
OVERRIDE_ARGS=(
  --override "model.magformer.pixel_mean=${PIXEL_MEAN}"
  --override "model.magformer.pixel_std=${PIXEL_STD}"
  --override "data.depth.clip_min=${DEPTH_CLIP_MIN}"
  --override "data.depth.clip_max=${DEPTH_CLIP_MAX}"
  --override "data.image_size=${IMAGE_SIZE}"
)
CFG_FINETUNE_WEIGHTS="$(ecc_read_magformer_finetune_weights "${CFG_BASE}")"
FORCE_FALLBACK_WARMSTART=0
if [[ "${VARIANT}" == "nodpth_ref" && "${IMAGE_SIZE}" != "1024" ]]; then
  FORCE_FALLBACK_WARMSTART=1
fi
if [[ -n "${CFG_FINETUNE_WEIGHTS}" ]]; then
  CFG_FINETUNE_RESOLVED="${CFG_FINETUNE_WEIGHTS}"
  if [[ "${CFG_FINETUNE_RESOLVED}" != /* ]]; then
    CFG_FINETUNE_RESOLVED="${REPO_ROOT}/${CFG_FINETUNE_RESOLVED}"
  fi
  if [[ "${FORCE_FALLBACK_WARMSTART}" == "1" || ! -f "${CFG_FINETUNE_RESOLVED}" ]]; then
    FALLBACK_WARMSTART_PTH="$(ecc_magformer_fallback_warmstart_path "${REGISTER_RAW}" "${DATASET_ROOT}" "${CFG_BASE}")"
    ecc_prepare_magformer_fallback_warmstart "${MODE}" "${RUN_LOG}" "${CFG_BASE}" "${REGISTER_RAW}" "${DATASET_ROOT}" "${FALLBACK_WARMSTART_PTH}"
    OVERRIDE_ARGS+=(--override "model.finetune_weights=${FALLBACK_WARMSTART_PTH}")
  fi
fi
if [[ "${DDP}" == "1" ]]; then
  gpu_list="$(python3 - <<PY
n=int("${NUM_GPUS}")
print("[" + ",".join(str(i) for i in range(n)) + "]")
PY
)"
  OVERRIDE_ARGS+=(
    --override "runtime.ddp_enabled=true"
    --override "runtime.gpus=${gpu_list}"
    --override "runtime.find_unused_parameters=true"
  )
fi

IMS_PER_BATCH=4
EPOCHS=20

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


METADATA_ARGS=(
  bash
  "$(basename "${BASH_SOURCE[0]}")"
  --register
  "${REGISTER_RAW}"
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
if [[ "${DDP}" == "1" ]]; then
  METADATA_ARGS+=(--ddp --num-gpus "${NUM_GPUS}")
fi
METADATA_ARGS+=(--image-size "${IMAGE_SIZE}")
if [[ -n "${FALLBACK_WARMSTART_PTH:-}" ]]; then
  METADATA_ARGS+=(--override "model.finetune_weights=${FALLBACK_WARMSTART_PTH}")
fi
METADATA_CMD="$(printf "%q " "${METADATA_ARGS[@]}")"
METADATA_CMD="${METADATA_CMD% }"

runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && ${HF_ENV_PREFIX}conda run -n magformer python scripts/analysis/write_run_metadata.py \
  --phase start \
  --out-dir '${OUT}' \
  --track '${TRACK_NAME}' \
  --register '${REGISTER_RAW}' \
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
RUN_NAME="${REGISTER}_20ep_${IMAGE_SIZE}_${VARIANT}"
runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && ${HF_ENV_PREFIX}conda run -n magformer python scripts/analysis/render_magformer_runtime_config.py \
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
  --num-workers ${NUM_WORKERS} \
  $(printf "%q " "${OVERRIDE_ARGS[@]}")"

SECONDS=0
if [[ "${DDP}" == "1" ]]; then
  runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && ${HF_ENV_PREFIX}conda run -n magformer torchrun --nproc_per_node=${NUM_GPUS} tools/train.py \
    --config '${RUNTIME_CFG}' \
    --dataset-root '${DATASET_ROOT}' \
    --output-dir '${OUT}' \
    --gpus $(seq -s, 0 $((NUM_GPUS-1))) \
    --num-workers ${NUM_WORKERS}"
else
  runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && ${HF_ENV_PREFIX}conda run -n magformer python tools/train.py \
    --config '${RUNTIME_CFG}' \
    --dataset-root '${DATASET_ROOT}' \
    --output-dir '${OUT}' \
    --num-workers ${NUM_WORKERS}"
fi

echo "${SECONDS}" > "${OUT}/wall_time_sec.txt"

runner_exec "${MODE}" "${RUN_LOG}" "${HF_ENV_PREFIX}conda run -n magformer python '${REPO_ROOT}/scripts/analysis/write_params_from_magformer_ckpt.py' --out-dir '${OUT}'"
runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && ${HF_ENV_PREFIX}conda run -n magformer python tools/evaluate.py \
  --config-file '${RUNTIME_CFG}' \
  --dataset-root '${DATASET_ROOT}' \
  --weights \"\$(${HF_ENV_PREFIX}python3 '${REPO_ROOT}/scripts/analysis/find_magformer_checkpoint.py' --out-dir '${OUT}')\" \
  --output '${OUT}' \
  --num-workers ${NUM_WORKERS}"
runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && ${HF_ENV_PREFIX}conda run -n magformer python scripts/experiments/postprocess_cocoeval.py --dataset-root '${DATASET_ROOT}' --out-dir '${OUT}' --metrics-out 'metrics.cocoeval.json'"
runner_exec "${MODE}" "${RUN_LOG}" "${HF_ENV_PREFIX}conda run -n magformer python '${REPO_ROOT}/scripts/analysis/write_metrics_std.py' --out-dir '${OUT}' --iters-per-epoch ${ITERS_PER_EPOCH}"
runner_exec "${MODE}" "${RUN_LOG}" "${HF_ENV_PREFIX}conda run -n magformer python '${REPO_ROOT}/scripts/analysis/prune_checkpoints.py' --out-dir '${OUT}' --framework magformer"
runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && ${HF_ENV_PREFIX}conda run -n magformer python scripts/analysis/write_run_metadata.py --phase end --out-dir '${OUT}'"

runner_log "${MODE}" "${RUN_LOG}" "[0831-1k-20ep-1024-revisit-magformer] done"
