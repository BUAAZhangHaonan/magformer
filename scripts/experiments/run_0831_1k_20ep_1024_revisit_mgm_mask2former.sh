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
VARIANT="depthnorm_on" # depthnorm_on | nodpth_ref
NUM_WORKERS=4
NUM_GPUS=1
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

MASK2FORMER_DIR="${REPO_ROOT}/baselines/MGM_Mask2Former"
CFG="${MASK2FORMER_DIR}/configs/mgm_swin_convnext_tiny.yaml"
WEIGHTS_URL="https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/maskformer2_swin_tiny_bs16_50ep/model_final_86143f.pkl"
DEPTH_WEIGHTS="${REPO_ROOT}/output/pretrained/convnext_tiny_imagenet_in1_mgm_depth_backbone.pth"
read -r PIXEL_MEAN PIXEL_STD < <(ecc_read_rgb_stats_rgb_for_dataset_root "${DATASET_ROOT}")
read -r DEPTH_CLIP_MIN DEPTH_CLIP_MAX < <(ecc_read_depth_clip_for_dataset_root "${DATASET_ROOT}")

case "${VARIANT}" in
  depthnorm_on)
    MODEL_ID="mgm_mask2former_depthnorm_on"
    MODEL_DEPTH_ENABLED="True"
    MODEL_MGM_ENABLED="True"
    MODEL_DPE_ENABLED="True"
    DEPTH_PER_SAMPLE_NORM="True"
    ;;
  nodpth_ref)
    MODEL_ID="mgm_mask2former_nodpth_ref"
    MODEL_DEPTH_ENABLED="False"
    MODEL_MGM_ENABLED="False"
    MODEL_DPE_ENABLED="False"
    DEPTH_PER_SAMPLE_NORM="False"
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
TRACK_NAME="$(basename "${OUTPUT_ROOT}")"
RUN_LOG="$(runner_setup_log "${OUT}" "${MODE}")"

runner_log "${MODE}" "${RUN_LOG}" "[0831-1k-20ep-1024-revisit-mgm] mode=${MODE}"
runner_log "${MODE}" "${RUN_LOG}" "[0831-1k-20ep-1024-revisit-mgm] variant=${VARIANT}"
runner_log "${MODE}" "${RUN_LOG}" "[0831-1k-20ep-1024-revisit-mgm] dataset_root=${DATASET_ROOT}"
runner_log "${MODE}" "${RUN_LOG}" "[0831-1k-20ep-1024-revisit-mgm] output_dir=${OUT}"
runner_log "${MODE}" "${RUN_LOG}" "[0831-1k-20ep-1024-revisit-mgm] image_size=${IMAGE_SIZE}"

IMS_PER_BATCH=4
EPOCHS=20

NUM_IMAGES="$(ecc_num_train_images "${DATASET_ROOT}")"
ITERS_PER_EPOCH="$(ecc_iters_per_epoch "${NUM_IMAGES}" "${IMS_PER_BATCH}")"
MAX_ITER=$(( ITERS_PER_EPOCH * EPOCHS ))
STEP1=$(( MAX_ITER * 8 / 10 ))
STEP2=$(( MAX_ITER * 9 / 10 ))
SOLVER_STEPS="(${STEP1},${STEP2})"
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
if [[ "${NUM_GPUS}" != "1" ]]; then
  METADATA_ARGS+=(--num-gpus "${NUM_GPUS}")
fi
METADATA_ARGS+=(--image-size "${IMAGE_SIZE}")
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

runner_exec "${MODE}" "${RUN_LOG}" "cd '${MASK2FORMER_DIR}' && ${HF_ENV_PREFIX}conda run -n magformer python train_net_mgm_0831.py --num-gpus ${NUM_GPUS} --config-file '${CFG}' \
  INPUT.DATASET_ROOT '${DATASET_ROOT}' \
  OUTPUT_DIR '${OUT}' \
  DDP.FIND_UNUSED_PARAMETERS $([[ \"${NUM_GPUS}\" != \"1\" ]] && echo True || echo False) \
  MODEL.FINETUNE_WEIGHTS '' \
  MODEL.WEIGHTS '${WEIGHTS_URL}' \
  MODEL.DEPTH_BACKBONE.WEIGHTS '${DEPTH_WEIGHTS}' \
  MODEL.PIXEL_MEAN '${PIXEL_MEAN}' \
  MODEL.PIXEL_STD '${PIXEL_STD}' \
  MODEL.DEPTH_BACKBONE.ENABLED ${MODEL_DEPTH_ENABLED} \
  MODEL.MGM.ENABLED ${MODEL_MGM_ENABLED} \
  MODEL.DPE.ENABLED ${MODEL_DPE_ENABLED} \
  INPUT.IMAGE_SIZE ${IMAGE_SIZE} \
  INPUT.MIN_SCALE 0.1 \
  INPUT.MAX_SCALE 2.0 \
  INPUT.RANDOM_FLIP 'horizontal' \
  INPUT.DEPTH_SCALE 1.0 \
  INPUT.DEPTH_SHIFT 0.0 \
  INPUT.DEPTH_CLIP_MIN ${DEPTH_CLIP_MIN} \
  INPUT.DEPTH_CLIP_MAX ${DEPTH_CLIP_MAX} \
  INPUT.DEPTH_NORM 'minmax' \
  INPUT.DEPTH_PER_SAMPLE_NORM ${DEPTH_PER_SAMPLE_NORM} \
  INPUT.DEPTH_NOISE.ENABLED False \
  INPUT.DEPTH_NOISE.GAUSSIAN_STD 0.0 \
  INPUT.DEPTH_NOISE.SPECKLE_STD 0.0 \
  INPUT.DEPTH_NOISE.DROP_PROB 0.0 \
  INPUT.DEPTH_NOISE.DROP_VAL 0.0 \
  SOLVER.MAX_ITER ${MAX_ITER} \
  SOLVER.STEPS '${SOLVER_STEPS}' \
  SOLVER.BASE_LR ${BASE_LR} \
  SOLVER.WARMUP_ITERS ${WARMUP_ITERS} \
  SOLVER.IMS_PER_BATCH ${IMS_PER_BATCH} \
  SOLVER.CHECKPOINT_PERIOD ${CHECKPOINT_PERIOD} \
  TEST.EVAL_PERIOD ${EVAL_PERIOD} \
  DATALOADER.NUM_WORKERS ${NUM_WORKERS} \
  MODEL.MASK_FORMER.NUM_OBJECT_QUERIES 100 \
  MODEL.MGM.PRIOR.COMPUTE_ON 'full'"

runner_exec "${MODE}" "${RUN_LOG}" "${HF_ENV_PREFIX}conda run -n magformer python '${REPO_ROOT}/scripts/analysis/write_params_from_detectron2_ckpt.py' --out-dir '${OUT}'"
runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && ${HF_ENV_PREFIX}conda run -n magformer python scripts/experiments/postprocess_cocoeval.py --dataset-root '${DATASET_ROOT}' --out-dir '${OUT}' --metrics-out 'metrics.cocoeval.json'"
runner_exec "${MODE}" "${RUN_LOG}" "${HF_ENV_PREFIX}conda run -n magformer python '${REPO_ROOT}/scripts/analysis/write_metrics_std.py' --out-dir '${OUT}' --iters-per-epoch ${ITERS_PER_EPOCH}"
runner_exec "${MODE}" "${RUN_LOG}" "${HF_ENV_PREFIX}conda run -n magformer python '${REPO_ROOT}/scripts/analysis/prune_checkpoints.py' --out-dir '${OUT}' --framework detectron2"
runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && ${HF_ENV_PREFIX}conda run -n magformer python scripts/analysis/write_run_metadata.py --phase end --out-dir '${OUT}'"

runner_log "${MODE}" "${RUN_LOG}" "[0831-1k-20ep-1024-revisit-mgm] done"
