#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
source "${SCRIPT_DIR}/common_runner.sh"

DATASET_ROOT_DEFAULT="${PROJECT_ROOT}/magformer_datasets/0831_1K"
OUTPUT_ROOT_DEFAULT="${REPO_ROOT}/output/experiments/0831_1k_5k_scratch8"

DATASET_ROOT="${DATASET_ROOT_DEFAULT}"
OUTPUT_ROOT="${OUTPUT_ROOT_DEFAULT}"
MODE="run"

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

D2_ROOT="${REPO_ROOT}/baselines/detectron2"
CFG_REL="configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
OUT="${OUTPUT_ROOT}/maskrcnn_scratch"

mkdir -p "${OUT}"
RUN_LOG="$(runner_setup_log "${OUT}" "${MODE}")"

runner_log "${MODE}" "${RUN_LOG}" "[maskrcnn-0831-1k-5k-scratch] mode=${MODE}"
runner_log "${MODE}" "${RUN_LOG}" "[maskrcnn-0831-1k-5k-scratch] dataset_root=${DATASET_ROOT}"
runner_log "${MODE}" "${RUN_LOG}" "[maskrcnn-0831-1k-5k-scratch] output_dir=${OUT}"
runner_log "${MODE}" "${RUN_LOG}" "[maskrcnn-0831-1k-5k-scratch] config=${D2_ROOT}/${CFG_REL}"

SECONDS=0
runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && conda run -n magformer python baselines/run_detectron2_0831_1k.py \
  --dataset-root '${DATASET_ROOT}' \
  --detectron2-root '${D2_ROOT}' \
  -- \
  --num-gpus 1 \
  --config-file '${CFG_REL}' \
  OUTPUT_DIR '${OUT}' \
  DATASETS.TRAIN \"('${DATASET_NAME_TRAIN:-ecc0831_1k_train}',)\" \
  DATASETS.TEST \"('${DATASET_NAME_VAL:-ecc0831_1k_val}',)\" \
  SOLVER.IMS_PER_BATCH 8 \
  SOLVER.BASE_LR 0.01 \
  SOLVER.WARMUP_ITERS 200 \
  SOLVER.MAX_ITER 5000 \
  SOLVER.STEPS '(4000,4500)' \
  TEST.EVAL_PERIOD 500 \
  SOLVER.CHECKPOINT_PERIOD 1000 \
  MODEL.WEIGHTS '' \
  MODEL.PIXEL_MEAN '[28.1363,30.5413,34.9731]' \
  MODEL.PIXEL_STD '[57.2803,60.9879,64.8187]' \
  MODEL.ROI_HEADS.NUM_CLASSES 1"
echo "${SECONDS}" > "${OUT}/wall_time_sec.txt"

if [[ "${MODE}" == "run" ]]; then
  runner_exec "${MODE}" "${RUN_LOG}" "conda run -n magformer python '${REPO_ROOT}/scripts/analysis/write_params_from_detectron2_ckpt.py' --out-dir '${OUT}'"
fi

runner_log "${MODE}" "${RUN_LOG}" "[maskrcnn-0831-1k-5k-scratch] done"
