#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"

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

MASK2FORMER_ROOT="${REPO_ROOT}/baselines/Mask2Former"
CFG_REL="configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml"
OUT="${OUTPUT_ROOT}/official_mask2former_scratch"

mkdir -p "${OUT}"

run_cmd() {
  echo "+ $*"
  if [[ "${MODE}" == "run" ]]; then
    eval "$@"
  fi
}

echo "[official-mask2former-0831-1k-5k-scratch] mode=${MODE}"
echo "[official-mask2former-0831-1k-5k-scratch] dataset_root=${DATASET_ROOT}"
echo "[official-mask2former-0831-1k-5k-scratch] output_dir=${OUT}"
echo "[official-mask2former-0831-1k-5k-scratch] config=${MASK2FORMER_ROOT}/${CFG_REL}"

SECONDS=0
run_cmd "cd '${REPO_ROOT}' && conda run -n magformer python baselines/run_official_mask2former_0831_1k.py \
  --dataset-root '${DATASET_ROOT}' \
  --mask2former-root '${MASK2FORMER_ROOT}' \
  -- \
  --num-gpus 1 \
  --config-file '${CFG_REL}' \
  OUTPUT_DIR '${OUT}' \
  DATASETS.TRAIN \"('${DATASET_NAME_TRAIN:-ecc0831_1k_train}',)\" \
  DATASETS.TEST \"('${DATASET_NAME_VAL:-ecc0831_1k_val}',)\" \
  SOLVER.IMS_PER_BATCH 8 \
  SOLVER.BASE_LR 0.0001 \
  SOLVER.WARMUP_ITERS 200 \
  SOLVER.MAX_ITER 5000 \
  SOLVER.STEPS '(4000,4500)' \
  TEST.EVAL_PERIOD 500 \
  SOLVER.CHECKPOINT_PERIOD 1000 \
  MODEL.WEIGHTS '' \
  MODEL.PIXEL_MEAN '[28.1363,30.5413,34.9731]' \
  MODEL.PIXEL_STD '[57.2803,60.9879,64.8187]' \
  MODEL.SEM_SEG_HEAD.NUM_CLASSES 1 \
  MODEL.MASK_FORMER.NUM_OBJECT_QUERIES 100"
echo "${SECONDS}" > "${OUT}/wall_time_sec.txt"

if [[ "${MODE}" == "run" ]]; then
  conda run -n magformer python scripts/analysis/write_params_from_detectron2_ckpt.py --out-dir "${OUT}"
fi

echo "[official-mask2former-0831-1k-5k-scratch] done"
