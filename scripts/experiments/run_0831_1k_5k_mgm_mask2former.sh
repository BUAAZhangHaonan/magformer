#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"

DATASET_ROOT_DEFAULT="${PROJECT_ROOT}/magformer_datasets/0831_1K"
OUTPUT_ROOT_DEFAULT="${REPO_ROOT}/output/experiments/0831_1k_5k"

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

MASK2FORMER_DIR="${REPO_ROOT}/baselines/MGM_Mask2Former"
CFG="${MASK2FORMER_DIR}/configs/mgm_swin_convnext_tiny.yaml"
OUT="${OUTPUT_ROOT}/mgm_mask2former"

mkdir -p "${OUT}"

run_cmd() {
  echo "+ $*"
  if [[ "${MODE}" == "run" ]]; then
    eval "$@"
  fi
}

echo "[mgm-mask2former-0831-1k-5k] mode=${MODE}"
echo "[mgm-mask2former-0831-1k-5k] dataset_root=${DATASET_ROOT}"
echo "[mgm-mask2former-0831-1k-5k] output_dir=${OUT}"

SECONDS=0
run_cmd "cd '${MASK2FORMER_DIR}' && conda run -n magformer python train_net_mgm_0831.py --num-gpus 1 --config-file '${CFG}' \
  INPUT.DATASET_ROOT '${DATASET_ROOT}' \
  OUTPUT_DIR '${OUT}' \
  MODEL.FINETUNE_WEIGHTS '' \
  MODEL.WEIGHTS '' \
  MODEL.PIXEL_MEAN '[34.9731,30.5413,28.1363]' \
  MODEL.PIXEL_STD '[64.8187,60.9879,57.2803]' \
  SOLVER.MAX_ITER 5000 \
  SOLVER.STEPS '(4000,4500)' \
  SOLVER.BASE_LR 0.0001 \
  SOLVER.WARMUP_ITERS 200 \
  SOLVER.IMS_PER_BATCH 8 \
  TEST.EVAL_PERIOD 500 \
  DATALOADER.NUM_WORKERS 4 \
  MODEL.MGM.PRIOR.COMPUTE_ON 'full'"
echo "${SECONDS}" > "${OUT}/wall_time_sec.txt"

echo "[mgm-mask2former-0831-1k-5k] done"
