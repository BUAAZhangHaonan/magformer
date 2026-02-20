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

MASK2FORMER_DIR="${PROJECT_ROOT}/mask2former/MGM_Mask2Former"
CFG="${MASK2FORMER_DIR}/configs/mgm_swin_convnext_tiny.yaml"
OUT="${OUTPUT_ROOT}/mgm_mask2former_scratch"

mkdir -p "${OUT}"
RUN_LOG="$(runner_setup_log "${OUT}" "${MODE}")"

runner_log "${MODE}" "${RUN_LOG}" "[mgm-mask2former-0831-1k-5k-scratch] mode=${MODE}"
runner_log "${MODE}" "${RUN_LOG}" "[mgm-mask2former-0831-1k-5k-scratch] dataset_root=${DATASET_ROOT}"
runner_log "${MODE}" "${RUN_LOG}" "[mgm-mask2former-0831-1k-5k-scratch] output_dir=${OUT}"

read -r DEPTH_CLIP_MIN DEPTH_CLIP_MAX < <(conda run -n magformer python -c "from baselines.depth_stats import load_0831_1k_depth_stats; s=load_0831_1k_depth_stats(); print(s.p1, s.p99)")
runner_log "${MODE}" "${RUN_LOG}" "[mgm-mask2former-0831-1k-5k-scratch] depth_clip=[${DEPTH_CLIP_MIN}, ${DEPTH_CLIP_MAX}]"

SECONDS=0
runner_exec "${MODE}" "${RUN_LOG}" "cd '${MASK2FORMER_DIR}' && conda run -n magformer python train_net_mgm_0831.py --num-gpus 1 --config-file '${CFG}' \
  INPUT.DATASET_ROOT '${DATASET_ROOT}' \
  OUTPUT_DIR '${OUT}' \
  MODEL.FINETUNE_WEIGHTS '' \
  MODEL.WEIGHTS '' \
  MODEL.PIXEL_MEAN '[34.9731,30.5413,28.1363]' \
  MODEL.PIXEL_STD '[64.8187,60.9879,57.2803]' \
  INPUT.DEPTH_SCALE 1.0 \
  INPUT.DEPTH_SHIFT 0.0 \
  INPUT.DEPTH_CLIP_MIN ${DEPTH_CLIP_MIN} \
  INPUT.DEPTH_CLIP_MAX ${DEPTH_CLIP_MAX} \
  INPUT.DEPTH_NOISE.ENABLED False \
  INPUT.DEPTH_NOISE.GAUSSIAN_STD 0.0 \
  INPUT.DEPTH_NOISE.SPECKLE_STD 0.0 \
  INPUT.DEPTH_NOISE.DROP_PROB 0.0 \
  INPUT.DEPTH_NOISE.DROP_VAL 0.0 \
  SOLVER.MAX_ITER 5000 \
  SOLVER.STEPS '(4000,4500)' \
  SOLVER.BASE_LR 0.0001 \
  SOLVER.WARMUP_ITERS 200 \
  SOLVER.IMS_PER_BATCH 8 \
  TEST.EVAL_PERIOD 500 \
  DATALOADER.NUM_WORKERS 4 \
  MODEL.MGM.PRIOR.COMPUTE_ON 'full'"
echo "${SECONDS}" > "${OUT}/wall_time_sec.txt"

if [[ "${MODE}" == "run" ]]; then
  runner_exec "${MODE}" "${RUN_LOG}" "conda run -n magformer python '${REPO_ROOT}/scripts/analysis/write_params_from_detectron2_ckpt.py' --out-dir '${OUT}'"
fi

runner_log "${MODE}" "${RUN_LOG}" "[mgm-mask2former-0831-1k-5k-scratch] done"
