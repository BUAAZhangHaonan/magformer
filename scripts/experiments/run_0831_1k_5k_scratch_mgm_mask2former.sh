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
SMOKE=0

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

MASK2FORMER_DIR="${REPO_ROOT}/baselines/MGM_Mask2Former"
CFG="${MASK2FORMER_DIR}/configs/mgm_swin_convnext_tiny.yaml"
OUT="${OUTPUT_ROOT}/mgm_mask2former_scratch"

mkdir -p "${OUT}"
RUN_LOG="$(runner_setup_log "${OUT}" "${MODE}")"

runner_log "${MODE}" "${RUN_LOG}" "[mgm-mask2former-0831-1k-5k-scratch] mode=${MODE}"
runner_log "${MODE}" "${RUN_LOG}" "[mgm-mask2former-0831-1k-5k-scratch] smoke=${SMOKE}"
runner_log "${MODE}" "${RUN_LOG}" "[mgm-mask2former-0831-1k-5k-scratch] dataset_root=${DATASET_ROOT}"
runner_log "${MODE}" "${RUN_LOG}" "[mgm-mask2former-0831-1k-5k-scratch] output_dir=${OUT}"

read -r DEPTH_CLIP_MIN DEPTH_CLIP_MAX < <(conda run -n magformer python -c "from baselines.depth_stats import load_0831_1k_depth_stats; s=load_0831_1k_depth_stats(); print(s.p1, s.p99)")
runner_log "${MODE}" "${RUN_LOG}" "[mgm-mask2former-0831-1k-5k-scratch] depth_clip=[${DEPTH_CLIP_MIN}, ${DEPTH_CLIP_MAX}]"

MAX_ITER=5000
SOLVER_STEPS="(4000,4500)"
WARMUP_ITERS=200
IMS_PER_BATCH=8
CHECKPOINT_PERIOD=500
EVAL_PERIOD=500
NUM_WORKERS=4
if [[ "${SMOKE}" == "1" ]]; then
  MAX_ITER=20
  SOLVER_STEPS="(15,18)"
  WARMUP_ITERS=10
  IMS_PER_BATCH=2
  CHECKPOINT_PERIOD=10
  EVAL_PERIOD=10
  NUM_WORKERS=2
fi

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
  SOLVER.MAX_ITER ${MAX_ITER} \
  SOLVER.STEPS '${SOLVER_STEPS}' \
  SOLVER.BASE_LR 0.0001 \
  SOLVER.WARMUP_ITERS ${WARMUP_ITERS} \
  SOLVER.IMS_PER_BATCH ${IMS_PER_BATCH} \
  SOLVER.CHECKPOINT_PERIOD ${CHECKPOINT_PERIOD} \
  TEST.EVAL_PERIOD ${EVAL_PERIOD} \
  DATALOADER.NUM_WORKERS ${NUM_WORKERS} \
  MODEL.MGM.PRIOR.COMPUTE_ON 'full'"
echo "${SECONDS}" > "${OUT}/wall_time_sec.txt"

if [[ "${MODE}" == "run" ]]; then
  runner_exec "${MODE}" "${RUN_LOG}" "conda run -n magformer python '${REPO_ROOT}/scripts/analysis/write_params_from_detectron2_ckpt.py' --out-dir '${OUT}'"
  runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && conda run -n magformer python scripts/experiments/postprocess_cocoeval.py --dataset-root '${DATASET_ROOT}' --out-dir '${OUT}' --metrics-out 'metrics.cocoeval.json'"
  runner_exec "${MODE}" "${RUN_LOG}" "conda run -n magformer python '${REPO_ROOT}/scripts/analysis/prune_checkpoints.py' --out-dir '${OUT}' --framework detectron2"
fi

runner_log "${MODE}" "${RUN_LOG}" "[mgm-mask2former-0831-1k-5k-scratch] done"
