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

MSMFORMER_ROOT="${REPO_ROOT}/baselines/icra_2026/msmformer/MSMFormer"
CFG="${REPO_ROOT}/configs/baselines/msmformer_0831_1k_5k_scratch.yaml"
OUT="${OUTPUT_ROOT}/msmformer_scratch"

mkdir -p "${OUT}"
RUN_LOG="$(runner_setup_log "${OUT}" "${MODE}")"

runner_log "${MODE}" "${RUN_LOG}" "[msmformer-0831-1k-5k-scratch] mode=${MODE}"
runner_log "${MODE}" "${RUN_LOG}" "[msmformer-0831-1k-5k-scratch] smoke=${SMOKE}"
runner_log "${MODE}" "${RUN_LOG}" "[msmformer-0831-1k-5k-scratch] dataset_root=${DATASET_ROOT}"
runner_log "${MODE}" "${RUN_LOG}" "[msmformer-0831-1k-5k-scratch] output_dir=${OUT}"
runner_log "${MODE}" "${RUN_LOG}" "[msmformer-0831-1k-5k-scratch] config=${CFG}"

MAX_ITER=5000
SOLVER_STEPS="(4000,4500)"
WARMUP_ITERS=200
IMS_PER_BATCH=8
CHECKPOINT_PERIOD=500
EVAL_PERIOD=500
if [[ "${SMOKE}" == "1" ]]; then
  MAX_ITER=20
  SOLVER_STEPS="(15,18)"
  WARMUP_ITERS=10
  IMS_PER_BATCH=2
  CHECKPOINT_PERIOD=10
  EVAL_PERIOD=10
fi

SECONDS=0
runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && conda run -n magformer python baselines/run_msmformer_0831_1k.py \
  --dataset-root '${DATASET_ROOT}' \
  --msmformer-root '${MSMFORMER_ROOT}' \
  -- \
  --num-gpus 1 \
  --config-file '${CFG}' \
  SOLVER.MAX_ITER ${MAX_ITER} \
  SOLVER.STEPS '${SOLVER_STEPS}' \
  SOLVER.WARMUP_ITERS ${WARMUP_ITERS} \
  SOLVER.IMS_PER_BATCH ${IMS_PER_BATCH} \
  SOLVER.CHECKPOINT_PERIOD ${CHECKPOINT_PERIOD} \
  TEST.EVAL_PERIOD ${EVAL_PERIOD} \
  OUTPUT_DIR '${OUT}'"
echo "${SECONDS}" > "${OUT}/wall_time_sec.txt"

if [[ "${MODE}" == "run" ]]; then
  runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && conda run -n magformer python scripts/experiments/postprocess_cocoeval.py --dataset-root '${DATASET_ROOT}' --out-dir '${OUT}' --metrics-out 'metrics.cocoeval.json'"
  runner_exec "${MODE}" "${RUN_LOG}" "conda run -n magformer python '${REPO_ROOT}/scripts/analysis/prune_checkpoints.py' --out-dir '${OUT}' --framework detectron2"
fi

runner_log "${MODE}" "${RUN_LOG}" "[msmformer-0831-1k-5k-scratch] done"
