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

MSMFORMER_ROOT="${REPO_ROOT}/baselines/msmformer/MSMFormer"
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

run_train_cmd() {
  local max_iter="$1"
  local solver_steps="$2"
  local warmup_iters="$3"
  local ims_per_batch="$4"
  local checkpoint_period="$5"
  local eval_period="$6"
  local cmd="cd '${REPO_ROOT}' && conda run -n magformer python baselines/run_msmformer_0831_1k.py \
    --dataset-root '${DATASET_ROOT}' \
    --msmformer-root '${MSMFORMER_ROOT}' \
    -- \
    --num-gpus 1 \
    --config-file '${CFG}' \
    SOLVER.MAX_ITER ${max_iter} \
    SOLVER.STEPS '${solver_steps}' \
    SOLVER.WARMUP_ITERS ${warmup_iters} \
    SOLVER.IMS_PER_BATCH ${ims_per_batch} \
    SOLVER.CHECKPOINT_PERIOD ${checkpoint_period} \
    TEST.EVAL_PERIOD ${eval_period} \
    OUTPUT_DIR '${OUT}'"

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
    exit 1
  fi
  if [[ "${SMOKE}" == "1" ]]; then
    runner_log "${MODE}" "${RUN_LOG}" "FAILED rc=1 (smoke mode, no retry)"
    exit 1
  fi
  if rg -qi "outofmemoryerror|cuda out of memory" "${RUN_LOG}"; then
    runner_log "${MODE}" "${RUN_LOG}" "[msmformer-0831-1k-5k-scratch] OOM detected, retry with batch=4 and iter-aligned budget"
    cat > "${OUT}/notes_oom.txt" <<EOF
OOM fallback activated for msmformer_scratch.
First attempt: batch=8, max_iter=5000.
Fallback attempt: batch=4, max_iter=10000, steps=(8000,9000), warmup_iters=400.
EOF
    rm -f "${OUT}"/model_*.pth "${OUT}"/model_final.pth "${OUT}"/last_checkpoint || true
    if ! run_train_cmd "10000" "(8000,9000)" "400" "4" "1000" "1000"; then
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
  runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && conda run -n magformer python scripts/experiments/postprocess_cocoeval.py --dataset-root '${DATASET_ROOT}' --out-dir '${OUT}' --metrics-out 'metrics.cocoeval.json'"
  runner_exec "${MODE}" "${RUN_LOG}" "conda run -n magformer python '${REPO_ROOT}/scripts/analysis/prune_checkpoints.py' --out-dir '${OUT}' --framework detectron2"
fi

runner_log "${MODE}" "${RUN_LOG}" "[msmformer-0831-1k-5k-scratch] done"
