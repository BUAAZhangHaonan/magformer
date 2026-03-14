#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
source "${SCRIPT_DIR}/common_runner.sh"
source "${SCRIPT_DIR}/ecc_common.sh"

DATASET_ROOT_DEFAULT="${PROJECT_ROOT}/magformer_datasets/0831_1K"
OUTPUT_ROOT_DEFAULT="${REPO_ROOT}/output/experiments/0831_1k_20ep_scratch"

DATASET_ROOT="${DATASET_ROOT_DEFAULT}"
OUTPUT_ROOT="${OUTPUT_ROOT_DEFAULT}"
MODE="run"
CANDIDATE_ID="C1"
RUN_TAG="final"  # final | sweep

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

    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

CFG_BASE="${REPO_ROOT}/configs/magformer_0831_1k_20ep_scratch.yaml"
MODEL_ID="magformer_scratch"

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

runner_log "${MODE}" "${RUN_LOG}" "[magformer-0831-1k-20ep-scratch] mode=${MODE}"
runner_log "${MODE}" "${RUN_LOG}" "[magformer-0831-1k-20ep-scratch] run_tag=${RUN_TAG} candidate=${CANDIDATE_ID}"
runner_log "${MODE}" "${RUN_LOG}" "[magformer-0831-1k-20ep-scratch] dataset_root=${DATASET_ROOT}"
runner_log "${MODE}" "${RUN_LOG}" "[magformer-0831-1k-20ep-scratch] output_dir=${OUT}"

# Candidate hyper-parameters
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

# Budget
EPOCHS=20
IMS_PER_BATCH=8
NUM_WORKERS=4
if [[ "${RUN_TAG}" == "sweep" ]]; then
  EPOCHS=5
  MAX_ITER=555
  STEPS="444,500"
  WARMUP_ITERS=55
  EVAL_PERIOD=111
  CHECKPOINT_PERIOD=111
else
  MAX_ITER=2220
  STEPS="1776,1998"
  WARMUP_ITERS=111
  EVAL_PERIOD=111
  CHECKPOINT_PERIOD=111
fi

if [[ -n "${WARMUP_OVERRIDE}" ]]; then
  WARMUP_ITERS="${WARMUP_OVERRIDE}"
fi


NUM_IMAGES="$(ecc_num_train_images "${DATASET_ROOT}")"
ITERS_PER_EPOCH="$(ecc_iters_per_epoch "${NUM_IMAGES}" "${IMS_PER_BATCH}")"

METADATA_ARGS=(
  bash
  "$(basename "${BASH_SOURCE[0]}")"
  --dataset-root
  "${DATASET_ROOT}"
  --output-root
  "${OUTPUT_ROOT}"
  --candidate-id
  "${CANDIDATE_ID}"
  --run-tag
  "${RUN_TAG}"
)
if [[ "${MODE}" == "run" ]]; then
  METADATA_ARGS+=(--run)
else
  METADATA_ARGS+=(--dry-run)
fi
METADATA_CMD="$(printf "%q " "${METADATA_ARGS[@]}")"
METADATA_CMD="${METADATA_CMD% }"
runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && conda run -n magformer python scripts/analysis/write_run_metadata.py \
  --phase start \
  --out-dir '${OUT}' \
  --track tracks \
  --register '0831' \
  --dataset-root '${DATASET_ROOT}' \
  --model-id '${MODEL_ID}' \
  --candidate-id '${CANDIDATE_ID}' \
  --run-tag '${RUN_TAG}' \
  --command \"${METADATA_CMD}\" \
  --iters-per-epoch ${ITERS_PER_EPOCH} \
  --max-iter ${MAX_ITER} \
  --epochs ${EPOCHS} \
  --ims-per-batch ${IMS_PER_BATCH}"

RUNTIME_CFG="${OUT}/magformer_runtime_config.yaml"
RUN_NAME="0831_1k_20ep_${RUN_TAG}_${MODEL_ID}_${CANDIDATE_ID}"
CFG_SRC="${CFG_BASE}"

render_cfg() {
  local ims_per_batch="$1"
  local max_iter="$2"
  local steps="$3"
  local warmup_iters="$4"
  local eval_period="$5"
  local checkpoint_period="$6"
  local base_lr="$7"
  runner_exec "${MODE}" "${RUN_LOG}" "conda run -n magformer python '${REPO_ROOT}/scripts/analysis/render_magformer_runtime_config.py' \
    --base-config '${CFG_SRC}' \
    --out-config '${RUNTIME_CFG}' \
    --output-dir '${OUT}' \
    --run-name '${RUN_NAME}' \
    --base-lr ${base_lr} \
    --max-iter ${max_iter} \
    --steps '${steps}' \
    --warmup-iters ${warmup_iters} \
    --ims-per-batch ${ims_per_batch} \
    --eval-period ${eval_period} \
    --checkpoint-period ${checkpoint_period} \
    --num-workers ${NUM_WORKERS}"
}

run_train_once() {
  local cmd="cd '${REPO_ROOT}' && conda run -n magformer python tools/train.py --config '${RUNTIME_CFG}' --dataset-root '${DATASET_ROOT}' --output-dir '${OUT}' --num-workers ${NUM_WORKERS}"
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
render_cfg "${IMS_PER_BATCH}" "${MAX_ITER}" "${STEPS}" "${WARMUP_ITERS}" "${EVAL_PERIOD}" "${CHECKPOINT_PERIOD}" "${BASE_LR}"
if run_train_once; then
  :
else
  if [[ "${MODE}" != "run" ]]; then
    runner_log "${MODE}" "${RUN_LOG}" "FAILED rc=1"
    exit 1
  fi

  if rg -qi "outofmemoryerror|cuda out of memory" "${RUN_LOG}"; then
    runner_log "${MODE}" "${RUN_LOG}" "[magformer-0831-1k-20ep-scratch] OOM detected, retry with batch=4 and epoch-aligned budget"
    local_fallback_iter=4440
    local_fallback_steps="3552,3996"
    local_fallback_warmup=222
    if [[ "${RUN_TAG}" == "sweep" ]]; then
      local_fallback_iter=1110
      local_fallback_steps="888,1000"
      local_fallback_warmup=110
    fi

    cat > "${OUT}/notes_oom.txt" <<EON
OOM fallback activated for ${MODEL_ID}.
Original: batch=8 max_iter=${MAX_ITER} steps=${STEPS} warmup_iters=${WARMUP_ITERS}
Fallback: batch=4 max_iter=${local_fallback_iter} steps=${local_fallback_steps} warmup_iters=${local_fallback_warmup}
EON

    rm -f "${OUT}"/checkpoint_iter_*.pth "${OUT}/model_best.pth" "${OUT}/coco_instances_results.json" "${OUT}/metrics.cocoeval.json" || true

    render_cfg "4" "${local_fallback_iter}" "${local_fallback_steps}" "${local_fallback_warmup}" "${EVAL_PERIOD}" "${CHECKPOINT_PERIOD}" "${BASE_LR}"
    if ! run_train_once; then
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
  runner_exec "${MODE}" "${RUN_LOG}" "conda run -n magformer python '${REPO_ROOT}/scripts/analysis/write_params_from_magformer_ckpt.py' --out-dir '${OUT}'"
  runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && conda run -n magformer python scripts/experiments/postprocess_cocoeval.py --dataset-root '${DATASET_ROOT}' --out-dir '${OUT}' --metrics-out 'metrics.cocoeval.json'"
  runner_exec "${MODE}" "${RUN_LOG}" "conda run -n magformer python '${REPO_ROOT}/scripts/analysis/write_metrics_std.py' --out-dir '${OUT}' --iters-per-epoch ${ITERS_PER_EPOCH}"
  runner_exec "${MODE}" "${RUN_LOG}" "conda run -n magformer python '${REPO_ROOT}/scripts/analysis/prune_checkpoints.py' --out-dir '${OUT}' --framework magformer"
  runner_exec "${MODE}" "${RUN_LOG}" "conda run -n magformer python '${REPO_ROOT}/scripts/analysis/write_run_metadata.py' --phase end --out-dir '${OUT}'"
fi

runner_log "${MODE}" "${RUN_LOG}" "[magformer-0831-1k-20ep-scratch] done"
