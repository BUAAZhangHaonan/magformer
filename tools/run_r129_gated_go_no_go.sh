#!/usr/bin/env bash
set -u

REPO_ROOT="/home/hdd3/zhanghaonan/magformer"
PYTHON="/home/hdd3/zhanghaonan/anaconda3/envs/magformer/bin/python"
SLEEP_SECONDS="${SLEEP_SECONDS:-300}"
LOG_PATH="${LOG_PATH:-output/diagnostics/r129_gated_go_no_go_20260518.log}"
R122_CHECKPOINT_LABEL="${R122_CHECKPOINT_LABEL:-iter0099}"
R122_CHECKPOINT_PATH="${R122_CHECKPOINT_PATH:-}"
R122_ARTIFACT_DATE="${R122_ARTIFACT_DATE:-20260518}"

GT_JSON="magformer_datasets/pseudo_real_512/annotations/instances_target_unlabeled_r114_balanced_minus125.json"
BASELINE_BUCKET="output/diagnostics/r120_error_atlas_20260518/bucket_compare.csv"

cd "${REPO_ROOT}" || exit 2
r122_path_args=(
  --repo-root "${REPO_ROOT}"
  --checkpoint-label "${R122_CHECKPOINT_LABEL}"
  --artifact-date "${R122_ARTIFACT_DATE}"
  --format shell
)
if [[ -n "${R122_CHECKPOINT_PATH}" ]]; then
  r122_path_args+=(--checkpoint-path "${R122_CHECKPOINT_PATH}")
fi
if ! r122_path_assignments=$("${PYTHON}" tools/r122_formal_eval_paths.py "${r122_path_args[@]}"); then
  printf 'failed to resolve R122 artifact paths for label=%s path=%s\n' "${R122_CHECKPOINT_LABEL}" "${R122_CHECKPOINT_PATH}" >&2
  exit 2
fi
eval "${r122_path_assignments}"

BUCKET_CSV="${BUCKET_COMPARE}"
REMAINING_DIR="${REMAINING75_DIR}"
REMAINING_PRED="${REMAINING_DIR}/coco_instances_results.json"
REMAINING_METRICS="${REMAINING_DIR}/metrics.cocoeval.json"
VAL28_PRED="${VAL28_DIR}/coco_instances_results.json"
VAL28_METRICS="${VAL28_DIR}/metrics.cocoeval.json"
GO_NO_GO_MD="${GO_NO_GO_DIR}/go_no_go.md"
if [[ "${R122_CHECKPOINT_LABEL}" == "iter0099" ]]; then
  R122_CANDIDATE_RUN="r122_remaining75"
elif [[ "${R122_CHECKPOINT_LABEL}" == "final" ]]; then
  R122_CANDIDATE_RUN="r122_final_remaining75"
else
  R122_CANDIDATE_RUN="r122_${R122_CHECKPOINT_LABEL}_remaining75"
fi

mkdir -p "$(dirname "${LOG_PATH}")"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %z')" "$*" | tee -a "${LOG_PATH}"
}

missing_inputs() {
  local missing=0
  for path in "${REMAINING_PRED}" "${REMAINING_METRICS}" "${VAL28_PRED}" "${VAL28_METRICS}" "${GT_JSON}" "${BASELINE_BUCKET}"; do
    if [[ ! -s "${path}" ]]; then
      log "missing input: ${path}"
      missing=1
    fi
  done
  return "${missing}"
}

train_or_eval_running() {
  pgrep -af '(^|/)(train\.py|tools/train\.py|evaluate[^ ]*\.py|eval[^ ]*\.py|tools/evaluate|tools/eval)' | grep -v 'run_r129_gated_go_no_go.sh' >/dev/null 2>&1
}

run_bucket_builder() {
  mkdir -p "$(dirname "${BUCKET_CSV}")"
  log "building candidate bucket_compare: ${BUCKET_CSV}"
  CUDA_VISIBLE_DEVICES="" "${PYTHON}" tools/build_r122_bucket_compare.py \
    --gt-json "${GT_JSON}" \
    --pred-json "${REMAINING_PRED}" \
    --run-name "${R122_CANDIDATE_RUN}" \
    --output-csv "${BUCKET_CSV}" \
    --output-json "${BUCKET_SUMMARY}" >>"${LOG_PATH}" 2>&1
}

run_comparator() {
  mkdir -p "${GO_NO_GO_DIR}"
  log "running go/no-go comparator: ${GO_NO_GO_JSON}"
  CUDA_VISIBLE_DEVICES="" "${PYTHON}" tools/compare_r122_go_no_go.py \
    --baseline-bucket-csv "${BASELINE_BUCKET}" \
    --candidate-bucket-csv "${BUCKET_CSV}" \
    --baseline-run r114_remaining75 \
    --candidate-run "${R122_CANDIDATE_RUN}" \
    --output-json "${GO_NO_GO_JSON}" \
    --output-md "${GO_NO_GO_MD}" >>"${LOG_PATH}" 2>&1
}

log "R129 gated post-eval watcher started; no training/eval will be launched"
log "r122_checkpoint_label=${R122_CHECKPOINT_LABEL} candidate_run=${R122_CANDIDATE_RUN}"

while true; do
  if [[ -s "${GO_NO_GO_JSON}" ]]; then
    log "go/no-go already exists: ${GO_NO_GO_JSON}; exiting"
    exit 0
  fi

  if ! missing_inputs; then
    if train_or_eval_running; then
      log "train/eval process detected; postprocess remains gated"
    fi
    log "inputs incomplete; sleeping ${SLEEP_SECONDS}s"
    sleep "${SLEEP_SECONDS}"
    continue
  fi

  if train_or_eval_running; then
    log "train/eval process detected; sleeping ${SLEEP_SECONDS}s"
    sleep "${SLEEP_SECONDS}"
    continue
  fi

  if [[ -s "${BUCKET_CSV}" ]]; then
    log "candidate bucket_compare exists; not overwriting: ${BUCKET_CSV}"
  else
    if ! run_bucket_builder; then
      log "bucket builder failed; exiting"
      exit 2
    fi
  fi

  run_comparator
  compare_status=$?
  if [[ ${compare_status} -eq 0 ]]; then
    log "go/no-go comparator completed with PASS"
    exit 0
  fi
  if [[ ${compare_status} -eq 1 && -s "${GO_NO_GO_JSON}" ]]; then
    log "go/no-go comparator completed with FAIL decision"
    exit 0
  fi
  log "go/no-go comparator failed with exit ${compare_status}; exiting"
  exit 2
done
