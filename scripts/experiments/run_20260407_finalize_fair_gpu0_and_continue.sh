#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/common_runner.sh"
source "${SCRIPT_DIR}/ecc_common.sh"

MODE="run"
TRAIN_PID=""
REGISTER="20260318_1K_1566"
DATASET_ROOT=""
OUTPUT_BASE="${REPO_ROOT}/output/experiments"
DATE_TAG="20260406"
EXPECTED_LAST_ITER=6319
NUM_WORKERS=4
ITERS_PER_EPOCH=316
WAIT_SLEEP_SEC=60

while [[ $# -gt 0 ]]; do
  case "$1" in
    --train-pid)
      TRAIN_PID="$2"
      shift 2
      ;;
    --register)
      REGISTER="$2"
      shift 2
      ;;
    --dataset-root)
      DATASET_ROOT="$2"
      shift 2
      ;;
    --output-base)
      OUTPUT_BASE="$2"
      shift 2
      ;;
    --date-tag)
      DATE_TAG="$2"
      shift 2
      ;;
    --expected-last-iter)
      EXPECTED_LAST_ITER="$2"
      shift 2
      ;;
    --num-workers)
      NUM_WORKERS="$2"
      shift 2
      ;;
    --iters-per-epoch)
      ITERS_PER_EPOCH="$2"
      shift 2
      ;;
    --wait-sleep-sec)
      WAIT_SLEEP_SEC="$2"
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

if [[ -z "${TRAIN_PID}" ]]; then
  echo "--train-pid is required" >&2
  exit 1
fi

if [[ -z "${DATASET_ROOT}" ]]; then
  DATASET_ROOT="$(ecc_default_dataset_root "${REGISTER}")"
fi

OUT="${OUTPUT_BASE}/${DATE_TAG}_1k_1566_20ep_1024_full19/magformer_nodpth_ref_fair"
RUN_LOG="$(runner_setup_log "${OUTPUT_BASE}/${DATE_TAG}_gpu0_finalize_and_queue" "${MODE}")"

runner_log "${MODE}" "${RUN_LOG}" "[gpu0-finalize] train_pid=${TRAIN_PID}"
runner_log "${MODE}" "${RUN_LOG}" "[gpu0-finalize] dataset_root=${DATASET_ROOT}"
runner_log "${MODE}" "${RUN_LOG}" "[gpu0-finalize] out_dir=${OUT}"
runner_log "${MODE}" "${RUN_LOG}" "[gpu0-finalize] expected_last_iter=${EXPECTED_LAST_ITER}"

if [[ "${MODE}" == "run" ]]; then
  while ps -p "${TRAIN_PID}" >/dev/null 2>&1; do
    runner_log "${MODE}" "${RUN_LOG}" "[gpu0-finalize] waiting for pid ${TRAIN_PID}"
    sleep "${WAIT_SLEEP_SEC}"
  done

  python3 - "${OUT}" "${EXPECTED_LAST_ITER}" "${RUN_LOG}" <<'PY'
import csv
import sys
from pathlib import Path

out = Path(sys.argv[1])
expected_last_iter = int(sys.argv[2])
run_log = Path(sys.argv[3])
rows = list(csv.DictReader((out / "metrics_log.csv").open()))
last_iter = max(int(row["iter"]) for row in rows if row.get("iter"))
prev = None
segment_max = 0.0
total = 0.0
for row in rows:
    raw = row.get("elapsed_sec")
    if not raw:
        continue
    cur = float(raw)
    if prev is not None and cur + 1e-9 < prev:
        total += segment_max
        segment_max = cur
    else:
        segment_max = max(segment_max, cur)
    prev = cur
total += segment_max
(out / "wall_time_sec.txt").write_text(f"{total}\n", encoding="utf-8")
with run_log.open("a", encoding="utf-8") as fh:
    fh.write(f"[gpu0-finalize] last_iter={last_iter}\n")
    fh.write(f"[gpu0-finalize] wall_time_sec={total}\n")
if last_iter < expected_last_iter:
    raise SystemExit(f"fair run stopped early at iter {last_iter}")
PY

  BEST_CKPT="$(python3 "${REPO_ROOT}/scripts/analysis/find_magformer_checkpoint.py" --out-dir "${OUT}")"
else
  runner_log "${MODE}" "${RUN_LOG}" "+ wait for pid ${TRAIN_PID} to exit"
  runner_log "${MODE}" "${RUN_LOG}" "+ python3 compute wall_time and verify last_iter >= ${EXPECTED_LAST_ITER}"
  BEST_CKPT="<best-magformer-checkpoint>"
fi

runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && conda run -n magformer python scripts/analysis/write_params_from_magformer_ckpt.py --out-dir '${OUT}'"
runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && conda run -n magformer python tools/evaluate.py \
  --config-file '${OUT}/magformer_runtime_config.yaml' \
  --dataset-root '${DATASET_ROOT}' \
  --weights '${BEST_CKPT}' \
  --output '${OUT}' \
  --num-workers ${NUM_WORKERS}"
runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && conda run -n magformer python scripts/experiments/postprocess_cocoeval.py --dataset-root '${DATASET_ROOT}' --out-dir '${OUT}' --metrics-out 'metrics.cocoeval.json'"
runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && conda run -n magformer python scripts/analysis/write_metrics_std.py --out-dir '${OUT}' --iters-per-epoch ${ITERS_PER_EPOCH}"
runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && conda run -n magformer python scripts/analysis/prune_checkpoints.py --out-dir '${OUT}' --framework magformer"
runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && conda run -n magformer python scripts/analysis/write_run_metadata.py --phase end --out-dir '${OUT}'"
runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && CUDA_VISIBLE_DEVICES=0 bash scripts/experiments/run_20260406_training_campaign_gpu0.sh --register '${REGISTER}' --dataset-root '${DATASET_ROOT}' --output-base '${OUTPUT_BASE}' --run"

runner_log "${MODE}" "${RUN_LOG}" "[gpu0-finalize] done"
