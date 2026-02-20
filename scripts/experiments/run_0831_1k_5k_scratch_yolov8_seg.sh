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

OUT="${OUTPUT_ROOT}/yolov8_seg_scratch"
YOLO_DATA_DIR="${REPO_ROOT}/output/baselines/yolo_0831_1k"
YOLO_DATA_YAML="${YOLO_DATA_DIR}/dataset.yaml"

mkdir -p "${OUT}"
RUN_LOG="$(runner_setup_log "${OUT}" "${MODE}")"

runner_log "${MODE}" "${RUN_LOG}" "[yolov8-seg-0831-1k-5k-scratch] mode=${MODE}"
runner_log "${MODE}" "${RUN_LOG}" "[yolov8-seg-0831-1k-5k-scratch] dataset_root=${DATASET_ROOT}"
runner_log "${MODE}" "${RUN_LOG}" "[yolov8-seg-0831-1k-5k-scratch] output_root=${OUT}"

if [[ "${MODE}" == "run" ]]; then
  # Polars can crash with "illegal instruction" on some CPUs when it selects an
  # incompatible runtime wheel. Force the compat runtime for reproducibility.
  export POLARS_FORCE_PKG=compat

  if [[ ! -f "${YOLO_DATA_YAML}" ]]; then
    runner_exec "${MODE}" "${RUN_LOG}" "conda run -n magformer python '${REPO_ROOT}/baselines/ultralytics_tools/convert_coco_to_yolo_seg.py' --dataset-root '${DATASET_ROOT}' --output-root '${YOLO_DATA_DIR}'"
  fi

  # Ultralytics auto-increments run names if the dir exists (train2/train3...),
  # but our summarizer expects `${OUT}/train/results.csv`. Archive any previous
  # run to keep the canonical path stable.
  if [[ -d "${OUT}/train" ]]; then
    ts="$(date +%Y%m%d_%H%M%S)"
    mv "${OUT}/train" "${OUT}/train_prev_${ts}"
  fi
fi

# 5,000 iters @ batch=8 ~= 45 epochs (886 train images / 8 ~= 111 iters/epoch)
EPOCHS=45

SECONDS=0
runner_exec "${MODE}" "${RUN_LOG}" "cd '${REPO_ROOT}' && conda run -n magformer yolo segment train \
  model=yolov8n-seg.yaml \
  data='${YOLO_DATA_YAML}' \
  imgsz=512 \
  batch=8 \
  epochs=${EPOCHS} \
  device=0 \
  pretrained=False \
  project='${OUT}' \
  name='train'"
echo "${SECONDS}" > "${OUT}/wall_time_sec.txt"

if [[ "${MODE}" == "run" ]]; then
  runner_exec "${MODE}" "${RUN_LOG}" "conda run -n magformer python '${REPO_ROOT}/scripts/analysis/write_params_from_yolo_weights.py' --out-dir '${OUT}'"
fi

if [[ "${MODE}" == "run" ]]; then
  ANN_VAL="${DATASET_ROOT}/annotations/instances_val.json"

  runner_exec "${MODE}" "${RUN_LOG}" "conda run -n magformer python '${REPO_ROOT}/baselines/yolo_export_coco.py' --dataset-root '${DATASET_ROOT}' --ann-file '${ANN_VAL}' --split val --output-json '${OUT}/coco_instances_results.json'"

  runner_exec "${MODE}" "${RUN_LOG}" "conda run -n magformer python '${REPO_ROOT}/baselines/coco_eval_results.py' --ann-file '${ANN_VAL}' --results-json '${OUT}/coco_instances_results.json' --output-metrics '${OUT}/metrics.json'"
fi

runner_log "${MODE}" "${RUN_LOG}" "[yolov8-seg-0831-1k-5k-scratch] done"
