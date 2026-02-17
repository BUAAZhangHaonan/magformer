#!/usr/bin/env bash
set -euo pipefail

# Reproduce frozen baseline runs for MagFormer and Mask2Former.
# Default mode is dry-run to avoid accidental long GPU jobs.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"

DATASET_ROOT_DEFAULT="${PROJECT_ROOT}/magformer_datasets/0909_512_0.12K"
OUTPUT_ROOT_DEFAULT="${REPO_ROOT}/output/experiments/baseline_freeze"

DATASET_ROOT="${DATASET_ROOT_DEFAULT}"
OUTPUT_ROOT="${OUTPUT_ROOT_DEFAULT}"
MODE="dry-run"

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

MAGFORMER_DIR="${REPO_ROOT}"
MASK2FORMER_DIR="${PROJECT_ROOT}/mask2former/MGM_Mask2Former"

MAGFORMER_CONFIG="${MAGFORMER_DIR}/configs/magformer_aligned_comparison.yaml"
MASK2FORMER_CONFIG="${MASK2FORMER_DIR}/configs/mgm_aligned_comparison.yaml"
MASK2FORMER_HIGH_AP_INIT="${MASK2FORMER_DIR}/pretrained-checkpoint/0909_512_0.12K_0909_20K.pth"

MAGFORMER_OUT="${OUTPUT_ROOT}/magformer_aligned_2k"
MASK2FORMER_OUT="${OUTPUT_ROOT}/mask2former_reference_2k"
SNAPSHOT_DIR="${OUTPUT_ROOT}/config_snapshots"

mkdir -p "${MAGFORMER_OUT}" "${MASK2FORMER_OUT}" "${SNAPSHOT_DIR}"
cp -f "${MAGFORMER_CONFIG}" "${SNAPSHOT_DIR}/magformer_aligned_comparison.yaml"
cp -f "${MASK2FORMER_CONFIG}" "${SNAPSHOT_DIR}/mgm_aligned_comparison.yaml"

run_cmd() {
  echo "+ $*"
  if [[ "${MODE}" == "run" ]]; then
    eval "$@"
  fi
}

echo "[baseline-freeze] mode=${MODE}"
echo "[baseline-freeze] dataset_root=${DATASET_ROOT}"
echo "[baseline-freeze] output_root=${OUTPUT_ROOT}"

run_cmd "cd '${MAGFORMER_DIR}' && conda run -n magformer python tools/train.py --config configs/magformer_aligned_comparison.yaml --dataset-root '${DATASET_ROOT}' --output-dir '${MAGFORMER_OUT}' --num-workers 4"

run_cmd "cd '${MASK2FORMER_DIR}' && conda run -n magformer python train_net_mgm_0831.py --num-gpus 1 --config-file configs/mgm_swin_convnext_tiny.yaml INPUT.DATASET_ROOT '${DATASET_ROOT}' OUTPUT_DIR '${MASK2FORMER_OUT}' MODEL.FINETUNE_WEIGHTS '${MASK2FORMER_HIGH_AP_INIT}' MODEL.WEIGHTS '' MODEL.PIXEL_MEAN '[145.8863,85.7720,88.8209]' MODEL.PIXEL_STD '[47.4676,69.8726,67.9742]' SOLVER.MAX_ITER 2000 SOLVER.STEPS '(1600,1800)' SOLVER.BASE_LR 0.0001 SOLVER.WARMUP_ITERS 100 SOLVER.IMS_PER_BATCH 8 TEST.EVAL_PERIOD 200 DATALOADER.NUM_WORKERS 4"

echo "[baseline-freeze] done"
