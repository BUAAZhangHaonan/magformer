#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"

DATASET_ROOT_DEFAULT="${PROJECT_ROOT}/magformer_datasets/0909_512_0.12K"
OUTPUT_ROOT_DEFAULT="${REPO_ROOT}/output/experiments/track_a_20k"

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

MAGFORMER_DIR="${REPO_ROOT}"
MASK2FORMER_DIR="${REPO_ROOT}/baselines/MGM_Mask2Former"

MAGFORMER_CONFIG="${MAGFORMER_DIR}/configs/magformer_track_a_20k.yaml"
MASK2FORMER_CONFIG="${MASK2FORMER_DIR}/configs/mgm_aligned_comparison.yaml"

MAGFORMER_OUT="${OUTPUT_ROOT}/magformer"
MASK2FORMER_OUT="${OUTPUT_ROOT}/mask2former"
SUMMARY_FILE="${OUTPUT_ROOT}/track_a_20k_summary.json"

mkdir -p "${MAGFORMER_OUT}" "${MASK2FORMER_OUT}"

run_cmd() {
  echo "+ $*"
  if [[ "${MODE}" == "run" ]]; then
    eval "$@"
  fi
}

echo "[track-a-20k] mode=${MODE}"
echo "[track-a-20k] dataset_root=${DATASET_ROOT}"
echo "[track-a-20k] output_root=${OUTPUT_ROOT}"

run_cmd "cd '${MAGFORMER_DIR}' && conda run -n magformer python tools/train.py --config '${MAGFORMER_CONFIG}' --dataset-root '${DATASET_ROOT}' --output-dir '${MAGFORMER_OUT}' --num-workers 4"

run_cmd "cd '${MASK2FORMER_DIR}' && conda run -n magformer python train_net_mgm_0831.py --num-gpus 1 --config-file '${MASK2FORMER_CONFIG}' INPUT.DATASET_ROOT '${DATASET_ROOT}' OUTPUT_DIR '${MASK2FORMER_OUT}' MODEL.FINETUNE_WEIGHTS '' MODEL.WEIGHTS '' SOLVER.MAX_ITER 20000 SOLVER.STEPS '(15000,18000)' SOLVER.WARMUP_ITERS 2000 SOLVER.BASE_LR 0.00005 SOLVER.IMS_PER_BATCH 8 TEST.EVAL_PERIOD 2000 DATALOADER.NUM_WORKERS 4"

run_cmd "cd '${REPO_ROOT}' && python - <<'PY'
import json
import math
from pathlib import Path

mag_metrics_file = Path('${MAGFORMER_OUT}') / 'metrics_log.jsonl'
m2f_metrics_file = Path('${MASK2FORMER_OUT}') / 'metrics.json'
summary_file = Path('${SUMMARY_FILE}')

def load_jsonl(path):
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

mag_rows = load_jsonl(mag_metrics_file)
mag_val_rows = [r for r in mag_rows if r.get('phase') == 'val' and 'val/segm_AP' in r]
if not mag_val_rows:
    raise RuntimeError(f'No val rows with segm_AP in {mag_metrics_file}')
mag_last = mag_val_rows[-1]
mag_best = max(mag_val_rows, key=lambda r: float(r.get('val/segm_AP', -1e9)))

def to_pct(x):
    v = float(x)
    return v * 100.0

def as_none_if_nan(x):
    if isinstance(x, float) and math.isnan(x):
        return None
    return float(x)

m2f_rows = load_jsonl(m2f_metrics_file)
m2f_val_rows = [r for r in m2f_rows if 'segm/AP' in r]
if not m2f_val_rows:
    raise RuntimeError(f'No segm/AP rows in {m2f_metrics_file}')
m2f_last = m2f_val_rows[-1]
m2f_best = max(m2f_val_rows, key=lambda r: float(r.get('segm/AP', -1e9)))

summary = {
    'track': 'track_a_20k',
    'scale': 'percentage',
    'magformer': {
        'last_iter': int(mag_last.get('iter', -1)),
        'last': {
            'AP': to_pct(mag_last.get('val/segm_AP', 0.0)),
            'AP50': to_pct(mag_last.get('val/segm_AP50', 0.0)),
            'AP75': to_pct(mag_last.get('val/segm_AP75', 0.0)),
            'APs': to_pct(mag_last.get('val/segm_AP_small', 0.0)),
            'APm': to_pct(mag_last.get('val/segm_AP_medium', 0.0)),
            'APl': None if float(mag_last.get('val/segm_AP_large', -1.0)) < 0 else to_pct(mag_last.get('val/segm_AP_large', 0.0)),
        },
        'best': {
            'iter': int(mag_best.get('iter', -1)),
            'AP': to_pct(mag_best.get('val/segm_AP', 0.0)),
            'AP50': to_pct(mag_best.get('val/segm_AP50', 0.0)),
            'AP75': to_pct(mag_best.get('val/segm_AP75', 0.0)),
            'APs': to_pct(mag_best.get('val/segm_AP_small', 0.0)),
            'APm': to_pct(mag_best.get('val/segm_AP_medium', 0.0)),
            'APl': None if float(mag_best.get('val/segm_AP_large', -1.0)) < 0 else to_pct(mag_best.get('val/segm_AP_large', 0.0)),
        },
    },
    'mask2former': {
        'last_iter': int(m2f_last.get('iteration', -1)),
        'last': {
            'AP': float(m2f_last.get('segm/AP', 0.0)),
            'AP50': float(m2f_last.get('segm/AP50', 0.0)),
            'AP75': float(m2f_last.get('segm/AP75', 0.0)),
            'APs': float(m2f_last.get('segm/APs', 0.0)),
            'APm': float(m2f_last.get('segm/APm', 0.0)),
            'APl': as_none_if_nan(m2f_last.get('segm/APl', float('nan'))),
        },
        'best': {
            'iter': int(m2f_best.get('iteration', -1)),
            'AP': float(m2f_best.get('segm/AP', 0.0)),
            'AP50': float(m2f_best.get('segm/AP50', 0.0)),
            'AP75': float(m2f_best.get('segm/AP75', 0.0)),
            'APs': float(m2f_best.get('segm/APs', 0.0)),
            'APm': float(m2f_best.get('segm/APm', 0.0)),
            'APl': as_none_if_nan(m2f_best.get('segm/APl', float('nan'))),
        },
    },
}

summary['gap_last_ap'] = summary['magformer']['last']['AP'] - summary['mask2former']['last']['AP']
summary['gap_best_ap'] = summary['magformer']['best']['AP'] - summary['mask2former']['best']['AP']

summary_file.parent.mkdir(parents=True, exist_ok=True)
with open(summary_file, 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(json.dumps(summary, indent=2, ensure_ascii=False))
print(f'[track-a-20k] summary saved to {summary_file}')
PY"

echo "[track-a-20k] done"
