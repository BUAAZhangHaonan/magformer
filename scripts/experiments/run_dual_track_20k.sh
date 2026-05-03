#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"

DATASET_ROOT_DEFAULT="${PROJECT_ROOT}/magformer_datasets/0909_512_0.12K"
DATASET_ROOT="${DATASET_ROOT_DEFAULT}"
MODE="run"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-root)
      DATASET_ROOT="$2"
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

run_cmd() {
  echo "+ $*"
  if [[ "${MODE}" == "run" ]]; then
    eval "$@"
  fi
}

echo "[dual-20k] mode=${MODE}"
echo "[dual-20k] dataset_root=${DATASET_ROOT}"

run_cmd "cd '${REPO_ROOT}' && bash scripts/experiments/run_track_a_20k.sh --dataset-root '${DATASET_ROOT}' --run"
run_cmd "cd '${REPO_ROOT}' && bash scripts/experiments/run_track_b_20k.sh --dataset-root '${DATASET_ROOT}' --run"

run_cmd "cd '${REPO_ROOT}' && python - <<'PY'
import json
from pathlib import Path

track_a = Path('output/experiments/track_a_20k/track_a_20k_summary.json')
track_b = Path('output/experiments/track_b_20k/track_b_20k_summary.json')
out = Path('output/experiments/final_20k/final_20k_summary.json')

if not track_a.exists():
    raise FileNotFoundError(track_a)
if not track_b.exists():
    raise FileNotFoundError(track_b)

a = json.loads(track_a.read_text(encoding='utf-8'))
b = json.loads(track_b.read_text(encoding='utf-8'))

payload = {
    'track_a_20k': a,
    'track_b_20k': b,
    'high_level': {
        'track_a_gap_best_ap': a.get('gap_best_ap'),
        'track_b_gap_best_ap': b.get('gap_best_ap'),
        'track_a_gap_last_ap': a.get('gap_last_ap'),
        'track_b_gap_last_ap': b.get('gap_last_ap'),
    },
}

out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps(payload, indent=2, ensure_ascii=False))
print(f'[dual-20k] summary saved to {out}')
PY"

echo "[dual-20k] done"
