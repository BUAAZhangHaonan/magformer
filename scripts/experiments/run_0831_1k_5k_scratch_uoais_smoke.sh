#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"

DATASET_ROOT_DEFAULT="${PROJECT_ROOT}/magformer_datasets/0831_1K"
OUTPUT_ROOT_DEFAULT="${REPO_ROOT}/output/experiments/0831_1k_5k_scratch8_smoke"

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

UOAIS_ROOT="${REPO_ROOT}/baselines/uoais"
CFG="${REPO_ROOT}/configs/baselines/uoais_0831_1k_5k_scratch.yaml"
OUT="${OUTPUT_ROOT}/uoais_scratch"

mkdir -p "${OUT}"

run_cmd() {
  echo "+ $*"
  if [[ "${MODE}" == "run" ]]; then
    eval "$@"
  fi
}

echo "[uoais-0831-1k-smoke] mode=${MODE}"
echo "[uoais-0831-1k-smoke] dataset_root=${DATASET_ROOT}"
echo "[uoais-0831-1k-smoke] output_dir=${OUT}"
echo "[uoais-0831-1k-smoke] config=${CFG}"

SECONDS=0
run_cmd "cd '${REPO_ROOT}' && conda run -n magformer python baselines/run_uoais_0831_1k.py \
  --dataset-root '${DATASET_ROOT}' \
  --uoais-root '${UOAIS_ROOT}' \
  -- \
  --num-gpus 1 \
  --config-file '${CFG}' \
  OUTPUT_DIR '${OUT}' \
  SOLVER.IMS_PER_BATCH 2 \
  SOLVER.MAX_ITER 20 \
  SOLVER.STEPS '(15,18)' \
  SOLVER.WARMUP_ITERS 20 \
  TEST.EVAL_PERIOD 10 \
  SOLVER.CHECKPOINT_PERIOD 20"
echo "${SECONDS}" > "${OUT}/wall_time_sec.txt"

if [[ "${MODE}" == "run" ]]; then
  OUT_DIR="${OUT}" conda run -n magformer python -c "import os; from pathlib import Path; import torch; out_dir=Path(os.environ['OUT_DIR']); ckpt=out_dir/'model_final.pth'; ckpts=[ckpt] if ckpt.exists() else sorted(out_dir.glob('model_*.pth')); print('[uoais-smoke] ckpts', [p.name for p in ckpts]); state=torch.load(ckpts[-1], map_location='cpu'); state=state.get('model', state) if isinstance(state, dict) else state; n=sum(int(v.numel()) for v in state.values() if hasattr(v,'numel')); (out_dir/'params_trainable.txt').write_text(str(n)+'\\n', encoding='utf-8'); print('[uoais-smoke] params_trainable', n)"
fi

echo "[uoais-0831-1k-smoke] done"
