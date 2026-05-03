#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUTPUT_ROOT="${REPO_ROOT}/output/experiments/0831_1k_20ep_trackp"

if [[ $# -gt 0 ]]; then
  OUTPUT_ROOT="$1"
fi

cd "${REPO_ROOT}"
python scripts/experiments/summarize_suite.py \
  --output-root "${OUTPUT_ROOT}" \
  --write \
  --write-name "summary_$(basename "${OUTPUT_ROOT}")_raw.json"
