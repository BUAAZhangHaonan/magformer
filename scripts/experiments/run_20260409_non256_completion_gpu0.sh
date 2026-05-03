#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec bash "${SCRIPT_DIR}/run_20260409_non256_completion_gpu1.sh" \
  --gpu 0 \
  --queue-tag 20260409_gpu0_non256_backfill \
  "$@"
