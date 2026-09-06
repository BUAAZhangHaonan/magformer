#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Format only our code, not vendored baselines.
TARGETS=(magformer tools scripts tests)

if python -c "import isort" >/dev/null 2>&1; then
  python -m isort "${TARGETS[@]}"
else
  echo "[format] isort not installed. Install dev deps: pip install -r requirements-dev.txt" >&2
fi

python -m black "${TARGETS[@]}"
