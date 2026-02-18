#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Lint only our code, not vendored baselines.
TARGETS=(magformer tools scripts tests)

if python -c "import ruff" >/dev/null 2>&1; then
  python -m ruff check "${TARGETS[@]}"
else
  echo "[lint] ruff not installed. Install dev deps: pip install -r requirements-dev.txt" >&2
fi

python -m black --check "${TARGETS[@]}"

if python -c "import isort" >/dev/null 2>&1; then
  python -m isort --check-only "${TARGETS[@]}"
else
  echo "[lint] isort not installed. Install dev deps: pip install -r requirements-dev.txt" >&2
fi
