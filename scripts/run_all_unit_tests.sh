#!/usr/bin/env bash
# scripts/run_all_unit_tests.sh - Run unit tests with uv (same idea as CI: uv sync --dev, then pytest)

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "Running tests in $ROOT_DIR"

uv sync --dev

if [ "$#" -eq 0 ]; then
  uv run pytest tests
else
  uv run pytest "$@"
fi

echo "Done!"
