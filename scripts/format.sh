#!/usr/bin/env bash
# scripts/format.sh - Simple script to format code with uv

set -euo pipefail

# Get the root directory of the project (parent of scripts directory)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Change to root directory
cd "$ROOT_DIR"

TARGET=${1:-"."}
LINE_LENGTH=${2:-120}

echo "Formatting Python code in $ROOT_DIR/$TARGET (line length: $LINE_LENGTH)"

# Install tools if needed
uv pip install black isort pyflakes 2>/dev/null || true

# Format with black
echo " Running black..."
uv run black --line-length $LINE_LENGTH $TARGET

# Sort imports with isort
echo " Running isort..."
uv run isort --line-length $LINE_LENGTH --profile black $TARGET

# Quick check with pyflakes (scoped to Python directories to avoid
# RecursionError on non-Python files like uv.lock)
echo " Running pyflakes..."
if [ "$TARGET" = "." ]; then
    uv run pyflakes src/ tests/
else
    uv run pyflakes $TARGET
fi

echo " Done!"