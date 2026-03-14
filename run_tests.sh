#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "Running all tests..."

# Run pytest using uv
uv run pytest tests/

echo "All tests passed successfully!"
