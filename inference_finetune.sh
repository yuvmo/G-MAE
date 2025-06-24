#!/bin/bash

# Set project root and add src to PYTHONPATH
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/src:$PYTHONPATH"

echo "Starting G-MAE Inference..."
echo "Project Root: ${PROJECT_ROOT}"

python "${PROJECT_ROOT}/src/scripts/inference.py" --config "${PROJECT_ROOT}/src/configs/inference.yaml"

echo "G-MAE Inference script finished."