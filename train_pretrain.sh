#!/bin/bash

# Set project root and add src to PYTHONPATH
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/src:$PYTHONPATH"

echo "Starting G-MAE Pre-training..."
echo "Project Root: ${PROJECT_ROOT}"

python "${PROJECT_ROOT}/src/scripts/pretrain.py" --config "${PROJECT_ROOT}/src/configs/pretrain.yaml"

echo "G-MAE Pre-training script finished."