#!/bin/bash

# Set project root and add src to PYTHONPATH
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/src:$PYTHONPATH"

echo "Starting G-MAE Fine-tuning..."
echo "Project Root: ${PROJECT_ROOT}"

python "${PROJECT_ROOT}/src/scripts/finetune.py" --config "${PROJECT_ROOT}/src/configs/finetune.yaml"

echo "G-MAE Fine-tuning script finished."