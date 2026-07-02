#!/bin/bash
# Usage: bash run.sh <dataset> [stage]
# Examples:
#   bash run.sh richpedia          → full pipeline
#   bash run.sh wikimel 3          → cross-encoder only
#   bash run.sh wikidiverse eval   → evaluation only

DATASET=${1:-richpedia}
STAGE=${2:-all}

echo "=============================="
echo "VSCE-EL | Dataset: $DATASET | Stage: $STAGE"
echo "=============================="

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3 main.py \
  --config configs/${DATASET}.yaml \
  --stage  ${STAGE}
