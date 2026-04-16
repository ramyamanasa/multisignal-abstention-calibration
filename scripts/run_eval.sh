#!/bin/bash
# run_eval.sh
# Runs the full evaluation pipeline end-to-end.
# Usage: bash scripts/run_eval.sh

set -e

echo "Running full evaluation pipeline..."

python src/pipeline.py \
    --dataset data/processed/triviaqa_dev.jsonl \
    --output experiments/latest_run.json \
    --threshold 0.5

echo "Done. Results saved to experiments/latest_run.json"
