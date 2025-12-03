#!/bin/bash
# Run full training pipeline

echo "======================================"
echo "Running Full Training Pipeline"
echo "======================================"

# Ensure we're in project root
cd "$(dirname "$0")/.."

echo ""
echo "Starting training..."
echo ""

python -m src.training.train_all_pipeline

echo ""
echo "======================================"
echo "Training Pipeline Complete!"
echo "======================================"
echo ""
echo "Checkpoints saved to: experiments/checkpoints/"
echo "Logs saved to: logs/"
