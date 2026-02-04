#!/bin/bash

set -e

echo "Running data processing..."
python3 "./data-collection/data_processing.py"

# echo "Training RNN model..."
# python3 "./ml_models/train_rnn_model.py"

echo "Generating daily projections..."
python3 "./ml_models/generate_daily_projections.py"

echo "All scripts finished successfully."
