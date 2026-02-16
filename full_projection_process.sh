#!/bin/bash

set -e

echo "Running data processing..."
python3 "./data-collection/data_processing.py"

# echo "Training RNN model..."
# python3 "./ml_models/train_rnn_model.py"

echo "Generating daily projections..."
python3 "./ml_models/generate_daily_projections.py"

echo "Removing old projections and uploading new projections to database..."
python "./data-collection/database/bulk_upload.py"

echo "Tracking model success..."
python "track_model_success.py"

echo "All scripts finished successfully."
