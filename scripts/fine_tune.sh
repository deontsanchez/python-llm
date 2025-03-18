#!/bin/bash
# Script to fine-tune an LLM on a dataset

# Default parameters
DEFAULT_MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_DATASET="imdb"
DEFAULT_OUTPUT="./fine_tuned_model"
DEFAULT_EPOCHS=3
DEFAULT_BATCH_SIZE=4
DEFAULT_LEARNING_RATE=5e-5
DEFAULT_MAX_LENGTH=512

# Parse command line arguments
MODEL=${1:-$DEFAULT_MODEL}
DATASET=${2:-$DEFAULT_DATASET}
OUTPUT=${3:-$DEFAULT_OUTPUT}
EPOCHS=${4:-$DEFAULT_EPOCHS}
BATCH_SIZE=${5:-$DEFAULT_BATCH_SIZE}
LEARNING_RATE=${6:-$DEFAULT_LEARNING_RATE}
MAX_LENGTH=${7:-$DEFAULT_MAX_LENGTH}

# Print header
echo "==============================================================="
echo "Python LLM: Fine-tuning model $MODEL on dataset $DATASET"
echo "==============================================================="

# Run the Python script
python -m src.cli.fine_tune \
  --model_name "$MODEL" \
  --dataset "$DATASET" \
  --output_dir "$OUTPUT" \
  --num_train_epochs $EPOCHS \
  --per_device_train_batch_size $BATCH_SIZE \
  --learning_rate $LEARNING_RATE \
  --max_length $MAX_LENGTH 