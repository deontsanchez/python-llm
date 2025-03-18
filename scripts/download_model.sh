#!/bin/bash
# Script to download a model from Hugging Face

# Default model
DEFAULT_MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_OUTPUT="./downloaded_model"

# Parse command line arguments
MODEL=${1:-$DEFAULT_MODEL}
OUTPUT=${2:-$DEFAULT_OUTPUT}

# Print header
echo "==============================================================="
echo "Python LLM: Downloading model $MODEL to $OUTPUT"
echo "==============================================================="

# Run the Python script
python -m src.cli.download_model --model_name "$MODEL" --output_dir "$OUTPUT" 