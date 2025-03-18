#!/bin/bash
# Script to run the LLM application

# Default model and parameters
DEFAULT_MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_PROMPT="Write a short poem about artificial intelligence:"
DEFAULT_MAX_LENGTH=100
DEFAULT_TEMPERATURE=0.7
DEFAULT_TOP_P=0.9

# Parse command line arguments
MODEL=${1:-$DEFAULT_MODEL}
PROMPT=${2:-$DEFAULT_PROMPT}
MAX_LENGTH=${3:-$DEFAULT_MAX_LENGTH}
TEMPERATURE=${4:-$DEFAULT_TEMPERATURE}
TOP_P=${5:-$DEFAULT_TOP_P}

# Print header
echo "==============================================================="
echo "Python LLM: Running model $MODEL"
echo "==============================================================="

# Run the Python script
python -m src.cli.llm_app \
  --model "$MODEL" \
  --prompt "$PROMPT" \
  --max_length $MAX_LENGTH \
  --temperature $TEMPERATURE \
  --top_p $TOP_P 