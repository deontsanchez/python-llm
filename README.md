# Python LLM Application with Hugging Face

This project provides a simple interface to use and fine-tune language models from Hugging Face. It supports running models on CPU, CUDA GPUs, and Apple Silicon (via MPS).

## Features

- Load and use pre-trained language models from Hugging Face
- Automatic device detection (CUDA, MPS for Apple Silicon, CPU)
- Proper handling of chat-formatted models
- Fine-tune models on custom datasets
- Download models for offline use

## Project Structure

```
python-llm/
├── config/                  # Configuration files
│   └── example_models.yaml  # Example model configurations
├── scripts/                 # Helper shell scripts
│   ├── download_model.sh    # Download a model
│   ├── fine_tune.sh         # Fine-tune a model
│   ├── run_llm.sh           # Run text generation
│   └── test_env.sh          # Test environment
├── src/                     # Source code
│   ├── cli/                 # Command-line interfaces
│   │   ├── download_model.py
│   │   ├── fine_tune.py
│   │   ├── llm_app.py
│   │   └── test_env.py
│   ├── models/              # Model-related functionality
│   │   ├── download.py      # Model downloading
│   │   ├── fine_tune.py     # Fine-tuning functionality
│   │   └── llm.py           # LLM interfaces
│   └── utils/               # Utility functions
│       ├── config.py        # Configuration loading utilities
│       └── environment.py   # Environment testing
├── .gitignore               # Git ignore file
├── llm_cli.py               # Unified command-line interface
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

## Installation

1. Clone this repository:

```bash
git clone <your-repo-url>
cd <repo-folder>
```

2. Create and activate a virtual environment:

```bash
python -m venv llm-env
source llm-env/bin/activate  # On Linux/Mac
llm-env\Scripts\activate     # On Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Unified Command-Line Interface

The project provides a unified CLI for all functionality:

```bash
# Test the environment
./llm_cli.py test

# Generate text with a model
./llm_cli.py generate --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --prompt "Write a story about space:"

# Download a model for offline use
./llm_cli.py download --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --output "./my_model"

# Fine-tune a model
./llm_cli.py finetune --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --dataset "imdb" --output "./fine_tuned"
```

### Individual Scripts

You can also use the individual shell scripts for specific tasks:

### Testing Your Environment

To verify your environment is set up correctly:

```bash
./scripts/test_env.sh
```

### Running the LLM

To generate text with a pre-trained model:

```bash
./scripts/run_llm.sh [MODEL_NAME] [PROMPT] [MAX_LENGTH] [TEMPERATURE] [TOP_P]
```

Example:

```bash
./scripts/run_llm.sh "TinyLlama/TinyLlama-1.1B-Chat-v1.0" "Write a poem about AI:"
```

### Downloading a Model

To download a model for offline use:

```bash
./scripts/download_model.sh [MODEL_NAME] [OUTPUT_DIR]
```

Example:

```bash
./scripts/download_model.sh "TinyLlama/TinyLlama-1.1B-Chat-v1.0" "./my_model"
```

### Fine-tuning a Model

To fine-tune a model on a dataset:

```bash
./scripts/fine_tune.sh [MODEL_NAME] [DATASET] [OUTPUT_DIR] [EPOCHS] [BATCH_SIZE] [LEARNING_RATE] [MAX_LENGTH]
```

Example:

```bash
./scripts/fine_tune.sh "TinyLlama/TinyLlama-1.1B-Chat-v1.0" "imdb" "./fine_tuned_model"
```

## Advanced Usage

For more advanced usage, you can directly use the Python modules:

```python
from src.models.llm import HuggingFaceLLM

# Initialize the model
llm = HuggingFaceLLM("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Generate text
response = llm.generate_text("Write a poem about artificial intelligence:",
                           max_length=100,
                           temperature=0.7,
                           top_p=0.9)
print(response)
```

## Configuration

The project uses YAML configuration files stored in the `config/` directory. Example configurations are prefixed with `example_` and are included in the repository. To create your own configuration:

1. Copy an example configuration file and remove the `example_` prefix:

```bash
cp config/example_models.yaml config/models.yaml
```

2. Edit the file to customize settings for your needs.

3. Access the configuration in code:

```python
from src.utils.config import load_config

# Load configuration
config = load_config('models')

# Access configuration values
default_chat_model = config['default_models']['chat']
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

[Your contribution guidelines]
