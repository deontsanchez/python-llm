# Python LLM Virtual Environment Guide

This guide provides instructions for activating and using your Python LLM virtual environment.

## Activating the Environment

### On macOS/Linux:

```bash
source llm-env/bin/activate
```

You'll know the environment is activated when you see `(llm-env)` at the beginning of your terminal prompt.

### On Windows:

```bash
llm-env\Scripts\activate
```

## Testing the Environment

After activating the environment, you can run the test script to verify everything is working correctly:

```bash
python test_env.py
```

## Using the Application

Once the environment is activated, you can use any of the LLM application scripts:

1. Generate text with a pre-trained model:

   ```bash
   python llm_app.py --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --prompt "Write a short story about space exploration:"
   ```

2. Fine-tune a model on a dataset:

   ```bash
   python fine_tune.py --model_name "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --dataset "imdb"
   ```

3. Download a model for offline use:
   ```bash
   python download_model.py --model_name "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
   ```

## Apple Silicon Support (M1/M2/M3 Macs)

The application includes support for Metal Performance Shaders (MPS) on Apple Silicon Macs. For optimal performance:

1. Ensure you have PyTorch 2.0+ installed with MPS support:

   ```bash
   pip install --upgrade torch
   ```

2. You can verify MPS availability with:

   ```bash
   python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
   ```

3. When running the application, it will automatically detect MPS and use it.

4. For best performance on Apple Silicon:
   - Start with smaller models (1B-2B parameters)
   - The application will automatically load models on CPU first and then move them to MPS
   - Limit batch sizes when fine-tuning to avoid memory issues

## Installing Additional Dependencies

If you need to install additional packages:

```bash
pip install package_name
```

## Deactivating the Environment

When you're done using the environment, deactivate it by running:

```bash
deactivate
```

## Troubleshooting

- **MPS Device Errors**: If you encounter "Placeholder storage has not been allocated on MPS device" errors, the application should handle this automatically. If errors persist, try:

  ```bash
  # Set environment variable to disable MPS
  export PYTORCH_ENABLE_MPS_FALLBACK=1
  ```

- If you encounter issues with sentencepiece, you might need to install system dependencies:

  ```bash
  # On macOS with Homebrew
  brew install cmake pkg-config

  # Then install sentencepiece
  pip install sentencepiece
  ```

- For any other dependency issues, check the error message and install any missing packages using pip.

- When using models that require authentication (like Gemma or Llama 2), you'll need to login with Hugging Face:
  ```bash
  huggingface-cli login
  ```
