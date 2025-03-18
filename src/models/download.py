import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

download_model = lambda model_name, output_dir: _download_model(model_name, output_dir)

def _download_model(model_name, output_dir):
    """Download a model and tokenizer from Hugging Face and save them for offline use"""
    print(f"Downloading model: {model_name}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device for model loading - adding MPS support
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    
    # Download tokenizer
    print("Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set appropriate torch dtype based on device
    if device == "cuda":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32
    
    # Download model with appropriate configuration
    print("Downloading model...")
    if device == "mps":
        # For MPS device, load on CPU first
        print("Loading model on CPU first, then moving to MPS...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype
        )
        # Move to MPS device for testing
        model = model.to(device)
    else:
        # For CUDA or CPU
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto"
        )
    
    # Save model and tokenizer to disk
    print(f"Saving model and tokenizer to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("Download complete!")
    return output_dir 