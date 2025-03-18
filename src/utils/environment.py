import os
import torch
import transformers
from huggingface_hub import HfApi

test_environment = lambda: _test_environment()

def _test_environment():
    """Test if the Python LLM environment is set up correctly"""
    print("Testing Python LLM environment...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Transformers version: {transformers.__version__}")
    
    # Test CUDA availability
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Test MPS availability (for Apple Silicon)
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    print(f"MPS available (Apple Silicon): {mps_available}")
    
    # Determine available device
    if torch.cuda.is_available():
        recommended_device = "cuda"
    elif mps_available:
        recommended_device = "mps"
    else:
        recommended_device = "cpu"
    print(f"Recommended device: {recommended_device}")
    
    # Test HuggingFace Hub connectivity
    api = HfApi()
    try:
        models = api.list_models(limit=5)
        print("\nSuccessfully connected to Hugging Face Hub!")
        print("Sample models available:")
        for model in models:
            print(f"- {model.id}")
    except Exception as e:
        print(f"Error connecting to Hugging Face Hub: {e}")
    
    print("\nEnvironment test completed!")
    
    results = {
        "pytorch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": mps_available,
        "recommended_device": recommended_device,
        "huggingface_connected": True if "models" in locals() else False
    }
    
    return results 