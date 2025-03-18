#!/usr/bin/env python3
import argparse
from src.models.download import download_model

def main():
    parser = argparse.ArgumentParser(description="Download a Hugging Face model for offline use")
    parser.add_argument("--model_name", type=str, required=True, help="Model name from Hugging Face")
    parser.add_argument("--output_dir", type=str, default="./downloaded_model", help="Directory to save the model")
    
    args = parser.parse_args()
    
    download_model(args.model_name, args.output_dir)

if __name__ == "__main__":
    main() 