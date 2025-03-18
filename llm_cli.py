#!/usr/bin/env python3
"""
Python LLM Command Line Interface
--------------------------------
A unified CLI for working with Hugging Face language models.
"""
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(
        description="Python LLM CLI - A unified interface for Hugging Face language models"
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Generate text command
    generate_parser = subparsers.add_parser("generate", help="Generate text with a language model")
    generate_parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", help="Model name")
    generate_parser.add_argument("--prompt", type=str, default="Write a short poem about artificial intelligence:", help="Text prompt")
    generate_parser.add_argument("--max_length", type=int, default=100, help="Maximum tokens to generate")
    generate_parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for generation")
    generate_parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling parameter")
    
    # Download command
    download_parser = subparsers.add_parser("download", help="Download a model for offline use")
    download_parser.add_argument("--model", type=str, required=True, help="Model name from Hugging Face")
    download_parser.add_argument("--output", type=str, default="./downloaded_model", help="Output directory")
    
    # Fine-tune command
    finetune_parser = subparsers.add_parser("finetune", help="Fine-tune a model")
    finetune_parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", help="Base model")
    finetune_parser.add_argument("--dataset", type=str, default="imdb", help="Dataset name")
    finetune_parser.add_argument("--dataset_config", type=str, default=None, help="Dataset configuration")
    finetune_parser.add_argument("--output", type=str, default="./fine_tuned_model", help="Output directory")
    finetune_parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    finetune_parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    finetune_parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    finetune_parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    
    # Test command
    subparsers.add_parser("test", help="Test the environment")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Process commands
    if args.command == "generate":
        from src.cli.llm_app import main as generate_main
        sys.argv = [sys.argv[0]]
        if args.model != "TinyLlama/TinyLlama-1.1B-Chat-v1.0": sys.argv.extend(["--model", args.model])
        if args.prompt != "Write a short poem about artificial intelligence:": sys.argv.extend(["--prompt", args.prompt])
        if args.max_length != 100: sys.argv.extend(["--max_length", str(args.max_length)])
        if args.temperature != 0.7: sys.argv.extend(["--temperature", str(args.temperature)])
        if args.top_p != 0.9: sys.argv.extend(["--top_p", str(args.top_p)])
        generate_main()
    
    elif args.command == "download":
        from src.cli.download_model import main as download_main
        sys.argv = [sys.argv[0], "--model_name", args.model, "--output_dir", args.output]
        download_main()
    
    elif args.command == "finetune":
        from src.cli.fine_tune import main as finetune_main
        sys.argv = [sys.argv[0]]
        sys.argv.extend(["--model_name", args.model])
        sys.argv.extend(["--dataset", args.dataset])
        sys.argv.extend(["--output_dir", args.output])
        if args.dataset_config: sys.argv.extend(["--dataset_config", args.dataset_config])
        sys.argv.extend(["--num_train_epochs", str(args.epochs)])
        sys.argv.extend(["--per_device_train_batch_size", str(args.batch_size)])
        sys.argv.extend(["--learning_rate", str(args.learning_rate)])
        sys.argv.extend(["--max_length", str(args.max_length)])
        finetune_main()
    
    elif args.command == "test":
        from src.cli.test_env import main as test_main
        test_main()
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 