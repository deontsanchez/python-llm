#!/usr/bin/env python3
import argparse
from src.models.fine_tune import fine_tune_model

def main():
    parser = argparse.ArgumentParser(description="Fine-tune a Hugging Face model")
    parser.add_argument("--model_name", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", help="Base model to fine-tune")
    parser.add_argument("--dataset", type=str, default="imdb", help="Dataset from Hugging Face to use for fine-tuning")
    parser.add_argument("--dataset_config", type=str, default=None, help="Dataset configuration name")
    parser.add_argument("--output_dir", type=str, default="./fine_tuned_model", help="Directory to save the fine-tuned model")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Batch size per device during training")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    
    args = parser.parse_args()
    
    fine_tune_model(
        model_name=args.model_name,
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        dataset_config=args.dataset_config,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length
    )

if __name__ == "__main__":
    main() 