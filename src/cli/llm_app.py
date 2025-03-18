#!/usr/bin/env python3
import argparse
from src.models.llm import HuggingFaceLLM

def main():
    parser = argparse.ArgumentParser(description="Hugging Face LLM Text Generator")
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", help="Model name from Hugging Face")
    parser.add_argument("--prompt", type=str, default="Write a short poem about artificial intelligence:", help="Text prompt")
    parser.add_argument("--max_length", type=int, default=100, help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for text generation")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling parameter")
    
    args = parser.parse_args()
    
    # Initialize and use the model
    llm = HuggingFaceLLM(args.model)
    generated_text = llm.generate_text(
        args.prompt, 
        max_length=args.max_length,
        temperature=args.temperature,
        top_p=args.top_p
    )
    
    print("\nPrompt:", args.prompt)
    print("\nGenerated Text:")
    print(generated_text)

if __name__ == "__main__":
    main() 