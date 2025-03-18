import os
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling
)

tokenize_function = lambda examples, tokenizer, max_length: {
    "input_ids": tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt"
    ).input_ids
}

fine_tune_model = lambda model_name, dataset_name, output_dir, **kwargs: _fine_tune_model(
    model_name, dataset_name, output_dir, **kwargs
)

def _fine_tune_model(
    model_name, 
    dataset_name, 
    output_dir, 
    dataset_config=None, 
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=5e-5,
    max_length=512
):
    """Fine-tune a Hugging Face model on a dataset"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device with MPS support
    if torch.cuda.is_available():
        device = "cuda"
        torch_dtype = torch.float16
    elif torch.backends.mps.is_available():
        device = "mps"
        torch_dtype = torch.float32
    else:
        device = "cpu"
        torch_dtype = torch.float32
    
    print(f"Using device: {device}")
    
    # Load dataset
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, dataset_config)
    
    # Load tokenizer
    print(f"Loading tokenizer for model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Tokenize the dataset
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, max_length),
        batched=True,
        remove_columns=["text"],
    )
    
    # Load the model
    print(f"Loading model: {model_name}")
    if device == "mps":
        # First load on CPU then move to MPS
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype
        )
        model = model.to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto"
        )
    
    # Training arguments
    print("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        save_strategy="epoch",
        save_total_limit=2,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        fp16=(device == "cuda"),
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        data_collator=data_collator,
    )
    
    # Train the model
    print("Starting training...")
    trainer.train()
    
    # Save the model
    print(f"Saving fine-tuned model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("Fine-tuning complete!")
    return output_dir 