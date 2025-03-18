import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class HuggingFaceLLM:
    def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.model_name = model_name
        
        # Check for available devices - adding MPS support for Apple Silicon
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
            
        print(f"Using device: {self.device}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set appropriate torch dtype based on device
        if self.device == "cuda":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32
            
        # Load model with appropriate configuration
        if self.device == "mps":
            # For MPS device, first load on CPU then move to MPS
            print("Loading model on CPU first, then moving to MPS...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype
            )
            # Manually move model to MPS device
            self.model = self.model.to(self.device)
        else:
            # For CUDA or CPU
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                torch_dtype=torch_dtype,
                device_map="auto"
            )
        
    generate_text = lambda self, prompt, max_length=100, temperature=0.7, top_p=0.9: self._generate_text(prompt, max_length, temperature, top_p)
        
    def _generate_text(self, prompt, max_length=100, temperature=0.7, top_p=0.9):
        """Generate text based on a prompt"""
        # Handle TinyLlama chat format
        if "TinyLlama" in self.model_name and "Chat" in self.model_name:
            formatted_prompt = f"<human>: {prompt}\n<assistant>:"
        else:
            formatted_prompt = prompt
            
        print(f"Using formatted prompt: {formatted_prompt}")
            
        # Ensure inputs are on the correct device
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate text
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,  # Use max_new_tokens instead of max_length
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.2
            )
        
        # Decode the generated text
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # For TinyLlama chat models, extract only the assistant's response
        if "TinyLlama" in self.model_name and "Chat" in self.model_name:
            # Check if the formatted prompt is in the generated text and remove it
            if formatted_prompt in generated_text:
                assistant_response = generated_text.replace(formatted_prompt, "").strip()
                return assistant_response
                
            # Alternative extraction method
            parts = generated_text.split("<assistant>:")
            if len(parts) > 1:
                return parts[1].strip()
            
            # If all else fails, return the full response
            return generated_text
        
        return generated_text 