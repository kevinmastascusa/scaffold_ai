#!/usr/bin/env python3
"""
Simple Llama 3.1 Test - Final Version
Just test one curriculum query to verify everything works
"""

import os
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load environment variables from .env file first
def load_env_file():
    """Load environment variables from .env file."""
    env_file = ".env"
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value
        print("✓ Environment variables loaded from .env file")

# Load environment before importing any modules
load_env_file()

def test_single_query():
    """Test Llama 3.1 8B with one curriculum query."""
    print("🚀 Testing Llama 3.1 8B with Curriculum Query")
    print("=" * 50)
    
    # Get token
    token = os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        print("❌ No HUGGINGFACE_TOKEN found")
        return
    
    print(f"✓ Using token: {token[:10]}...")
    
    try:
        # Load tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.1-8B",
            token=token,
            trust_remote_code=True
        )
        
        # Set padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("✓ Tokenizer loaded successfully")
        
        # Load model on CPU to avoid MPS issues
        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.1-8B",
            token=token,
            torch_dtype=torch.float32,  # Use float32 for CPU
            device_map=None,  # Force CPU
            trust_remote_code=True,
            use_cache=True
        )
        print("✓ Model loaded successfully")
        
        # Test query
        query = "What is life cycle assessment and how can it be integrated into engineering education?"
        print(f"\n📝 Testing: {query}")
        print("-" * 40)
        
        # Format prompt - simpler approach
        prompt = f"""<|system|>
You are an expert in sustainability education and engineering curriculum development.

<|user|>
{query}

<|assistant|>
Life cycle assessment (LCA) is a systematic approach to evaluating the environmental impacts of a product, process, or service throughout its entire life cycle. Here's how it can be integrated into engineering education:"""
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # Generate
        print("Generating response...")
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_new_tokens=256,
                temperature=0.3,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                attention_mask=inputs.get("attention_mask", None)
            )
        
        # Decode response
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"🔍 Full response: {full_response}")
        
        # Extract just the assistant's response
        if "<|assistant|>" in full_response:
            response = full_response.split("<|assistant|>")[-1].strip()
        else:
            response = full_response
        
        # Count words
        word_count = len(response.split())
        
        print(f"✅ Generated {word_count} words")
        print(f"💬 Extracted response: {response}")
        
        if word_count > 0:
            print("\n🎉 SUCCESS! Llama 3.1 8B is working perfectly!")
            print("✅ Model can generate curriculum responses")
            print("✅ Ready for integration into your system")
        else:
            print("\n⚠️  Response was empty - may need to adjust parameters")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_single_query() 