#!/usr/bin/env python3
"""
Llama 3.1 Test using Transformers Pipeline
Modern approach using pipeline abstraction
"""

import os
import sys
from transformers import pipeline
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

def test_llama_pipeline():
    """Test Llama 3.1 8B using the modern pipeline approach."""
    print("🚀 Testing Llama 3.1 8B with Pipeline")
    print("=" * 50)
    
    # Get token
    token = os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        print("❌ No HUGGINGFACE_TOKEN found")
        return
    
    print(f"✓ Using token: {token[:10]}...")
    
    try:
        # Create text-generation pipeline
        print("Loading Llama 3.1 8B pipeline...")
        pipe = pipeline(
            "text-generation",
            model="meta-llama/Llama-3.1-8B",
            token=token,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        print("✓ Pipeline loaded successfully")
        
        # Test query
        query = "What is life cycle assessment and how can it be integrated into engineering education?"
        print(f"\n📝 Testing: {query}")
        print("-" * 40)
        
        # Use a simpler prompt format
        prompt = f"""You are an expert in sustainability education and engineering curriculum development.

Question: {query}

Answer:"""
        
        # Generate response
        print("Generating response...")
        outputs = pipe(
            prompt,
            max_new_tokens=512,
            temperature=0.3,
            top_p=0.9,
            do_sample=True
        )
        
        # Extract response
        response = outputs[0]['generated_text']
        
        print(f"🔍 Full response: {response}")
        
        # Extract just the answer part
        if "Answer:" in response:
            answer = response.split("Answer:")[-1].strip()
        else:
            answer = response
        
        # Count words
        word_count = len(answer.split())
        
        print(f"✅ Generated {word_count} words")
        print(f"💬 Answer: {answer}")
        
        if word_count > 0:
            print("\n🎉 SUCCESS! Llama 3.1 8B Pipeline is working perfectly!")
            print("✅ Model can generate curriculum responses")
            print("✅ Ready for integration into your system")
        else:
            print("\n⚠️  Response was empty - may need to adjust parameters")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_llama_pipeline() 