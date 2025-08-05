#!/usr/bin/env python3
"""
Script to help set up Hugging Face token for the project.
"""

import os
from pathlib import Path

def setup_hf_token():
    """Set up Hugging Face token for the project."""
    print("🔑 Hugging Face Token Setup")
    print("=" * 40)
    
    # Check if token is already set
    current_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
    if current_token:
        print(f"✅ Token already set: {current_token[:8]}...")
        return current_token
    
    print("❌ No Hugging Face token found in environment variables.")
    print("\nTo fix this, you have several options:")
    print("\n1. 🌐 Get a token from https://huggingface.co/settings/tokens")
    print("   - Create a 'Read' token (free)")
    print("   - Copy the token")
    
    print("\n2. 🔧 Set the token in your environment:")
    print("   Option A - Temporary (current session only):")
    print("   export HUGGINGFACE_TOKEN='your_token_here'")
    
    print("\n   Option B - Permanent (add to ~/.bashrc or ~/.zshrc):")
    print("   echo 'export HUGGINGFACE_TOKEN=\"your_token_here\"' >> ~/.bashrc")
    
    print("\n   Option C - Create a .env file in this directory:")
    print("   echo 'HUGGINGFACE_TOKEN=your_token_here' > .env")
    
    print("\n3. 🤖 For TinyLlama (no token needed):")
    print("   We can use a public model that doesn't require authentication")
    
    # Try to use a model that doesn't require authentication
    print("\n🔄 Testing with public models...")
    try:
        # Try importing without token
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        print("✅ Can access some models without token")
        return "public_access"
    except Exception as e:
        print(f"⚠️  Public access limited: {e}")
    
    return None

def suggest_alternative_models():
    """Suggest models that don't require tokens."""
    print("\n🤖 Alternative Models (No Token Required):")
    print("=" * 50)
    
    models = [
        {
            "name": "microsoft/DialoGPT-medium",
            "desc": "Conversational model, good for chat",
            "size": "Medium (345M parameters)"
        },
        {
            "name": "distilgpt2",
            "desc": "Lightweight GPT-2 variant",
            "size": "Small (82M parameters)"
        },
        {
            "name": "gpt2",
            "desc": "Original GPT-2 model",
            "size": "Medium (124M parameters)"
        }
    ]
    
    for i, model in enumerate(models, 1):
        print(f"{i}. {model['name']}")
        print(f"   Description: {model['desc']}")
        print(f"   Size: {model['size']}")
        print()

def main():
    token = setup_hf_token()
    
    if not token:
        suggest_alternative_models()
        
        print("\n💡 Quick Fix Options:")
        print("1. Set token and restart the UI")
        print("2. Use alternative models listed above")
        print("3. Run: export HUGGINGFACE_TOKEN='your_token' && cd frontend && python start_enhanced_ui.py")

if __name__ == "__main__":
    main()