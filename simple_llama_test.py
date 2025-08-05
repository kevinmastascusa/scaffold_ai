"""
Test script for running optimized models with ONNX Runtime.
This provides significantly faster inference on CPU devices.
"""

import os
import time
import torch
import logging
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForCausalLM

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Model selection - using a smaller model that works well with ONNX
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Small model that works well with ONNX
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

def run_onnx_inference():
    """Run inference using ONNX Runtime for optimized performance."""
    
    start_time = time.time()
    logger.info(f"Loading model: {MODEL_ID}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    # Load model with ONNX Runtime optimization
    model = ORTModelForCausalLM.from_pretrained(
        MODEL_ID,
        export=True,  # Export to ONNX format
        provider="CPUExecutionProvider",  # Use CPU provider
    )
    
    load_time = time.time() - start_time
    logger.info(f"Model loaded in {load_time:.2f} seconds")
    
    # Prepare prompt
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Explain quantum computing in simple terms."}
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Run inference
    logger.info("Starting inference...")
    inference_start = time.time()
    
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Generate with ONNX Runtime
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
    
    inference_time = time.time() - inference_start
    
    # Decode output
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the assistant's response
    if "assistant" in response.lower():
        response = response.split("assistant")[-1].strip()
    
    logger.info(f"Inference completed in {inference_time:.2f} seconds")
    logger.info(f"\nResponse:\n{response}")
    
    return {
        "load_time": load_time,
        "inference_time": inference_time,
        "total_time": load_time + inference_time,
        "response": response
    }

if __name__ == "__main__":
    print("=" * 50)
    print("ONNX Runtime Optimized Inference Test")
    print("=" * 50)
    
    results = run_onnx_inference()
    
    print("\nPerformance Summary:")
    print(f"Model load time: {results['load_time']:.2f} seconds")
    print(f"Inference time: {results['inference_time']:.2f} seconds")
    print(f"Total time: {results['total_time']:.2f} seconds")