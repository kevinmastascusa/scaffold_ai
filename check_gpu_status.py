#!/usr/bin/env python3
"""
GPU Status Checker for Scaffold AI
Quick utility to verify GPU availability and optimization settings.
"""

import torch
import onnxruntime as ort
from pathlib import Path
import sys

def check_gpu_status():
    """Check and display GPU status and optimization settings."""
    print("🔍 Scaffold AI GPU Status Check")
    print("=" * 50)
    
    # Check PyTorch CUDA
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Count: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        print(f"GPU Memory Available: {torch.cuda.memory_allocated(0) / 1e9:.1f}GB used")
        print("✅ PyTorch GPU: READY")
    else:
        print("❌ PyTorch GPU: NOT AVAILABLE")
    
    print()
    
    # Check ONNX Runtime
    print("ONNX Runtime Providers:")
    providers = ort.get_available_providers()
    for provider in providers:
        print(f"  - {provider}")
    
    if "CUDAExecutionProvider" in providers:
        print("✅ ONNX Runtime GPU: READY")
    else:
        print("❌ ONNX Runtime GPU: NOT AVAILABLE")
    
    print()
    
    # Check ONNX cache
    onnx_cache = Path("outputs/onnx_models/TinyLlama__TinyLlama-1.1B-Chat-v1.0")
    if onnx_cache.exists():
        print(f"✅ ONNX Cache: EXISTS ({onnx_cache})")
        cache_files = list(onnx_cache.glob("*"))
        print(f"   Cache files: {len(cache_files)} files")
    else:
        print("❌ ONNX Cache: NOT FOUND (will be created on first run)")
    
    print()
    
    # Recommendations
    print("💡 Recommendations:")
    if torch.cuda.is_available() and "CUDAExecutionProvider" in providers:
        print("  - GPU is ready! Use 'run_gpu_optimized.bat' for best performance")
        print("  - Model should load in ~10-15 seconds")
        print("  - Response generation should be very fast")
    elif torch.cuda.is_available():
        print("  - PyTorch GPU available but ONNX GPU not ready")
        print("  - Install onnxruntime-gpu: pip install onnxruntime-gpu==1.22.0")
    else:
        print("  - No GPU detected, using CPU optimizations")
        print("  - Consider installing PyTorch with CUDA support")
    
    print("=" * 50)

if __name__ == "__main__":
    check_gpu_status()
