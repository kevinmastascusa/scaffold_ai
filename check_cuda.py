#!/usr/bin/env python3
"""
Check CUDA availability and device information
"""

import torch

print("🔍 Checking CUDA availability...")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name()}")
    print(f"Device memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("❌ CUDA not available - using CPU")
    print("This will be much slower for model inference")

print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}") 