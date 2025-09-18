# Model Optimization Guide

This guide explains how to optimize the performance of LLM models in the Scaffold AI system, particularly on machines without GPU acceleration.

## Available Optimization Methods

1. **ONNX Runtime Optimization**
   - Uses the ONNX (Open Neural Network Exchange) format for faster inference
   - Particularly effective on CPU-only machines
   - Works with TinyLlama model

2. **Model Quantization**
   - Reduces model precision to improve speed and memory usage
   - Options include 8-bit and 4-bit quantization

3. **Torch Compile**
   - Compiles PyTorch models for faster execution
   - Enabled by default in the config

4. **Smaller Models**
   - TinyLlama (1.1B parameters) offers much faster performance than larger models
   - Good for development and testing

## Switching Between Models

Use the `switch_model.py` script to easily change which model is being used:

```bash
# List available models
python switch_model.py --list

# Switch to a specific model
python switch_model.py tinyllama-onnx
```

## Benchmarking Models

Use the `benchmark_models.py` script to compare performance between different models:

```bash
# List available models for benchmarking
python benchmark_models.py --list

# Benchmark specific models
python benchmark_models.py --models tinyllama tinyllama-onnx

# Benchmark all available models
python benchmark_models.py --all

# Specify output file
python benchmark_models.py --models tinyllama-onnx --output my_benchmark.json
```

## Performance Recommendations

For the best performance on machines without a GPU:

1. **Best Speed**: Use `tinyllama-onnx` model
   - Fastest option, uses ONNX runtime optimization
   - Good for development and testing

2. **Balance of Speed/Quality**: Use `mistral` model with 4-bit quantization
   - Better quality than TinyLlama with reasonable speed
   - Good for production use cases with moderate requirements

3. **Best Quality**: Use `llama3.1-8b` model
   - Highest quality responses
   - Significantly slower on CPU-only machines

## Installation Requirements

To use ONNX optimization, you need to install:

```bash
pip install onnx onnxruntime optimum
```

## Configuration Options

Key configuration options in `scaffold_core/config.py`:

```python
# Quantization settings
LLM_LOAD_IN_8BIT = False
LLM_LOAD_IN_4BIT = True

# Optimization settings
TORCH_COMPILE = True
USE_ONNX = False  # Automatically set when using ONNX-compatible models

# Generation parameters
LLM_MAX_NEW_TOKENS = 2048  # Default value for Llama 3.1
LLM_TEMPERATURE = 0.3

# Search parameters (lower values = faster but potentially less accurate)
TOP_K_INITIAL = 30
TOP_K_FINAL = 3
```

### Fixing Truncation Issues

If you see "Response appears to be truncated" warnings in the logs:

1. You can increase the `max_new_tokens` parameter when calling `generate_response`
2. Or adjust the `LLM_MAX_NEW_TOKENS` value in `scaffold_core/config.py` (default: 2048)
3. For very long responses, consider increasing `MAX_RESPONSE_WORDS` as well
4. Note that increasing token limits will use more memory and may slow down inference

## Troubleshooting

If you encounter memory issues:
1. Switch to a smaller model (TinyLlama)
2. Enable 4-bit quantization
3. Reduce batch size and context length
4. Close other memory-intensive applications