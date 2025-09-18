# Local Setup Guide - Running Models Locally

This guide explains how to set up and run all models locally for the Scaffold AI pipeline, including embedding models, cross-encoders, and LLMs with GPU acceleration.

## Overview

The system now runs **all models locally** instead of using API calls:
- **Embedding Model**: `all-MiniLM-L6-v2` for semantic search
- **Cross-Encoder**: `cross-encoder/ms-marco-MiniLM-L-6-v2` for reranking
- **LLM**: `mistralai/Mistral-7B-Instruct-v0.2` for response generation

## Prerequisites

- Python 3.11+ installed
- Git repository cloned
- **16GB+ RAM recommended** (8GB minimum)
- **NVIDIA GPU with 8GB+ VRAM recommended** for optimal performance
- CUDA toolkit installed (for GPU acceleration)

## Setting Up the Environment

1. **Create and activate virtual environment:**
   ```bash
   # Windows (PowerShell)
   python -m venv scaffold_env
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
   .\scaffold_env\Scripts\activate

   # macOS/Linux
   python -m venv scaffold_env
   source scaffold_env/bin/activate
   ```

2. **Install PyTorch with CUDA support (for GPU acceleration):**
   ```bash
   # For CUDA 11.8 (most common)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   
   # For CUDA 12.1
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

3. **Install remaining dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Local Model Configuration

All model configurations are defined in `scaffold_core/config.py`:

### Embedding Model
```python
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 384-dimensional embeddings
```
- **Purpose**: Converts text to vector embeddings for semantic search
- **Memory**: ~90MB model size
- **Performance**: Very fast inference on CPU/GPU

### Cross-Encoder Model
```python
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
```
- **Purpose**: Reranks search results for better relevance
- **Memory**: ~90MB model size
- **Performance**: Processes query-document pairs for scoring

### LLM Configuration
```python
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"  # 7B parameter model
LLM_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LLM_MAX_LENGTH = 2048  # Response length limit
LLM_TEMPERATURE = 0.3  # Lower = more focused responses
LLM_TOP_P = 0.9  # Nucleus sampling parameter
LLM_LOAD_IN_8BIT = False  # Quantization (disabled on Windows)
LLM_LOAD_IN_4BIT = False  # More aggressive quantization
```

## GPU Acceleration Setup

### CUDA Installation
1. **Check GPU compatibility:**
   ```bash
   nvidia-smi  # Should show your GPU details
   ```

2. **Verify PyTorch CUDA:**
   ```python
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"CUDA device: {torch.cuda.get_device_name(0)}")
   ```

### GPU Memory Requirements
- **Minimum**: 8GB VRAM (with quantization)
- **Recommended**: 11GB+ VRAM (full precision)
- **Tested on**: RTX 2080 Ti (11GB) - works excellently

### Performance Optimizations
- **Float16 precision**: Automatically enabled on GPU for memory efficiency
- **Device mapping**: Models use 'auto' device mapping for optimal GPU utilization
- **Batch processing**: Configured for efficient GPU utilization

## Model Loading and Caching

### First Run Behavior
On first run, models will be downloaded and cached locally:
```
~/.cache/huggingface/transformers/  # Model files
~/.cache/huggingface/datasets/      # Tokenizer files
```

### Model Sizes
- **Embedding model**: ~90MB
- **Cross-encoder**: ~90MB  
- **Mistral-7B**: ~13GB (full precision) / ~7GB (8-bit) / ~3.5GB (4-bit)

## Quantization Options

### 8-bit Quantization (Recommended for 8GB VRAM)
```python
LLM_LOAD_IN_8BIT = True
```
- Reduces memory usage by ~50%
- Minimal quality loss
- **Note**: Currently disabled on Windows due to bitsandbytes compatibility

### 4-bit Quantization (For limited VRAM)
```python
LLM_LOAD_IN_4BIT = True
```
- Reduces memory usage by ~75%
- Some quality degradation
- Fastest inference

## Using the Local Models

### Basic Usage
```python
from scaffold_core.vector.enhanced_query import EnhancedQuerySystem

# Initialize the system (loads all models locally)
query_system = EnhancedQuerySystem()

# Run a query (uses all local models)
results = query_system.enhanced_search(
    "What is life cycle assessment?",
    top_k=5
)
```

### Model Loading Process
1. **Embedding model** loads first (~2-3 seconds)
2. **Cross-encoder** loads next (~2-3 seconds)
3. **LLM** loads last (~10-30 seconds depending on GPU/quantization)

## Performance Benchmarks

### GPU Performance (RTX 2080 Ti)
- **Model loading**: ~15 seconds total
- **Query processing**: ~30-60 seconds per query
- **Memory usage**: ~8GB VRAM (full precision)

### CPU Performance
- **Model loading**: ~30-60 seconds total  
- **Query processing**: ~2-5 minutes per query
- **Memory usage**: ~12-16GB RAM

## Alternative Models

### Smaller LLMs (for limited resources)
```python
# Faster, smaller models
LLM_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # 1.1B params
LLM_MODEL = "microsoft/phi-2"                      # 2.7B params
```

### Larger LLMs (for better quality)
```python
# Requires HF token and more VRAM
LLM_MODEL = "meta-llama/Llama-2-7b-chat-hf"      # 7B params
LLM_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1" # 8x7B params
```

## Environment Variables

For gated models (like Llama 2), set up a Hugging Face token:

1. Get your token from https://huggingface.co/settings/tokens
2. Create a `.env` file in your project root:
   ```bash
   HUGGINGFACE_TOKEN=your_token_here
   ```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```
   Solution: Enable quantization or use smaller model
   LLM_LOAD_IN_8BIT = True  # in config.py
   ```

2. **Slow CPU Inference**
   ```
   Solution: Use smaller model or enable GPU
   LLM_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
   ```

3. **Model Download Failures**
   ```
   Solution: Check internet connection and HF token
   - Verify token permissions
   - Try manual download: huggingface-cli download model_name
   ```

4. **Windows Quantization Issues**
   ```
   Current: Quantization disabled on Windows
   Workaround: Use full precision or switch to Linux/WSL
   ```

### Performance Tips

1. **For faster inference:**
   - Use GPU when available
   - Enable appropriate quantization
   - Use smaller models for development
   - Keep batch sizes small

2. **For better quality:**
   - Use full precision models
   - Adjust temperature (0.1-0.7)
   - Use larger models when resources allow
   - Fine-tune generation parameters

## System Status

✅ **Currently Working:**
- All models running locally
- GPU acceleration on CUDA-compatible devices
- Hybrid search with local reranking
- Complete pipeline from search to response generation

🔧 **Known Limitations:**
- Quantization disabled on Windows
- First model loading takes time
- Large VRAM requirements for full precision

## Next Steps

1. Run the enhanced query system: `python generate_query_responses.py`
2. Test with your own queries
3. Adjust model parameters as needed
4. Monitor GPU/CPU usage and optimize accordingly

## Support

If you encounter issues:
1. Check GPU compatibility and CUDA installation
2. Verify model downloads in cache directory
3. Monitor system resources during model loading
4. Review configuration in `config.py`
5. Check the project's issue tracker 