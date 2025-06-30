# Local Setup Guide

This guide explains how to set up the local components required for the Scaffold AI pipeline, specifically the Hugging Face integration for LLM functionality.

## Prerequisites

- Python 3.11+ installed
- Git repository cloned
- 16GB+ RAM recommended
- NVIDIA GPU recommended but not required

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

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## LLM Configuration

The project uses Hugging Face's Transformers library with the Mistral-7B-Instruct model. The configuration is defined in `scaffold_core/config.py`:

```python
# LLM configuration
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"  # Hugging Face model ID
LLM_TASK = "text-generation"
LLM_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LLM_MAX_LENGTH = 2048
LLM_TEMPERATURE = 0.7
LLM_TOP_P = 0.95
```

### GPU Acceleration

If you have an NVIDIA GPU:
1. The model will automatically use GPU if available
2. Mixed precision (FP16) is enabled by default for better memory efficiency
3. The model uses the 'auto' device map for optimal GPU memory usage

If no GPU is available, the model will run on CPU with FP32 precision.

### Changing Models

To use a different model:

1. Update `LLM_MODEL` in `config.py` to any Hugging Face model ID
2. Ensure the model is compatible with text generation
3. Adjust generation parameters as needed:
   - `LLM_MAX_LENGTH`: Maximum length of generated responses
   - `LLM_TEMPERATURE`: Controls randomness (0.0 to 1.0)
   - `LLM_TOP_P`: Controls nucleus sampling

Example alternative models:
- `meta-llama/Llama-2-7b-chat-hf` (requires Hugging Face access token)
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (smaller model, faster inference)
- `microsoft/phi-2` (good balance of size and performance)

## Using the LLM

The LLM functionality is implemented in `scaffold_core/llm.py`. Here's how to use it:

```python
from scaffold_core.llm import llm

# Single response generation
response = llm.generate_response(
    "What are the key principles of sustainability?",
    temperature=0.7  # Optional: override default temperature
)

# Batch generation
prompts = [
    "What is climate resilience?",
    "How does sustainability affect education?"
]
responses = llm.batch_generate(prompts)
```

## Environment Variables

For some models (like Llama 2), you'll need to set up a Hugging Face access token:

1. Get your token from https://huggingface.co/settings/tokens
2. Create a `.env` file in your project root:
   ```bash
   HUGGINGFACE_TOKEN=your_token_here
   ```

## Troubleshooting

### Common Issues

1. **Out of Memory (OOM) errors**
   - Try a smaller model
   - Reduce batch size
   - Enable gradient checkpointing
   - Use CPU if GPU memory is insufficient

2. **Slow inference on CPU**
   - Consider using a smaller model
   - Reduce context length
   - Use quantized models (8-bit or 4-bit)

3. **Model download issues**
   - Check your internet connection
   - Verify Hugging Face token if using gated models
   - Try downloading the model manually

### Performance Tips

1. **For faster inference:**
   - Use GPU when available
   - Keep prompts concise
   - Use appropriate `max_length`
   - Consider quantized models

2. **For better quality:**
   - Adjust temperature and top_p
   - Use well-structured prompts
   - Consider larger models if resources allow

## Next Steps

1. Run the text processing pipeline
2. Test the complete workflow
3. Adjust model parameters as needed

## Support

If you encounter issues:
1. Check the [Hugging Face documentation](https://huggingface.co/docs)
2. Review model-specific documentation
3. Check the project's issue tracker
4. Ensure all prerequisites are met 