# Model Summary and Selection

**Last Updated:** July 9, 2025  
**Status:** Updated for Model Registry, Benchmarking, and Mixtral Support

---

## Model Selection Interface

Model selection is now centralized in `scaffold_core/config.py` using registries for each model type:
- **Embedding Models:** `EMBEDDING_MODELS` (MiniLM, MPNet, DistilUSE, etc.)
- **Cross-Encoder Models:** `CROSS_ENCODER_MODELS` (MiniLM, MPNet, etc.)
- **LLM Models:** `LLM_MODELS` (Mistral, Mixtral, TinyLlama, etc.)

Switch models by changing the `SELECTED_*_MODEL` variables in `config.py`.

### Example (config.py):
```python
SELECTED_EMBEDDING_MODEL = EMBEDDING_MODELS["miniLM"]["name"]
SELECTED_CROSS_ENCODER_MODEL = CROSS_ENCODER_MODELS["miniLM"]["name"]
SELECTED_LLM_MODEL = LLM_MODELS["mistral"]["name"]
```

## Available Models

| Type           | Key        | Model Name                                   | Description                                 |
|----------------|------------|----------------------------------------------|---------------------------------------------|
| Embedding      | miniLM     | all-MiniLM-L6-v2                             | Recommended: Fast, high-quality             |
| Embedding      | mpnet      | all-mpnet-base-v2                            | Larger, higher quality, slower              |
| Embedding      | distiluse  | distiluse-base-multilingual-cased-v2         | Multilingual support                        |
| Cross-Encoder  | miniLM     | cross-encoder/ms-marco-MiniLM-L-6-v2         | Recommended: Fast, accurate reranker        |
| Cross-Encoder  | mpnet      | cross-encoder/ms-marco-MiniLM-L-12-v2        | Larger, more accurate, slower               |
| LLM            | mistral    | mistralai/Mistral-7B-Instruct-v0.2           | Recommended: Good balance                   |
| LLM            | mixtral    | mistralai/Mixtral-8x7B-Instruct-v0.1         | Larger, higher quality, more resources      |
| LLM            | tinyllama  | TinyLlama/TinyLlama-1.1B-Chat-v1.0           | Very fast, low resource, lower quality      |

## Model Version/Hash Logging

- All model names, descriptions, and (if available) model card hashes are logged to `outputs/model_version_log.json` using `scaffold_core/model_logging.py`.
- To log all model versions/hashes:
  ```bash
  python -m scaffold_core.model_logging
  ```
- This supports reproducibility and experiment tracking.

## Benchmarking

- Use `scaffold_core/benchmark_models.py` to benchmark all registered models for latency, memory, and output.
- Example usage:
  ```bash
  python -m scaffold_core.benchmark_models
  ```
- Outputs summary tables for all embedding, cross-encoder, and LLM models.

## Model Switching & Troubleshooting
- To switch models, edit the `SELECTED_*_MODEL` variables in `config.py`.
- For Mixtral, the API key is set in `MIXTRAL_API_KEY` (see config.py).
- If a model fails to load, check the Hugging Face token and model compatibility.

## Migration Guide
See `documentation/huggingface_migration_guide.md` for details on switching models, troubleshooting, and best practices.