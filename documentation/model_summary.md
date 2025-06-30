# Model Summary and Selection

**Last Updated:** June 29, 2025  
**Status:** Updated for Hugging Face Integration

This document summarizes the models used in the Scaffold AI project and the rationale for their selection, reflecting the current codebase and configuration.

---

## 1. Sentence Embedding Model

- **Model:** `all-MiniLM-L6-v2`
- **Library:** sentence-transformers
- **Purpose:**
  - Converts text chunks into dense vector embeddings for semantic search.
  - Used in the vectorization pipeline to create the FAISS index for similarity search and retrieval.
- **Location in Codebase:**
  - Used in `scaffold_core/vector/transformVector.py` for embedding generation.
  - Referenced in `scaffold_core/config.py` as `EMBEDDING_MODEL`.

---

## 2. Cross-Encoder Model (Reranking)

- **Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Library:** sentence-transformers (CrossEncoder)
- **Purpose:**
  - Reranks the top candidate chunks retrieved from the FAISS index based on their relevance to the user query.
  - Provides more accurate ranking by considering the query and chunk together.
- **Location in Codebase:**
  - Used in `scaffold_core/vector/query.py` as `CROSS_ENCODER_MODEL`.

---

## 3. Large Language Model (LLM) for Answer Generation

- **Current Implementation (June 29, 2025):**
  - **Model:** `mistralai/Mistral-7B-Instruct-v0.2`
  - **Platform:** Hugging Face Transformers
  - **Integration:** Python API via `scaffold_core/llm.py`
  - **Status:** ✅ Fully tested and operational (100% test success rate)

- **Previously Tested Models:**
  - **TinyLlama/TinyLlama-1.1B-Chat-v1.0** - Smaller, faster alternative
  - **teknium/OpenHermes-2.5-Mistral-7B** - Community Mistral variant
  - **mistralai/Mistral-7B-Instruct-v0.3** - Had tokenizer compatibility issues

- **Purpose:**
  - Generates the final answer to the user's query, using only the retrieved and reranked chunks as context.
  - Ensures answers are grounded in the source material and can provide citations.
  - Uses Mistral's `[INST]...[/INST]` chat format for proper instruction following.

- **Location in Codebase:**
  - **LLM Manager:** `scaffold_core/llm.py` - Handles model loading and text generation
  - **Query Integration:** `scaffold_core/vector/query.py` - Integrates LLM with retrieval pipeline
  - **Configuration:** `scaffold_core/config.py` - Model settings and parameters

---

## Model Selection Rationale

- **Embedding Model:**
  - Chosen for its balance of speed, accuracy, and resource efficiency.
  - `all-MiniLM-L6-v2` is a widely used, lightweight model suitable for large-scale document embedding.

- **Cross-Encoder:**
  - Selected to improve retrieval quality by reranking based on both query and chunk content.
  - `ms-marco-MiniLM-L-6-v2` is a standard for semantic search reranking.

- **LLM:**
  - Open-source models are prioritized for transparency, cost, and local deployment.
  - Mistral-7B-Instruct-v0.2 was selected for its excellent performance and official support.
  - Hugging Face Transformers provides better integration, debugging, and cross-platform compatibility than Ollama.
  - The migration from Ollama to Hugging Face resolved setup issues and improved system reliability.

---

## Summary Table

| Stage                | Model Name                        | Library/Platform      | Purpose                                 |
|----------------------|-----------------------------------|-----------------------|-----------------------------------------|
| Embedding            | all-MiniLM-L6-v2                  | sentence-transformers | Chunk embedding for FAISS search        |
| Reranking            | cross-encoder/ms-marco-MiniLM-L-6-v2 | sentence-transformers | Rerank top retrieved chunks             |
| LLM (Answer Gen)     | mistralai/Mistral-7B-Instruct-v0.2 | Hugging Face Transformers | Generate grounded, cited answers        |

---

## How to Change Models

- **Embedding or Cross-Encoder:**
  - Update the model name in `scaffold_core/config.py` (use `EMBEDDING_MODEL` and `CROSS_ENCODER_MODEL`).
  - The system will automatically download and cache the new models.

- **LLM:**
  - Change the `LLM_MODEL` value in `scaffold_core/config.py`.
  - For gated models, ensure you have proper Hugging Face token access.
  - See `documentation/huggingface_migration_guide.md` for detailed setup instructions.

- **Testing Changes:**
  - Run `python scaffold_core/scripts/run_tests.py` to verify all models work correctly.
  - Generate a test report with `python scaffold_core/scripts/generate_test_report.py`.

---

## Migration Notes (June 29, 2025)

The system was successfully migrated from Ollama to Hugging Face Transformers:
- **Reason:** Ollama setup issues and cross-platform compatibility problems
- **Benefits:** Better Python integration, easier debugging, comprehensive testing
- **Status:** 100% test success rate with all components functional

For more details, see:
- **Code:** `scaffold_core/vector/` and `scaffold_core/llm.py`
- **Configuration:** `scaffold_core/config.py`
- **Migration Guide:** `documentation/huggingface_migration_guide.md`
- **Test Results:** `documentation/query_system_test_report.md` 