# Model Summary and Selection

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
  - Referenced in `scaffold_core/config.py` as `EMBEDDING_MODEL_NAME`.

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

- **Models Under Consideration:**
  - **Llama 3 (Meta):** meta-llama/Llama-3.1-8B-Instruct, Llama-3.2-1B
  - **Mistral-7B (Mistral AI):** Mistral-7B-v0.1, Mistral-7B-Instruct-v0.2, v0.3
  - **Phi-3 Mini (Microsoft):** Phi-3.5-mini-instruct, Phi-3-mini-4k-instruct
- **Current Default:**
  - **Mistral** (via Ollama, as set in `scaffold_core/config.py` with `OLLAMA_MODEL = "mistral"`)
- **Purpose:**
  - Generates the final answer to the user's query, using only the retrieved and reranked chunks as context.
  - Ensures answers are grounded in the source material and can provide citations.
- **Location in Codebase:**
  - Used in `scaffold_core/vector/query.py` in the `rerank_with_ollama` function.
  - Model and endpoint are set in `scaffold_core/config.py`.

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
  - Mistral is set as the default for its strong performance and compatibility with Ollama.
  - Llama 3 and Phi-3 are also considered for future or alternative deployments, depending on licensing, performance, and hardware requirements.

---

## Summary Table

| Stage                | Model Name                        | Library/Platform      | Purpose                                 |
|----------------------|-----------------------------------|-----------------------|-----------------------------------------|
| Embedding            | all-MiniLM-L6-v2                  | sentence-transformers | Chunk embedding for FAISS search        |
| Reranking            | cross-encoder/ms-marco-MiniLM-L-6-v2 | sentence-transformers | Rerank top retrieved chunks             |
| LLM (Answer Gen)     | Mistral (default), Llama 3, Phi-3 | Ollama, open-source   | Generate grounded, cited answers        |

---

## How to Change Models

- **Embedding or Cross-Encoder:**
  - Update the model name in `scaffold_core/config.py` or directly in the relevant script.
- **LLM:**
  - Change the `OLLAMA_MODEL` value in `scaffold_core/config.py`.
  - Ensure the model is available in your Ollama or LLM serving environment.

---

For more details, see the code in `scaffold_core/vector/` and the configuration in `scaffold_core/config.py`. 