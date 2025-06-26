# Import central configuration
from scaffold_core.config import (
    DATA_DIR, VECTOR_OUTPUTS_DIR, ITERATION, CHUNK_SIZE,
    EMBEDDING_MODEL_NAME, OLLAMA_MODEL, OLLAMA_ENDPOINT,
    PROCESSED_JSON_PATH, get_faiss_index_path, get_metadata_json_path
)

# Paths are now defined in config.py
PDF_INPUT_DIR = str(DATA_DIR)
OUTPUT_DIR = str(VECTOR_OUTPUTS_DIR)

# Get dynamic paths for this iteration
FAISS_INDEX_PATH = str(get_faiss_index_path(ITERATION))
METADATA_JSON_PATH = str(get_metadata_json_path(ITERATION))

# Query LLM reranking settings
# (OLLAMA_MODEL and OLLAMA_ENDPOINT are now imported from config)

# TODO: Fix words that are combined in chunked_text_extract.json (e.g., "environmentalsustainability" should be "environmental sustainability")
