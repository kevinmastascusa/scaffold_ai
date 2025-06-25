# Edit to match local system file paths

# 1) where your raw PDFs (and subfolders) live:
PDF_INPUT_DIR       = r"c:\Users\dlaev\OneDrive\Documents\GitHub\scaffold_ai\data"

# 2) where you want your chunked JSON files to go:
OUTPUT_DIR          = r"c:\Users\dlaev\OneDrive\Documents\GitHub\scaffold_ai\vector_outputs"

# 3) bump this when you want a fresh run:
ITERATION           = 1

# 4) how big, in words, each chunk should be:
CHUNK_SIZE          = 500

# your embedding model identifier
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# transformVector.py will write these files (should automatically be pathed based on output json):
FAISS_INDEX_PATH    = f"{OUTPUT_DIR}/scaffold_index_{ITERATION}.faiss"
METADATA_JSON_PATH  = f"{OUTPUT_DIR}/scaffold_metadata_{ITERATION}.json"

# Point to precomputed chunk JSON instead of regenerating chunks
PROCESSED_JSON_PATH = r"c:\Users\dlaev\OneDrive\Documents\GitHub\scaffold_ai\outputs\chunked_text_extracts.json"

# Query LLM reranking settings

# Ollama model to use for reranking (make sure it's installed locally via ollama)
OLLAMA_MODEL     = "mistral"

#point at the OpenAI‐compatible endpoint
OLLAMA_ENDPOINT  = "http://localhost:11434/v1/chat/completions"

# TODO: Fix words that are combined in chunked_text_extract.json (e.g., "environmentalsustainability" should be "environmental sustainability")
