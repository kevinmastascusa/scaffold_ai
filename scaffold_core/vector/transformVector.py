import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import sys
import unicodedata
import re

# Add the parent directory to sys.path to import from main.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import central configuration
from scaffold_core.config import (
    VECTOR_OUTPUTS_DIR, ITERATION, EMBEDDING_MODEL_NAME,
    CHUNKED_TEXT_EXTRACTS_JSON, get_faiss_index_path, get_metadata_json_path
)

# Load precomputed chunks from chunk test output
CHUNKED_JSON_PATH = str(CHUNKED_TEXT_EXTRACTS_JSON)

# Get dynamic paths for this iteration
FAISS_INDEX_PATH = str(get_faiss_index_path(ITERATION))
METADATA_JSON_PATH = str(get_metadata_json_path(ITERATION))

# Clean and normalize chunk text for vectorization
def clean_for_vector(text):
    """
    Normalize to NFC, remove control and zero-width chars, collapse whitespace.
    """
    text = unicodedata.normalize("NFC", text)
    # Remove control characters and zero-width spaces
    text = "".join(ch for ch in text
                   if unicodedata.category(ch)[0] != "C"
                   and ch not in ["\u200B", "\u200C", "\u200D"])
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

def main():

    with open(CHUNKED_JSON_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    # Pre-clean texts for vectorization
    texts    = [clean_for_vector(c["text"]) for c in chunks]
    metadata = [{k:v for k,v in c.items() if k != "text"} for c in chunks]

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    # Add embeddings to FAISS index; specify number of vectors for type-checkers
    index.add(embeddings, embeddings.shape[0])

    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(METADATA_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Built FAISS index ({len(texts)} vectors) → {FAISS_INDEX_PATH}")
    print(f"Saved metadata                     → {METADATA_JSON_PATH}")

if __name__ == "__main__":
    main()
