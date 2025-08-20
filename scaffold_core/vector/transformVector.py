import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import sys
import unicodedata
import re
from typing import List, Dict, Any

# Ensure project root is on sys.path so 'scaffold_core' package imports work
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Import central configuration
from scaffold_core.config import (
    VECTOR_OUTPUTS_DIR, ITERATION, EMBEDDING_MODEL,
    CHUNKED_TEXT_EXTRACTS_JSON, get_faiss_index_path, get_metadata_json_path
)

# Use the cleaned version of the chunked text extracts
CHUNKED_JSON_PATH = str(CHUNKED_TEXT_EXTRACTS_JSON).replace('.json', '_cleaned.json')

# Get dynamic paths for this iteration
FAISS_INDEX_PATH = str(get_faiss_index_path(ITERATION))
METADATA_JSON_PATH = str(get_metadata_json_path(ITERATION))

# Optional exclusion patterns to drop off-topic/template chunks during (re)build
def _load_exclude_patterns() -> List[str]:
    env_val = os.getenv("SC_EXCLUDE_PATTERNS", "").strip()
    # Default to template-like content only; do NOT exclude domain terms by default
    defaults = [
        "essay", "template", "rubric", "guide", "format"
    ]
    if not env_val:
        return defaults
    try:
        parts = [p.strip() for p in env_val.split(",") if p.strip()]
        return parts or defaults
    except Exception:
        return defaults

EXCLUDE_PATTERNS = [p.lower() for p in _load_exclude_patterns()]

def _should_exclude_chunk(chunk: Dict[str, Any]) -> bool:
    try:
        txt = (chunk.get("text") or "").lower()
        meta = chunk.get("metadata") or {}
        fname = str(meta.get("filename", "")).lower()
        haystack = " ".join([txt, fname])
        return any(pat in haystack for pat in EXCLUDE_PATTERNS)
    except Exception:
        return False

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
    print("Starting vectorization process...")
    print(f"Using cleaned chunks from: {CHUNKED_JSON_PATH}")
    
    # Check if cleaned file exists; fall back to the original if not
    if not os.path.exists(CHUNKED_JSON_PATH):
        fallback_path = str(CHUNKED_TEXT_EXTRACTS_JSON)
        if os.path.exists(fallback_path):
            print(f"⚠️  Cleaned chunks not found at {CHUNKED_JSON_PATH}. Falling back to {fallback_path}")
            CHUNKED = fallback_path
        else:
            print(f"❌ Error: No chunks file found at {CHUNKED_JSON_PATH} or {fallback_path}")
            return
    else:
        CHUNKED = CHUNKED_JSON_PATH
    
    print("Loading chunks...")
    with open(CHUNKED, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    print(f"✓ Loaded {len(chunks)} cleaned chunks")

    # Pre-clean texts for vectorization (additional Unicode normalization)
    print("Applying final Unicode cleaning for vectorization...")
    texts = []
    metadata = []
    
    kept = 0
    for i, chunk in enumerate(chunks):
        if i % 1000 == 0:
            print(f"  Processing chunk {i+1}/{len(chunks)}...")
        # Drop off-topic/template chunks
        if _should_exclude_chunk(chunk):
            continue
        
        cleaned_text = clean_for_vector(chunk["text"])
        texts.append(cleaned_text)
        
        # Include all metadata AND the cleaned text
        chunk_metadata = {k: v for k, v in chunk.items() if k != "text"}
        chunk_metadata["text"] = cleaned_text  # Add the cleaned text back
        metadata.append(chunk_metadata)
        kept += 1

    print(f"✓ Prepared {len(texts)} texts for vectorization (kept {kept} / {len(chunks)})")
    
    # Load the embedding model
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    # Generate embeddings
    print("Generating embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')
    
    print(f"✓ Generated embeddings with shape: {embeddings.shape}")

    # Create FAISS index
    print("Creating FAISS index...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    
    print(f"✓ Created FAISS index with {index.ntotal} vectors")

    # Save the index and metadata
    print("Saving files...")
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(METADATA_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"✓ Built FAISS index ({len(texts)} vectors) → {FAISS_INDEX_PATH}")
    print(f"✓ Saved metadata → {METADATA_JSON_PATH}")
    print("🎉 Vectorization completed successfully!")

if __name__ == "__main__":
    main()
