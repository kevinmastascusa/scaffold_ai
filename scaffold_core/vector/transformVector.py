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
    VECTOR_OUTPUTS_DIR, ITERATION, EMBEDDING_MODEL,
    CHUNKED_TEXT_EXTRACTS_JSON, get_faiss_index_path, get_metadata_json_path
)

# Use the cleaned version of the chunked text extracts
CHUNKED_JSON_PATH = str(CHUNKED_TEXT_EXTRACTS_JSON).replace('.json', '_cleaned.json')

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
    print("Starting vectorization process...")
    print(f"Using cleaned chunks from: {CHUNKED_JSON_PATH}")
    
    # Check if cleaned file exists
    if not os.path.exists(CHUNKED_JSON_PATH):
        print(f"❌ Error: Cleaned chunks file not found at {CHUNKED_JSON_PATH}")
        print("Please run the post-processing script first:")
        print("python scaffold_core/scripts/postprocess_combined_words.py")
        return
    
    print("Loading cleaned chunks...")
    with open(CHUNKED_JSON_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    print(f"✓ Loaded {len(chunks)} cleaned chunks")

    # Pre-clean texts for vectorization (additional Unicode normalization)
    print("Applying final Unicode cleaning for vectorization...")
    texts = []
    metadata = []
    
    for i, chunk in enumerate(chunks):
        if i % 1000 == 0:
            print(f"  Processing chunk {i+1}/{len(chunks)}...")
        
        cleaned_text = clean_for_vector(chunk["text"])
        texts.append(cleaned_text)
        
        # Include all metadata AND the cleaned text
        chunk_metadata = {k: v for k, v in chunk.items() if k != "text"}
        chunk_metadata["text"] = cleaned_text  # Add the cleaned text back
        metadata.append(chunk_metadata)

    print(f"✓ Prepared {len(texts)} texts for vectorization")
    
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
