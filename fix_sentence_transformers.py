#!/usr/bin/env python3
"""
Fixed Sentence Transformers Vector Index Creation
Addresses segmentation fault issues by forcing CPU usage and better memory management.
"""

import os
import sys
import json
import numpy as np
import faiss
import gc
from pathlib import Path
import logging
from typing import List, Dict, Any
import unicodedata
import re

# Force CPU usage to avoid MPS issues
os.environ['PYTORCH_MPS_AVAILABLE'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

from scaffold_core.config import (
    VECTOR_OUTPUTS_DIR, 
    SELECTED_EMBEDDING_MODEL,
    get_faiss_index_path,
    get_metadata_json_path,
    ITERATION
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_sentence_transformers_safely():
    """Load sentence transformers with safe settings."""
    try:
        # Import after setting environment variables
        from sentence_transformers import SentenceTransformer
        import torch
        
        # Force CPU usage
        torch.set_num_threads(1)  # Limit threading
        
        logger.info(f"Loading embedding model: {SELECTED_EMBEDDING_MODEL}")
        
        # Load model with explicit CPU device
        model = SentenceTransformer(SELECTED_EMBEDDING_MODEL, device='cpu')
        
        # Set conservative settings
        model.max_seq_length = 256  # Reduced sequence length
        
        logger.info("✓ Model loaded successfully on CPU")
        return model
        
    except Exception as e:
        logger.error(f"Failed to load sentence transformers: {e}")
        raise

def clean_text_for_embedding(text: str) -> str:
    """Clean text for embedding with conservative approach."""
    if not text or not isinstance(text, str):
        return ""
    
    # Basic Unicode normalization
    text = unicodedata.normalize('NFKC', text)
    
    # Remove problematic characters
    text = ''.join(char for char in text if ord(char) < 65536)  # Remove high Unicode
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # Limit length to avoid memory issues
    if len(text) > 512:
        text = text[:512]
    
    return text

def create_embeddings_conservative(chunks: List[Dict[str, Any]], model) -> tuple:
    """Create embeddings with very conservative memory usage."""
    logger.info(f"Processing {len(chunks)} chunks with conservative approach")
    
    all_embeddings = []
    all_metadata = []
    
    # Process one chunk at a time to avoid memory issues
    for i, chunk in enumerate(chunks):
        if i % 100 == 0:
            logger.info(f"Processing chunk {i+1}/{len(chunks)}")
            gc.collect()  # Force garbage collection
        
        # Clean text
        text = clean_text_for_embedding(chunk.get("text", ""))
        
        if len(text) < 10:  # Skip very short texts
            continue
        
        try:
            # Generate embedding for single text
            embedding = model.encode([text], convert_to_tensor=False, show_progress_bar=False)
            embedding = np.array(embedding, dtype=np.float32)
            
            all_embeddings.append(embedding[0])  # Take first (and only) embedding
            
            # Prepare metadata
            metadata = {k: v for k, v in chunk.items() if k != "text"}
            metadata["text"] = text
            metadata["original_index"] = i
            all_metadata.append(metadata)
            
        except Exception as e:
            logger.warning(f"Failed to process chunk {i}: {e}")
            continue
    
    if not all_embeddings:
        raise ValueError("No embeddings were generated")
    
    # Stack all embeddings
    final_embeddings = np.stack(all_embeddings)
    logger.info(f"✓ Generated {len(final_embeddings)} embeddings with shape: {final_embeddings.shape}")
    
    return final_embeddings, all_metadata

def create_simple_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """Create a simple FAISS index."""
    logger.info("Creating FAISS index...")
    
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)  # Simple L2 distance index
    index.add(embeddings)
    
    logger.info(f"✓ FAISS index created with {index.ntotal} vectors")
    return index

def save_results(index: faiss.Index, metadata: List[Dict], iteration: int = 1) -> None:
    """Save the FAISS index and metadata."""
    # Ensure output directory exists
    VECTOR_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get file paths
    index_path = get_faiss_index_path(iteration)
    metadata_path = get_metadata_json_path(iteration)
    
    logger.info("Saving files...")
    
    # Save FAISS index
    faiss.write_index(index, str(index_path))
    logger.info(f"✓ FAISS index saved to: {index_path}")
    
    # Save metadata
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"✓ Metadata saved to: {metadata_path}")

def load_chunks() -> List[Dict[str, Any]]:
    """Load processed chunks."""
    possible_paths = [
        Path("vector_outputs/processed_1.json"),
        Path("outputs/chunked_text_extracts_cleaned.json"),
        Path("outputs/chunked_text_extracts.json")
    ]
    
    for path in possible_paths:
        if path.exists():
            logger.info(f"Loading chunks from: {path}")
            with open(path, "r", encoding="utf-8") as f:
                chunks = json.load(f)
            logger.info(f"✓ Loaded {len(chunks)} chunks")
            return chunks
    
    raise FileNotFoundError("Could not find processed chunks file")

def main():
    """Main function with conservative approach."""
    try:
        logger.info("🚀 Starting Fixed Vector Index Creation")
        logger.info("=" * 50)
        
        # Load chunks
        chunks = load_chunks()
        
        # Load model safely
        model = load_sentence_transformers_safely()
        
        # Create embeddings conservatively
        embeddings, metadata = create_embeddings_conservative(chunks, model)
        
        # Create index
        index = create_simple_faiss_index(embeddings)
        
        # Save results
        save_results(index, metadata, ITERATION)
        
        logger.info("=" * 50)
        logger.info("✅ Vector index creation completed successfully!")
        logger.info(f"📊 Total vectors: {index.ntotal}")
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()