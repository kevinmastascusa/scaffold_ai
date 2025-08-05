#!/usr/bin/env python3
"""
Robust vector index creation with memory management and error handling.
This script processes all the extracted chunks and creates a FAISS index.
"""

import json
import numpy as np
import faiss
import gc
import logging
from pathlib import Path
import sys
import os

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

from sentence_transformers import SentenceTransformer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clean_for_vector(text):
    """Clean text for vectorization."""
    import unicodedata
    import re
    
    if not text:
        return ""
    
    # Unicode normalization
    text = unicodedata.normalize('NFKC', text)
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

def process_chunks_in_batches(chunks, batch_size=500):
    """Process chunks in smaller batches to avoid memory issues."""
    logger.info(f"Processing {len(chunks)} chunks in batches of {batch_size}")
    
    texts = []
    metadata = []
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        logger.info(f"Processing batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")
        
        for j, chunk in enumerate(batch):
            if (i + j) % 1000 == 0:
                logger.info(f"  Processing chunk {i + j + 1}/{len(chunks)}...")
            
            cleaned_text = clean_for_vector(chunk.get("text", ""))
            if cleaned_text:  # Only add non-empty texts
                texts.append(cleaned_text)
                
                # Include all metadata AND the cleaned text
                chunk_metadata = {k: v for k, v in chunk.items() if k != "text"}
                chunk_metadata["text"] = cleaned_text
                metadata.append(chunk_metadata)
        
        # Force garbage collection after each batch
        gc.collect()
    
    logger.info(f"✓ Prepared {len(texts)} texts for vectorization")
    return texts, metadata

def create_embeddings_in_batches(model, texts, batch_size=100):
    """Create embeddings in smaller batches to avoid memory issues."""
    logger.info(f"Creating embeddings for {len(texts)} texts in batches of {batch_size}")
    
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        logger.info(f"Embedding batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
        
        try:
            # Generate embeddings for this batch
            batch_embeddings = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
            all_embeddings.append(batch_embeddings)
            
            # Force garbage collection
            gc.collect()
            
        except Exception as e:
            logger.error(f"Error processing batch {i//batch_size + 1}: {e}")
            # Create dummy embeddings for this batch to maintain alignment
            dummy_embeddings = np.zeros((len(batch), 384), dtype=np.float32)  # MiniLM-L6-v2 has 384 dimensions
            all_embeddings.append(dummy_embeddings)
    
    # Concatenate all embeddings
    logger.info("Concatenating all embeddings...")
    embeddings = np.concatenate(all_embeddings, axis=0).astype('float32')
    logger.info(f"✓ Generated embeddings with shape: {embeddings.shape}")
    
    return embeddings

def main():
    # Define paths
    data_dir = project_root / "data"
    vector_outputs_dir = project_root / "vector_outputs"
    processed_json_path = vector_outputs_dir / "processed_1.json"
    
    # Output paths
    faiss_index_path = vector_outputs_dir / "scaffold_index_1.faiss"
    metadata_json_path = vector_outputs_dir / "scaffold_metadata_1.json"
    
    # Ensure output directory exists
    vector_outputs_dir.mkdir(exist_ok=True)
    
    logger.info("Starting robust vectorization process...")
    logger.info(f"Using processed chunks from: {processed_json_path}")
    
    # Check if processed file exists
    if not processed_json_path.exists():
        logger.error(f"❌ Error: Processed chunks file not found at {processed_json_path}")
        logger.error("Please run the chunking process first:")
        logger.error("python -m scaffold_core.vector.chunk")
        return False
    
    try:
        # Load processed chunks
        logger.info("Loading processed chunks...")
        with open(processed_json_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        
        logger.info(f"✓ Loaded {len(chunks)} chunks")
        
        # Process chunks in batches to avoid memory issues
        texts, metadata = process_chunks_in_batches(chunks, batch_size=500)
        
        if not texts:
            logger.error("❌ No valid texts found after processing")
            return False
        
        # Load the embedding model
        logger.info("Loading embedding model: all-MiniLM-L6-v2")
        try:
            model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            return False
        
        # Generate embeddings in batches
        embeddings = create_embeddings_in_batches(model, texts, batch_size=100)
        
        # Create FAISS index
        logger.info("Creating FAISS index...")
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        
        # Add embeddings to index in batches
        batch_size = 1000
        for i in range(0, len(embeddings), batch_size):
            batch_embeddings = embeddings[i:i + batch_size]
            index.add(batch_embeddings)
            logger.info(f"Added batch {i//batch_size + 1}/{(len(embeddings) + batch_size - 1)//batch_size} to index")
        
        logger.info(f"✓ Created FAISS index with {index.ntotal} vectors")
        
        # Save the index and metadata
        logger.info("Saving files...")
        faiss.write_index(index, str(faiss_index_path))
        
        with open(metadata_json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✓ Built FAISS index ({len(texts)} vectors) → {faiss_index_path}")
        logger.info(f"✓ Saved metadata → {metadata_json_path}")
        
        # Verify the files exist and have reasonable sizes
        if faiss_index_path.exists() and metadata_json_path.exists():
            faiss_size = faiss_index_path.stat().st_size / (1024 * 1024)  # MB
            metadata_size = metadata_json_path.stat().st_size / (1024 * 1024)  # MB
            logger.info(f"✓ FAISS index size: {faiss_size:.2f} MB")
            logger.info(f"✓ Metadata size: {metadata_size:.2f} MB")
            logger.info("🎉 Vector indexing completed successfully!")
            return True
        else:
            logger.error("❌ Files were not created properly")
            return False
            
    except MemoryError as e:
        logger.error(f"❌ Memory error: {e}")
        logger.error("Consider reducing batch sizes or processing fewer documents")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Vector indexing completed successfully!")
        print("You can now run the UI with: cd frontend && python start_enhanced_ui.py")
    else:
        print("\n❌ Vector indexing failed!")
        sys.exit(1)