#!/usr/bin/env python3
"""
Improved Vector Index Creation Script
Handles memory-efficient processing and better tokenization for large datasets.
"""

import os
import sys
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import gc
from pathlib import Path
import logging
from typing import List, Dict, Any
import unicodedata
import re

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

class ImprovedVectorizer:
    def __init__(self, batch_size: int = 256, max_seq_length: int = 512):
        """
        Initialize the improved vectorizer.
        
        Args:
            batch_size: Number of texts to process at once
            max_seq_length: Maximum sequence length for the model
        """
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.model = None
        
    def load_model(self) -> None:
        """Load the sentence transformer model with optimized settings."""
        logger.info(f"Loading embedding model: {SELECTED_EMBEDDING_MODEL}")
        try:
            # Load model with specific device and optimization settings
            self.model = SentenceTransformer(SELECTED_EMBEDDING_MODEL)
            
            # Set max sequence length
            if hasattr(self.model, 'max_seq_length'):
                self.model.max_seq_length = self.max_seq_length
            
            # Try to use MPS if available (Mac M1/M2), otherwise CPU
            if hasattr(self.model, '_target_device'):
                try:
                    import torch
                    if torch.backends.mps.is_available():
                        self.model = self.model.to('mps')
                        logger.info("Using MPS (Metal Performance Shaders) acceleration")
                    else:
                        self.model = self.model.to('cpu')
                        logger.info("Using CPU for embeddings")
                except Exception as e:
                    logger.warning(f"Could not set device, using default: {e}")
                    
            logger.info("✓ Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def clean_text_advanced(self, text: str) -> str:
        """
        Advanced text cleaning for better tokenization.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Unicode normalization
        text = unicodedata.normalize('NFKC', text)
        
        # Remove control characters but keep newlines and tabs
        text = ''.join(char for char in text if unicodedata.category(char)[0] != 'C' or char in '\n\t')
        
        # Fix common encoding issues
        text = text.replace('\ufeff', '')  # Remove BOM
        text = text.replace('\u200b', '')  # Remove zero-width space
        text = text.replace('\u00a0', ' ')  # Replace non-breaking space
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Ensure reasonable length
        if len(text) > 8000:  # Truncate very long texts
            text = text[:8000] + "..."
        
        return text
    
    def prepare_texts_batch(self, chunks: List[Dict[str, Any]], start_idx: int, end_idx: int) -> tuple:
        """
        Prepare a batch of texts for embedding.
        
        Args:
            chunks: List of chunk dictionaries
            start_idx: Start index for batch
            end_idx: End index for batch
            
        Returns:
            Tuple of (cleaned_texts, metadata)
        """
        texts = []
        metadata = []
        
        for i in range(start_idx, min(end_idx, len(chunks))):
            chunk = chunks[i]
            
            # Clean and prepare text
            cleaned_text = self.clean_text_advanced(chunk.get("text", ""))
            
            if len(cleaned_text) < 10:  # Skip very short texts
                continue
                
            texts.append(cleaned_text)
            
            # Prepare metadata
            chunk_metadata = {k: v for k, v in chunk.items() if k != "text"}
            chunk_metadata["text"] = cleaned_text
            chunk_metadata["original_index"] = i
            metadata.append(chunk_metadata)
        
        return texts, metadata
    
    def create_embeddings_batched(self, chunks: List[Dict[str, Any]]) -> tuple:
        """
        Create embeddings in batches to avoid memory issues.
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            Tuple of (embeddings_array, metadata_list)
        """
        if not self.model:
            self.load_model()
        
        total_chunks = len(chunks)
        logger.info(f"Processing {total_chunks} chunks in batches of {self.batch_size}")
        
        all_embeddings = []
        all_metadata = []
        
        for start_idx in range(0, total_chunks, self.batch_size):
            end_idx = start_idx + self.batch_size
            batch_num = (start_idx // self.batch_size) + 1
            total_batches = (total_chunks + self.batch_size - 1) // self.batch_size
            
            logger.info(f"Processing batch {batch_num}/{total_batches} (chunks {start_idx}-{min(end_idx, total_chunks)})")
            
            # Prepare batch
            texts, metadata = self.prepare_texts_batch(chunks, start_idx, end_idx)
            
            if not texts:
                continue
                
            try:
                # Generate embeddings for this batch
                batch_embeddings = self.model.encode(
                    texts, 
                    batch_size=min(32, len(texts)),  # Smaller sub-batches
                    show_progress_bar=False,
                    convert_to_tensor=False,
                    normalize_embeddings=True  # Normalize for better similarity search
                )
                
                # Convert to numpy and ensure float32
                batch_embeddings = np.array(batch_embeddings, dtype=np.float32)
                
                all_embeddings.append(batch_embeddings)
                all_metadata.extend(metadata)
                
                logger.info(f"✓ Batch {batch_num} completed: {len(texts)} embeddings generated")
                
                # Force garbage collection to free memory
                gc.collect()
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_num}: {e}")
                # Skip this batch and continue
                continue
        
        if not all_embeddings:
            raise ValueError("No embeddings were generated successfully")
        
        # Combine all embeddings
        logger.info("Combining all embeddings...")
        final_embeddings = np.vstack(all_embeddings)
        
        logger.info(f"✓ Generated {len(final_embeddings)} total embeddings with shape: {final_embeddings.shape}")
        
        return final_embeddings, all_metadata
    
    def create_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """
        Create an optimized FAISS index.
        
        Args:
            embeddings: Numpy array of embeddings
            
        Returns:
            FAISS index
        """
        logger.info("Creating FAISS index...")
        
        dim = embeddings.shape[1]
        n_vectors = embeddings.shape[0]
        
        # Choose index type based on dataset size
        if n_vectors < 1000:
            # For small datasets, use flat index
            index = faiss.IndexFlatIP(dim)  # Inner product for normalized vectors
            logger.info(f"Using IndexFlatIP for {n_vectors} vectors")
        elif n_vectors < 10000:
            # For medium datasets, use IVF with small number of clusters
            nlist = min(100, n_vectors // 10)
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, nlist)
            index.train(embeddings)
            logger.info(f"Using IndexIVFFlat with {nlist} clusters for {n_vectors} vectors")
        else:
            # For large datasets, use IVF with more clusters
            nlist = min(1000, n_vectors // 50)
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, nlist)
            index.train(embeddings)
            logger.info(f"Using IndexIVFFlat with {nlist} clusters for {n_vectors} vectors")
        
        # Add vectors to index
        index.add(embeddings)
        
        logger.info(f"✓ FAISS index created with {index.ntotal} vectors")
        return index
    
    def save_index_and_metadata(self, index: faiss.Index, metadata: List[Dict], iteration: int = 1) -> None:
        """
        Save the FAISS index and metadata to files.
        
        Args:
            index: FAISS index
            metadata: List of metadata dictionaries
            iteration: Iteration number for file naming
        """
        # Ensure output directory exists
        VECTOR_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Get file paths
        index_path = get_faiss_index_path(iteration)
        metadata_path = get_metadata_json_path(iteration)
        
        logger.info("Saving files...")
        
        # Save FAISS index
        faiss.write_index(index, str(index_path))
        logger.info(f"✓ FAISS index saved to: {index_path}")
        
        # Save metadata with error handling
        try:
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2, default=str)
            logger.info(f"✓ Metadata saved to: {metadata_path}")
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
            # Try saving without indentation to reduce memory usage
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, default=str)
            logger.info(f"✓ Metadata saved (compact format) to: {metadata_path}")

def load_processed_chunks() -> List[Dict[str, Any]]:
    """Load processed chunks from the JSON file."""
    # Try multiple possible locations for the processed data
    possible_paths = [
        Path("vector_outputs/processed_1.json"),
        Path("outputs/chunked_text_extracts_cleaned.json"),
        Path("outputs/chunked_text_extracts.json"),
        Path("vector_outputs/chunked_text_extracts.json")
    ]
    
    chunks_file = None
    for path in possible_paths:
        if path.exists():
            chunks_file = path
            logger.info(f"Found chunks file: {path}")
            break
    
    if not chunks_file:
        raise FileNotFoundError(f"Could not find processed chunks file. Tried: {[str(p) for p in possible_paths]}")
    
    logger.info(f"Loading chunks from: {chunks_file}")
    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    logger.info(f"✓ Loaded {len(chunks)} chunks")
    return chunks

def main():
    """Main function to create improved vector index."""
    try:
        logger.info("🚀 Starting Improved Vector Index Creation")
        logger.info("=" * 60)
        
        # Load processed chunks
        chunks = load_processed_chunks()
        
        # Initialize vectorizer with optimized settings
        vectorizer = ImprovedVectorizer(
            batch_size=128,  # Smaller batches for better memory management
            max_seq_length=512  # Reasonable sequence length
        )
        
        # Create embeddings
        embeddings, metadata = vectorizer.create_embeddings_batched(chunks)
        
        # Create FAISS index
        index = vectorizer.create_faiss_index(embeddings)
        
        # Save results
        vectorizer.save_index_and_metadata(index, metadata, ITERATION)
        
        logger.info("=" * 60)
        logger.info("✅ Vector index creation completed successfully!")
        logger.info(f"📊 Total vectors: {index.ntotal}")
        logger.info(f"📁 Index saved to: {get_faiss_index_path(ITERATION)}")
        logger.info(f"📁 Metadata saved to: {get_metadata_json_path(ITERATION)}")
        
    except Exception as e:
        logger.error(f"❌ Error creating vector index: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()