"""
Enhanced Query System for Scaffold AI
Improved chunk retrieval and ranking for better LLM responses.
"""

import os
import json
import numpy as np
import faiss
import torch
import re
from sentence_transformers import SentenceTransformer, CrossEncoder
import sys
import logging
from typing import List, Dict, Any, Tuple
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

logger.debug("Starting enhanced query module initialization...")

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)
logger.debug(f"Added project root to Python path: {project_root}")

logger.debug("Importing configuration...")
from scaffold_core.config import (
    EMBEDDING_MODEL,
    CROSS_ENCODER_MODEL,
    TOP_K_INITIAL,
    TOP_K_FINAL,
    get_faiss_index_path,
    get_metadata_json_path
)
logger.debug("Configuration imported successfully")

logger.debug("Importing LLM manager...")
from scaffold_core.llm import llm
logger.debug("LLM manager imported successfully")

class EnhancedQuerySystem:
    def __init__(self):
        """Initialize the enhanced query system."""
        self.embedding_model = None
        self.cross_encoder = None
        self.index = None
        self.metadata = None
        self.initialized = False
        
    def initialize(self):
        """Initialize all models and data."""
        if self.initialized:
            return
            
        logger.debug("Initializing enhanced query system...")
        
        # Load embedding model
        logger.debug(f"Loading embedding model: {EMBEDDING_MODEL}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL, device=device)
        
        # Load cross-encoder
        logger.debug(f"Loading cross-encoder model: {CROSS_ENCODER_MODEL}")
        self.cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL, device=device)
        
        # Load FAISS index
        logger.debug("Loading FAISS index...")
        index_path = get_faiss_index_path()
        self.index = faiss.read_index(str(index_path))
        
        # Load metadata
        logger.debug("Loading metadata...")
        self.metadata = self.load_metadata(get_metadata_json_path())
        
        self.initialized = True
        logger.debug("Enhanced query system initialized successfully")
    
    def load_metadata(self, path):
        """Load metadata from JSON file."""
        logger.debug(f"Loading metadata from: {path}")
        try:
            with open(path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
                logger.debug(f"Successfully loaded metadata with {len(metadata)} entries")
                return metadata
        except Exception as e:
            logger.error(f"Failed to load metadata: {str(e)}")
            raise
    
    def extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from the query."""
        # Remove common stop words and extract meaningful terms
        stop_words = {'what', 'is', 'are', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'how', 'why', 'when', 'where', 'which', 'who', 'that', 'this', 'these', 'those', 'do', 'does', 'did', 'can', 'could', 'will', 'would', 'should', 'may', 'might', 'must', 'have', 'has', 'had', 'be', 'been', 'being', 'am', 'is', 'are', 'was', 'were'}
        
        # Clean and tokenize
        words = re.findall(r'\b[a-zA-Z]+\b', query.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Add bigrams for better matching
        bigrams = []
        for i in range(len(words) - 1):
            bigram = f"{words[i]} {words[i+1]}"
            if len(bigram) > 5:
                bigrams.append(bigram)
        
        return keywords + bigrams
    
    def semantic_search(self, query: str, k: int = TOP_K_INITIAL) -> List[Dict]:
        """Perform semantic search using embeddings."""
        logger.debug(f"Performing semantic search with k={k}")
        
        # Check if index and metadata are properly loaded
        if self.index is None:
            logger.error("FAISS index is not loaded")
            return []
        
        if self.metadata is None or len(self.metadata) == 0:
            logger.error("Metadata is not loaded or empty")
            return []
        
        logger.debug(f"Index size: {self.index.ntotal}, Metadata size: {len(self.metadata)}")
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query], show_progress_bar=False)
        
        # Search FAISS index
        distances, indices = self.index.search(query_embedding, k=k)
        
        # Prepare results
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.metadata):
                entry = self.metadata[idx]
                if isinstance(entry, dict) and "text" in entry:
                    results.append({
                        "chunk_id": int(idx),
                        "score": float(distance),
                        "text": entry.get("text", ""),
                        "source": entry.get("source_path", ""),
                        "search_type": "semantic"
                    })
        
        logger.debug(f"Semantic search found {len(results)} results")
        return results
    
    def keyword_search(self, query: str, k: int = TOP_K_INITIAL) -> List[Dict]:
        """Perform keyword-based search."""
        logger.debug(f"Performing keyword search with k={k}")
        
        # Check if metadata is properly loaded
        if self.metadata is None or len(self.metadata) == 0:
            logger.error("Metadata is not loaded or empty for keyword search")
            return []
        
        keywords = self.extract_keywords(query)
        logger.debug(f"Extracted keywords: {keywords}")
        
        results = []
        for i, entry in enumerate(self.metadata):
            if isinstance(entry, dict) and "text" in entry:
                text = entry.get("text", "").lower()
                
                # Calculate keyword match score
                matches = sum(1 for keyword in keywords if keyword in text)
                if matches > 0:
                    # Normalize score by text length and keyword count
                    score = matches / (len(keywords) + 1) * (1 + matches / len(text.split()))
                    results.append({
                        "chunk_id": i,
                        "score": score,
                        "text": entry.get("text", ""),
                        "source": entry.get("source_path", ""),
                        "search_type": "keyword",
                        "keyword_matches": matches
                    })
        
        # Sort by score and take top k
        results.sort(key=lambda x: x["score"], reverse=True)
        logger.debug(f"Keyword search found {len(results)} results")
        return results[:k]
    
    def hybrid_search(self, query: str, k: int = TOP_K_INITIAL) -> List[Dict]:
        """Combine semantic and keyword search for better results."""
        logger.debug(f"Performing hybrid search with k={k}")
        
        # Get results from both search methods
        semantic_results = self.semantic_search(query, k=k//2)
        keyword_results = self.keyword_search(query, k=k//2)
        
        # Combine and deduplicate
        combined = {}
        
        # Add semantic results
        for result in semantic_results:
            chunk_id = result["chunk_id"]
            combined[chunk_id] = {
                **result,
                "semantic_score": result["score"],
                "keyword_score": 0.0
            }
        
        # Add keyword results
        for result in keyword_results:
            chunk_id = result["chunk_id"]
            if chunk_id in combined:
                combined[chunk_id]["keyword_score"] = result["score"]
                # Boost score if both methods found it
                combined[chunk_id]["score"] = (combined[chunk_id]["semantic_score"] + result["score"]) / 2
            else:
                combined[chunk_id] = {
                    **result,
                    "semantic_score": 0.0,
                    "keyword_score": result["score"]
                }
        
        # Convert back to list and sort
        results = list(combined.values())
        results.sort(key=lambda x: x["score"], reverse=True)
        
        return results[:k]
    
    def cross_encoder_rerank(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """Rerank candidates using cross-encoder."""
        logger.debug(f"Reranking {len(candidates)} candidates with cross-encoder")
        
        # Handle empty candidates list
        if not candidates:
            logger.warning("No candidates to rerank")
            return []
        
        # Prepare pairs for cross-encoder
        pairs = [[query, candidate["text"]] for candidate in candidates]
        
        # Get cross-encoder scores
        cross_scores = self.cross_encoder.predict(pairs)
        
        # Add cross-encoder scores to candidates
        for candidate, score in zip(candidates, cross_scores):
            candidate["cross_score"] = float(score)
        
        # Sort by cross-encoder score
        candidates.sort(key=lambda x: x["cross_score"], reverse=True)
        
        return candidates
    
    def contextual_filtering(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """Apply contextual filtering to improve relevance."""
        logger.debug("Applying contextual filtering")
        
        query_lower = query.lower()
        keywords = self.extract_keywords(query)
        
        filtered_candidates = []
        for candidate in candidates:
            text = candidate["text"].lower()
            
            # Check for direct keyword matches
            keyword_matches = sum(1 for keyword in keywords if keyword in text)
            
            # Check for semantic similarity indicators
            semantic_indicators = 0
            if any(word in text for word in ['sustainability', 'environmental', 'assessment', 'life cycle']):
                semantic_indicators += 1
            if any(word in text for word in ['method', 'process', 'procedure', 'technique']):
                semantic_indicators += 1
            if any(word in text for word in ['example', 'case study', 'application', 'implementation']):
                semantic_indicators += 1
            
            # Calculate contextual relevance score
            contextual_score = (keyword_matches * 0.6) + (semantic_indicators * 0.4)
            
            if contextual_score > 0.1:  # Minimum relevance threshold
                candidate["contextual_score"] = contextual_score
                filtered_candidates.append(candidate)
        
        # Sort by contextual score
        filtered_candidates.sort(key=lambda x: x["contextual_score"], reverse=True)
        
        return filtered_candidates
    
    def generate_enhanced_prompt(self, query: str, chunks: List[Dict]) -> str:
        """Generate an enhanced prompt for the LLM with better context."""
        logger.debug(f"Generating enhanced prompt with {len(chunks)} chunks")
        
        # Group chunks by source for better organization
        source_groups = defaultdict(list)
        for chunk in chunks:
            source = chunk.get("source", "Unknown")
            source_groups[source].append(chunk)
        
        prompt = f"""You are an expert assistant specializing in sustainability, life cycle assessment, and environmental sciences. 

TASK: Answer the following query using ONLY the information provided in the text chunks below. If the information is not available in the chunks, respond with "The requested information is not found in the available documentation."

QUERY: {query}

AVAILABLE DOCUMENTATION:
"""
        
        # Add chunks organized by source
        for source, source_chunks in source_groups.items():
            prompt += f"\n--- Source: {source} ---\n"
            for i, chunk in enumerate(source_chunks, 1):
                # Clean and format text
                text = chunk["text"].replace("\n", " ").strip()
                if len(text) > 600:
                    text = text[:600] + "..."
                
                prompt += f"{i}. {text}\n"
        
        prompt += """

INSTRUCTIONS:
1. Use ONLY the information from the provided chunks
2. If multiple chunks contain relevant information, synthesize them coherently
3. Cite specific chunks by number when referencing information
4. If the query cannot be answered from the chunks, say so clearly
5. Provide a clear, well-structured response

ANSWER:"""
        
        return prompt
    
    def query(self, query: str) -> Dict[str, Any]:
        """Main query function with enhanced retrieval."""
        if not self.initialized:
            self.initialize()
        
        logger.info(f"Processing query: {query}")
        
        # Step 1: Hybrid search
        logger.debug("Step 1: Hybrid search")
        initial_candidates = self.hybrid_search(query, k=TOP_K_INITIAL)
        logger.debug(f"Found {len(initial_candidates)} initial candidates")
        
        # Step 2: Cross-encoder reranking
        logger.debug("Step 2: Cross-encoder reranking")
        reranked_candidates = self.cross_encoder_rerank(query, initial_candidates)
        
        # Step 3: Contextual filtering
        logger.debug("Step 3: Contextual filtering")
        filtered_candidates = self.contextual_filtering(query, reranked_candidates)
        
        # Step 4: Select top candidates for LLM
        final_candidates = filtered_candidates[:TOP_K_FINAL]
        logger.debug(f"Selected {len(final_candidates)} final candidates for LLM")
        
        # Step 5: Generate enhanced prompt and get LLM response
        logger.debug("Step 5: Generating LLM response")
        enhanced_prompt = self.generate_enhanced_prompt(query, final_candidates)
        
        try:
            llm_response = llm.generate_response(enhanced_prompt, temperature=0.3)
        except Exception as e:
            logger.error(f"LLM response generation failed: {str(e)}")
            llm_response = f"Error generating response: {str(e)}"
        
        # Prepare comprehensive results
        results = {
            "query": query,
            "response": llm_response,
            "candidates": final_candidates,
            "search_stats": {
                "initial_candidates": len(initial_candidates),
                "reranked_candidates": len(reranked_candidates),
                "filtered_candidates": len(filtered_candidates),
                "final_candidates": len(final_candidates)
            }
        }
        
        return results

# Global instance
enhanced_query_system = EnhancedQuerySystem()

def query_enhanced(query: str) -> Dict[str, Any]:
    """Convenience function for enhanced querying."""
    return enhanced_query_system.query(query) 