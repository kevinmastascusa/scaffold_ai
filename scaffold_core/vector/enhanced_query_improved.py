"""
Improved Enhanced Query System for Scaffold AI
Better prompt engineering and citation handling for stable responses.
"""

import datetime
import json
import logging
import os
import re
import sys
from typing import Any, Dict, List, Optional

import faiss
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Add the project root to the Python path
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if project_root not in sys.path:
    sys.path.append(project_root)

# Import configuration and LLM
from scaffold_core.config import (
    VECTOR_OUTPUTS_DIR, ITERATION, EMBEDDING_MODEL,
    get_faiss_index_path, get_metadata_json_path
)
from scaffold_core.llm import llm

# Constants
TOP_K_INITIAL = 50
TOP_K_FINAL = 5
MIN_CROSS_SCORE = -2.0  # Minimum cross-encoder score threshold
MIN_CONTEXTUAL_SCORE = 2  # Minimum contextual score threshold

class ImprovedEnhancedQuerySystem:
    """Improved enhanced query system with better prompt engineering."""
    
    def __init__(self):
        self.initialized = False
        self.embedding_model = None
        self.cross_encoder = None
        self.faiss_index = None
        self.metadata = []
        
    def initialize(self):
        """Initialize the enhanced query system."""
        logger.info("Initializing improved enhanced query system...")
        
        try:
            # Load embedding model
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
            
            # Load cross-encoder for reranking
            self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            
            # Load FAISS index and metadata
            index_path = get_faiss_index_path(ITERATION)
            metadata_path = get_metadata_json_path(ITERATION)
            
            if not os.path.exists(index_path) or not os.path.exists(metadata_path):
                raise FileNotFoundError(f"Index or metadata not found: {index_path}, {metadata_path}")
            
            self.faiss_index = faiss.read_index(str(index_path))
            self.metadata = self.load_metadata(str(metadata_path))
            
            logger.info(f"✅ Enhanced query system initialized successfully")
            logger.info(f"   - FAISS index: {self.faiss_index.ntotal} vectors")
            logger.info(f"   - Metadata: {len(self.metadata)} entries")
            
            self.initialized = True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize enhanced query system: {e}")
            raise
    
    def load_metadata(self, path: str) -> List[Dict]:
        """Load metadata from JSON file."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            logger.info(f"Loaded {len(metadata)} metadata entries from {path}")
            return metadata
        except Exception as e:
            logger.error(f"Error loading metadata from {path}: {e}")
            return []
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text."""
        # Remove punctuation and convert to lowercase
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        words = text.split()
        
        # Filter out common stop words and short words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her',
            'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their'
        }
        
        keywords = [word for word in words if len(word) > 2 and word not in stop_words]
        return keywords
    
    def semantic_search(self, query: str, k: int = TOP_K_INITIAL) -> List[Dict]:
        """Perform semantic search using sentence transformers."""
        try:
            # Encode query
            query_embedding = self.embedding_model.encode([query])
            
            # Search FAISS index
            scores, indices = self.faiss_index.search(query_embedding, k)
            
            # Build results
            candidates = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.metadata):
                    candidate = self.metadata[idx].copy()
                    candidate['semantic_score'] = float(score)
                    candidate['search_type'] = 'semantic'
                    candidates.append(candidate)
            
            return candidates
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []
    
    def keyword_search(self, query: str, k: int = TOP_K_INITIAL) -> List[Dict]:
        """Perform keyword-based search."""
        try:
            query_keywords = set(self.extract_keywords(query))
            if not query_keywords:
                return []
            
            candidates = []
            for i, metadata_item in enumerate(self.metadata):
                text_keywords = set(self.extract_keywords(metadata_item.get('text', '')))
                shared_keywords = query_keywords.intersection(text_keywords)
                
                if shared_keywords:
                    candidate = metadata_item.copy()
                    candidate['keyword_score'] = len(shared_keywords) / len(query_keywords)
                    candidate['keyword_matches'] = len(shared_keywords)
                    candidate['search_type'] = 'keyword'
                    candidates.append(candidate)
            
            # Sort by keyword score and take top k
            candidates.sort(key=lambda x: x['keyword_score'], reverse=True)
            return candidates[:k]
            
        except Exception as e:
            logger.error(f"Error in keyword search: {e}")
            return []
    
    def hybrid_search(self, query: str, k: int = TOP_K_INITIAL) -> List[Dict]:
        """Combine semantic and keyword search results."""
        semantic_candidates = self.semantic_search(query, k)
        keyword_candidates = self.keyword_search(query, k)
        
        # Create a combined set of candidates
        all_candidates = {}
        
        # Add semantic candidates
        for candidate in semantic_candidates:
            chunk_id = candidate.get('chunk_id')
            if chunk_id not in all_candidates:
                all_candidates[chunk_id] = candidate
            else:
                # Merge scores if candidate exists
                existing = all_candidates[chunk_id]
                existing['semantic_score'] = candidate['semantic_score']
                existing['search_type'] = 'hybrid'
        
        # Add keyword candidates
        for candidate in keyword_candidates:
            chunk_id = candidate.get('chunk_id')
            if chunk_id not in all_candidates:
                all_candidates[chunk_id] = candidate
            else:
                # Merge scores if candidate exists
                existing = all_candidates[chunk_id]
                existing['keyword_score'] = candidate['keyword_score']
                existing['keyword_matches'] = candidate['keyword_matches']
                existing['search_type'] = 'hybrid'
        
        # Convert back to list and sort by combined score
        candidates = list(all_candidates.values())
        for candidate in candidates:
            semantic_score = candidate.get('semantic_score', 0)
            keyword_score = candidate.get('keyword_score', 0)
            # Combine scores (semantic weighted more heavily)
            candidate['combined_score'] = (semantic_score * 0.7) + (keyword_score * 0.3)
        
        candidates.sort(key=lambda x: x['combined_score'], reverse=True)
        return candidates[:k]
    
    def cross_encoder_rerank(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """Rerank candidates using cross-encoder with improved error handling."""
        if not candidates:
            return []
        
        try:
            # Prepare pairs for cross-encoder
            pairs = [[query, candidate.get("text", "")] for candidate in candidates]
            
            # Get cross-encoder scores
            cross_scores = self.cross_encoder.predict(pairs)
            
            # Add cross-encoder scores to candidates
            for candidate, score in zip(candidates, cross_scores):
                candidate["cross_score"] = float(score)
                
                # Filter out candidates with very low scores
                if candidate["cross_score"] < MIN_CROSS_SCORE:
                    candidate["filtered_out"] = True
            
            # Remove filtered candidates
            candidates = [c for c in candidates if not c.get("filtered_out", False)]
            
            # Sort by cross-encoder score
            candidates.sort(key=lambda x: x["cross_score"], reverse=True)
            
            return candidates
            
        except Exception as e:
            logger.error(f"Error in cross-encoder reranking: {e}")
            return candidates
    
    def contextual_filtering(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """Apply improved contextual filtering."""
        if not candidates:
            return []
        
        try:
            query_keywords = set(self.extract_keywords(query))
            if not query_keywords:
                return candidates
            
            for candidate in candidates:
                text_keywords = set(self.extract_keywords(candidate.get('text', '')))
                shared_keywords = query_keywords.intersection(text_keywords)
                candidate['contextual_score'] = len(shared_keywords)
                
                # Filter out candidates with very low contextual scores
                if candidate['contextual_score'] < MIN_CONTEXTUAL_SCORE:
                    candidate["filtered_out"] = True
            
            # Remove filtered candidates
            candidates = [c for c in candidates if not c.get("filtered_out", False)]
            
            # Sort by contextual score
            candidates.sort(key=lambda x: x['contextual_score'], reverse=True)
            
            return candidates
            
        except Exception as e:
            logger.error(f"Error in contextual filtering: {e}")
            return candidates
    
    def generate_improved_prompt(self, query: str, chunks: List[Dict]) -> str:
        """Generate an improved prompt with better engineering."""
        if not chunks:
            return f"Query: {query}\n\nI don't have enough relevant information to answer this query accurately."
        
        # Prepare context and citations
        context_parts = []
        citation_refs = []
        estimated_tokens = 0
        max_context_tokens = 2000  # Increased context window
        
        for i, chunk in enumerate(chunks):
            chunk_text = chunk.get('text', '').strip()
            if not chunk_text:
                continue
                
            chunk_tokens = len(chunk_text) // 4
            
            # Check token limit
            if estimated_tokens + chunk_tokens > max_context_tokens:
                logger.warning(f"Truncating context at chunk {i+1}")
                break
            
            source = chunk.get('source', {})
            citation_id = source.get('id', f'source_{i+1}')
            citation_name = source.get('name', 'Unknown Source')
            ref = f"[{i+1}]"
            
            # Add chunk with better formatting
            context_parts.append(f"Source {ref}:\n{chunk_text}\n")
            estimated_tokens += chunk_tokens
            
            # Store citation details
            if citation_id not in [c['id'] for c in citation_refs]:
                citation_refs.append({
                    'ref': ref, 'id': citation_id, 'name': citation_name
                })
        
        # Create context string
        context = "\n".join(context_parts)
        
        # Create citation list
        citations_str = "\n".join([f"{c['ref']}: {c['name']}" for c in citation_refs])
        
        # Improved prompt template
        prompt = f"""You are a helpful AI assistant that provides accurate, relevant, and well-cited responses based on the provided sources.

TASK: Answer the following query using ONLY the information from the provided sources.

QUERY: {query}

SOURCES:
{context}

CITATION LIST:
{citations_str}

INSTRUCTIONS:
1. Answer the query comprehensively using information from the sources
2. Use specific details and examples from the sources
3. Cite sources using [1], [2], etc. format at the end of relevant sentences
4. Avoid repetition and stay focused on the query
5. If the sources don't contain enough information, say so clearly
6. Write in a clear, professional tone
7. Keep the response concise but complete

ANSWER:"""
        
        total_tokens = len(prompt) // 4
        logger.info(f"Generated improved prompt with ~{total_tokens} tokens")
        
        return prompt
    
    def query(self, query: str) -> Dict[str, Any]:
        """Main query function with improved processing."""
        if not self.initialized:
            self.initialize()
        
        logger.info(f"Processing query: {query}")
        
        try:
            # Step 1: Hybrid search
            initial_candidates = self.hybrid_search(query, TOP_K_INITIAL)
            logger.debug(f"Found {len(initial_candidates)} initial candidates")
            
            # Step 2: Cross-encoder reranking
            reranked_candidates = self.cross_encoder_rerank(query, initial_candidates)
            logger.debug(f"Reranked to {len(reranked_candidates)} candidates")
            
            # Step 3: Contextual filtering
            filtered_candidates = self.contextual_filtering(query, reranked_candidates)
            logger.debug(f"Filtered to {len(filtered_candidates)} candidates")
            
            # Step 4: Select top candidates
            final_candidates = filtered_candidates[:TOP_K_FINAL]
            logger.debug(f"Selected {len(final_candidates)} final candidates")
            
            # Step 5: Generate improved prompt and get LLM response
            improved_prompt = self.generate_improved_prompt(query, final_candidates)
            
            # Generate response with lower temperature for more stable output
            llm_response = llm.generate_response(improved_prompt, temperature=0.1)
            
            # Prepare results with improved structure
            results = {
                "query": query,
                "response": llm_response,
                "candidates_found": len(final_candidates),
                "sources": [
                    {
                        "score": candidate.get("cross_score", 0),
                        "source": candidate.get("source", {}),
                        "text_preview": candidate.get("text", "")[:200] + "..."
                    }
                    for candidate in final_candidates
                ],
                "search_stats": {
                    "initial_candidates": len(initial_candidates),
                    "reranked_candidates": len(reranked_candidates),
                    "filtered_candidates": len(filtered_candidates),
                    "final_candidates": len(final_candidates)
                },
                "timestamp": str(datetime.datetime.now())
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in query processing: {e}")
            return {
                "query": query,
                "response": f"Error processing query: {str(e)}",
                "candidates_found": 0,
                "sources": [],
                "search_stats": {},
                "timestamp": str(datetime.datetime.now())
            }

# Global instance
improved_enhanced_query_system = ImprovedEnhancedQuerySystem()

def query_enhanced_improved(query: str) -> Dict[str, Any]:
    """Convenience function for improved enhanced querying."""
    return improved_enhanced_query_system.query(query) 