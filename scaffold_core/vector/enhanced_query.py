"""
Enhanced Query System for Scaffold AI
Improved chunk retrieval and ranking for better LLM responses.
"""

import json
import logging
import os
import re
import sys
from typing import Any, Dict, List

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

logger.debug("Starting enhanced query module initialization...")

# Add the project root to the Python path
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if project_root not in sys.path:
    sys.path.append(project_root)
logger.debug(f"Added project root to Python path: {project_root}")

from scaffold_core.citation_handler import Citation
from scaffold_core.config import (CROSS_ENCODER_MODEL, EMBEDDING_MODEL,
                                  TOP_K_FINAL, TOP_K_INITIAL,
                                  get_faiss_index_path, get_metadata_json_path)
from scaffold_core.llm import llm

logger.debug("Configuration, LLM, and Citation handler imported.")


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
        self.embedding_model = SentenceTransformer(
            EMBEDDING_MODEL, device=device
        )

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
                logger.debug(
                    f"Successfully loaded metadata with {len(metadata)} "
                    "entries"
                )
                return metadata
        except Exception as e:
            logger.error(f"Failed to load metadata: {str(e)}")
            raise

    def extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from the query."""
        stop_words = {
            'what', 'is', 'are', 'the', 'a', 'an', 'and', 'or', 'but', 'in',
            'on', 'at', 'to', 'for', 'of', 'with', 'by', 'how', 'why',
            'when', 'where', 'which', 'who', 'that', 'this', 'these',
            'those', 'do', 'does', 'did', 'can', 'could', 'will', 'would',
            'should', 'may', 'might', 'must', 'have', 'has', 'had', 'be',
            'been', 'being', 'am', 'is', 'are', 'was', 'were'
        }

        # Clean and tokenize
        words = re.findall(r'\b[a-zA-Z]+\b', query.lower())
        keywords = [
            word for word in words if word not in stop_words and len(word) > 2
        ]

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

        if self.index is None:
            logger.error("FAISS index is not loaded")
            return []

        if self.metadata is None or len(self.metadata) == 0:
            logger.error("Metadata is not loaded or empty")
            return []

        logger.debug(
            f"Index size: {self.index.ntotal}, "
            f"Metadata size: {len(self.metadata)}"
        )

        query_embedding = self.embedding_model.encode(
            [query], show_progress_bar=False
        )
        distances, indices = self.index.search(query_embedding, k=k)

        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.metadata):
                entry = self.metadata[idx]
                if isinstance(entry, dict) and "text" in entry:
                    source_path = entry.get("source_path", "")
                    results.append({
                        "chunk_id": int(idx),
                        "score": float(distance),
                        "text": entry.get("text", ""),
                        "source": Citation(source_path).to_dict(),
                        "search_type": "semantic"
                    })

        logger.debug(f"Semantic search found {len(results)} results")
        return results

    def keyword_search(self, query: str, k: int = TOP_K_INITIAL) -> List[Dict]:
        """Perform keyword-based search."""
        logger.debug(f"Performing keyword search with k={k}")

        if self.metadata is None or len(self.metadata) == 0:
            logger.error("Metadata is not loaded or empty for keyword search")
            return []

        keywords = self.extract_keywords(query)
        logger.debug(f"Extracted keywords: {keywords}")

        results = []
        for i, entry in enumerate(self.metadata):
            if isinstance(entry, dict) and "text" in entry:
                text = entry.get("text", "").lower()
                matches = sum(1 for keyword in keywords if keyword in text)
                if matches > 0:
                    score = (
                        matches / (len(keywords) + 1) *
                        (1 + matches / len(text.split()))
                    )
                    source_path = entry.get("source_path", "")
                    results.append({
                        "chunk_id": i,
                        "score": score,
                        "text": entry.get("text", ""),
                        "source": Citation(source_path).to_dict(),
                        "search_type": "keyword",
                        "keyword_matches": matches
                    })

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
        logger.debug(f"Performing contextual filtering on {len(candidates)} candidates")
        if not candidates:
            return []

        # Simple contextual filtering based on shared keywords
        query_keywords = set(self.extract_keywords(query))
        if not query_keywords:
            return candidates

        for candidate in candidates:
            text_keywords = set(self.extract_keywords(candidate['text']))
            shared_keywords = query_keywords.intersection(text_keywords)
            candidate['contextual_score'] = len(shared_keywords)

        # Sort by contextual score
        candidates.sort(key=lambda x: x['contextual_score'], reverse=True)
        return candidates

    def generate_enhanced_prompt(
        self, query: str, chunks: List[Dict]
    ) -> str:
        """
        Generate an enhanced prompt for the LLM with citations.
        """
        context = ""
        citation_refs = []
        
        # Estimate token count (rough approximation: 1 token ≈ 4 characters)
        estimated_tokens = 0
        max_context_tokens = 1500  # Leave room for query, instructions, and response
        
        for i, chunk in enumerate(chunks):
            chunk_text = chunk['text']
            chunk_tokens = len(chunk_text) // 4  # Rough token estimation
            
            # Check if adding this chunk would exceed the limit
            if estimated_tokens + chunk_tokens > max_context_tokens:
                logger.warning(f"Truncating context at chunk {i+1} to stay within token limits")
                break
                
            source = chunk.get('source', {})
            citation_id = source.get('id', f'source_{i+1}')
            citation_name = source.get('name', 'Unknown Source')
            # Create a unique reference tag like [1], [2]
            ref = f"[{i+1}]"
            context += f"Source {ref}:\n{chunk_text}\n\n"
            estimated_tokens += chunk_tokens
            
            # Store the full citation details for later lookup
            if citation_id not in [c['id'] for c in citation_refs]:
                citation_refs.append({
                    'ref': ref, 'id': citation_id, 'name': citation_name
                })

        # Sort citations by their reference number
        citation_refs.sort(key=lambda x: int(x['ref'][1:-1]))

        # Create the citation list string
        citations_str = "\n".join(
            [f"{c['ref']}: {c['name']}" for c in citation_refs]
        )

        prompt = (
            "Please answer the following query based on the provided sources."
            "\nUse the sources to provide a comprehensive and accurate answer."
            "\n**IMPORTANT**: You MUST cite the sources you use in your "
            "response using the format [1], [2], etc. at the end of each "
            "sentence."
            "\n\n---"
            f"\n**QUERY**: {query}"
            f"\n\n---"
            f"\n**SOURCES**:\n{context}"
            f"\n\n---"
            f"\n**CITATION LIST**:\n{citations_str}"
            f"\n\n---"
            "\n**ANSWER** (with citations):"
        )
        
        # Log estimated token count
        total_estimated_tokens = len(prompt) // 4
        logger.info(f"Generated prompt with ~{total_estimated_tokens} tokens")
        
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