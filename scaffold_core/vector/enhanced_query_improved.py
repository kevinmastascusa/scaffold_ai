"""
Improved Enhanced Query System for Scaffold AI
Better prompt engineering and citation handling for stable responses.
Now with chat memory support.
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
from scaffold_core.config_manager import ConfigManager

# Initialize config manager
config_manager = ConfigManager()

# Constants
TOP_K_INITIAL = 50
TOP_K_FINAL = 5
MIN_CROSS_SCORE = -2.0  # Minimum cross-encoder score threshold
MIN_CONTEXTUAL_SCORE = 1  # Minimum contextual score threshold
MAX_MEMORY_MESSAGES = 10  # Maximum number of previous messages to include
MAX_MEMORY_TOKENS = 1500  # Maximum tokens for conversation history

class ImprovedEnhancedQuerySystem:
    """Improved enhanced query system with better prompt engineering and chat memory."""
    
    def __init__(self):
        self.initialized = False
        self.embedding_model = None
        self.cross_encoder = None
        self.faiss_index = None
        self.metadata = []
        self.conversation_memory = {}  # Store conversation history by session_id
        
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
    
    def add_to_memory(self, session_id: str, message: Dict[str, Any]):
        """Add a message to conversation memory."""
        if session_id not in self.conversation_memory:
            self.conversation_memory[session_id] = []
        
        # Add message with timestamp
        message_with_timestamp = {
            **message,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        self.conversation_memory[session_id].append(message_with_timestamp)
        
        # Keep only the last MAX_MEMORY_MESSAGES messages
        if len(self.conversation_memory[session_id]) > MAX_MEMORY_MESSAGES:
            self.conversation_memory[session_id] = self.conversation_memory[session_id][-MAX_MEMORY_MESSAGES:]
        
        logger.debug(f"Added message to memory for session {session_id}. Total messages: {len(self.conversation_memory[session_id])}")
    
    def get_conversation_context(self, session_id: str) -> str:
        """Get conversation context for the given session, including syllabus context."""
        if session_id not in self.conversation_memory:
            # Try to load from external conversation files (for app_enhanced.py compatibility)
            try:
                import json
                from pathlib import Path
                conversations_dir = Path("conversations")
                conversation_file = conversations_dir / f"{session_id}.json"
                
                if conversation_file.exists():
                    with open(conversation_file, 'r') as f:
                        external_messages = json.load(f)
                    
                    # Look for syllabus context in external messages
                    syllabus_context = ""
                    recent_messages = []
                    
                    for msg in external_messages:
                        if msg.get('type') == 'syllabus_context':
                            syllabus_context = msg.get('content', '')
                        elif msg.get('type') in ['user', 'assistant'] and len(recent_messages) < MAX_MEMORY_MESSAGES:
                            recent_messages.append(msg)
                    
                    # Build context with syllabus info
                    context_parts = []
                    if syllabus_context:
                        context_parts.append(syllabus_context)
                    
                    for msg in recent_messages[-MAX_MEMORY_MESSAGES:]:
                        role = msg.get('type', 'user')
                        content = msg.get('content', '')
                        if role == 'user':
                            context_parts.append(f"User: {content}")
                        elif role == 'assistant':
                            context_parts.append(f"Assistant: {content}")
                    
                    return "\n".join(context_parts)
            except Exception as e:
                logger.debug(f"Could not load external conversation file: {e}")
            
            return ""
        
        messages = self.conversation_memory[session_id]
        if not messages:
            return ""
        
        # Build conversation context
        context_parts = []
        total_tokens = 0
        
        for msg in messages[-MAX_MEMORY_MESSAGES:]:  # Get last N messages
            role = msg.get('type', 'user')
            content = msg.get('content', '')
            
            if role == 'user':
                context_parts.append(f"User: {content}")
            elif role == 'assistant':
                context_parts.append(f"Assistant: {content}")
            elif role == 'syllabus_context':
                # Include syllabus context at the beginning
                context_parts.insert(0, content)
            
            # Estimate tokens (rough approximation)
            message_tokens = len(content) // 4
            total_tokens += message_tokens
            
            # Stop if we exceed token limit
            if total_tokens > MAX_MEMORY_TOKENS:
                logger.warning(f"Conversation context truncated due to token limit")
                break
        
        conversation_context = "\n".join(context_parts)
        logger.debug(f"Generated conversation context with ~{total_tokens} tokens")
        
        return conversation_context
    
    def clear_memory(self, session_id: str):
        """Clear conversation memory for a session."""
        if session_id in self.conversation_memory:
            del self.conversation_memory[session_id]
            logger.info(f"Cleared conversation memory for session {session_id}")

    def extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text."""
        # Remove punctuation and convert to lowercase
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        
        # Split into words and filter
        words = text.split()
        
        # Filter out common stop words and short words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
        
        keywords = [word for word in words if len(word) > 2 and word not in stop_words]
        
        return keywords[:20]  # Limit to top 20 keywords
    
    def semantic_search(self, query: str, k: int = TOP_K_INITIAL) -> List[Dict]:
        """Perform semantic search using sentence transformers."""
        if not self.initialized or self.embedding_model is None or self.faiss_index is None:
            return []
            
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
        if not candidates or not self.initialized or self.cross_encoder is None:
            return candidates
        
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
    
    def generate_improved_prompt(self, query: str, chunks: List[Dict], conversation_context: str = "") -> str:
        """Generate an improved prompt with better engineering and conversation context."""
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
        
        # Build conversation context section
        conversation_section = ""
        if conversation_context:
            conversation_section = f"""
CONVERSATION HISTORY:
{conversation_context}

"""
        
        # Improved prompt template with conversation context
        prompt = f"""You are a helpful AI assistant that provides accurate, relevant, and well-cited responses based on the provided sources. You have access to the current conversation history to provide context-aware responses.

TASK: Answer the following query using ONLY the information from the provided sources. Consider the conversation history for context and continuity.

{conversation_section}QUERY: {query}

SOURCES:
{context}

CITATION LIST:
{citations_str}

INSTRUCTIONS:
1. Answer the query comprehensively using information from the sources
2. Use specific details and examples from the sources
3. Cite sources using [1], [2], etc. format at the end of relevant sentences
4. Consider the conversation history for context and continuity
5. If referring to previous parts of the conversation, acknowledge them naturally
6. Avoid repetition and stay focused on the query
7. If the sources don't contain enough information, say so clearly
8. Write in a clear, professional tone
9. Keep the response concise but complete

ANSWER:"""
        
        total_tokens = len(prompt) // 4
        logger.info(f"Generated improved prompt with ~{total_tokens} tokens (including conversation context)")
        
        return prompt
    
    def query(self, query: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Process a query and return relevant results with improved prompt engineering."""
        if not self.initialized:
            self.initialize()
        
        try:
            # Get conversation context if session_id provided
            conversation_context = ""
            if session_id:
                conversation_context = self.get_conversation_context(session_id)
                logger.debug(f"Generated conversation context with ~{len(conversation_context.split())} tokens")
            
            # Log query processing
            logger.info(f"Processing query: {query}" + (f" (session: {session_id})" if session_id else ""))
            
            # Get initial candidates using hybrid search
            candidates = self.hybrid_search(query)
            logger.debug(f"Found {len(candidates)} initial candidates")
            
            if not candidates:
                return {
                    "response": "I couldn't find any relevant information to answer your question.",
                    "sources": []
                }
            
            # Rerank candidates using cross-encoder
            reranked_candidates = self.cross_encoder_rerank(query, candidates)
            logger.debug(f"Reranked to {len(reranked_candidates)} candidates")
            
            # Apply contextual filtering
            filtered_candidates = self.contextual_filtering(query, reranked_candidates)
            logger.debug(f"Filtered to {len(filtered_candidates)} candidates")
            
            # Select top candidates
            final_candidates = filtered_candidates[:TOP_K_FINAL]
            logger.debug(f"Selected {len(final_candidates)} final candidates")
            
            # Generate improved prompt
            improved_prompt = self.generate_improved_prompt(
                query, final_candidates, conversation_context
            )
            logger.info(f"Generated improved prompt with ~{len(improved_prompt.split())} tokens (including conversation context)")
            
            # Get temperature from config manager
            temperature = config_manager.get_model_settings('llm').get('temperature', 0.3)
            
            # Generate response using LLM
            llm_response = llm.generate_response(improved_prompt, temperature=temperature)
            
            # Store query and response in memory if session_id provided
            if session_id:
                self.add_to_memory(session_id, {"role": "user", "content": query})
                self.add_to_memory(session_id, {"role": "assistant", "content": llm_response})
            
            return {
                "response": llm_response,
                "sources": [
                    {
                        "score": candidate.get("cross_score", 0),
                        "source": {
                            "id": candidate.get("document_id", "Unknown"),
                            "name": candidate.get("metadata", {}).get("filename", "Unknown Source"),
                            "path": candidate.get("source_path", ""),
                            "page": candidate.get("metadata", {}).get("page_number", "")
                        },
                        "text_preview": candidate.get("text", "")[:200] + "..."
                    }
                    for candidate in final_candidates
                ],
                "conversation_context": conversation_context if session_id else None
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "response": "Sorry, I encountered an error while processing your query.",
                "sources": [],
                "error": str(e)
            }

# Global instance
improved_enhanced_query_system = ImprovedEnhancedQuerySystem()

def query_enhanced_improved(query: str, session_id: Optional[str] = None) -> Dict[str, Any]:
    """Convenience function for improved enhanced querying with chat memory."""
    return improved_enhanced_query_system.query(query, session_id)

def clear_conversation_memory(session_id: str):
    """Clear conversation memory for a session."""
    improved_enhanced_query_system.clear_memory(session_id)

def get_conversation_memory(session_id: str) -> List[Dict]:
    """Get conversation memory for a session."""
    if session_id in improved_enhanced_query_system.conversation_memory:
        return improved_enhanced_query_system.conversation_memory[session_id]
    return [] 