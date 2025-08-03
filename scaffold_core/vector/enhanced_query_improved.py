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

# Constants
TOP_K_INITIAL = 50
TOP_K_FINAL = 5  # Increased from 3 to 5 to provide more context
MIN_CROSS_SCORE = -5.0  # Relaxed from -2.0 to keep more sources
MIN_CONTEXTUAL_SCORE = 0  # Relaxed from 1 to keep more sources
MAX_MEMORY_MESSAGES = 4  # Increased from 2 to provide better context
MAX_MEMORY_TOKENS = 800  # Increased from 400 to allow more conversation history
MAX_CONTEXT_TOKENS = 800  # Increased from 300 to allow more source context
MAX_TOTAL_TOKENS = 3000  # Increased from 1000 to allow longer responses

# Configurable main prompt for testing different prompt variations
MAIN_PROMPT = "You are Scaffold AI, a course curriculum assistant helping students and educators. Answer this question comprehensively using the provided sources. Focus on educational value, practical insights, and clear explanations. If the sources don't fully address the question, acknowledge this and provide the best available information."

# Configurable minimal prompt for fallback scenarios
MINIMAL_PROMPT = "You are Scaffold AI, a course curriculum assistant. Answer this question directly and clearly using the available information:"

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
        """Get conversation context for the given session, with strict token management."""
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
                    
                    # Build context with strict token management
                    return self._build_context_with_tokens(syllabus_context, recent_messages)
            except Exception as e:
                logger.debug(f"Could not load external conversation file: {e}")
            
            return ""
        
        messages = self.conversation_memory[session_id]
        if not messages:
            return ""
        
        # Extract syllabus context and recent messages
        syllabus_context = ""
        recent_messages = []
        
        for msg in messages[-MAX_MEMORY_MESSAGES:]:
            role = msg.get('type', 'user')
            content = msg.get('content', '')
            
            if role == 'syllabus_context':
                syllabus_context = content
            elif role in ['user', 'assistant']:
                recent_messages.append(msg)
        
        return self._build_context_with_tokens(syllabus_context, recent_messages)
    
    def _build_context_with_tokens(self, syllabus_context: str, messages: List[Dict]) -> str:
        """Build conversation context with strict token management to prevent overflow."""
        context_parts = []
        total_tokens = 0
        max_tokens = MAX_MEMORY_TOKENS
        
        # Add syllabus context first (if available and not too long)
        if syllabus_context:
            syllabus_tokens = len(syllabus_context) // 4
            if syllabus_tokens <= max_tokens // 3:  # Use max 1/3 of tokens for syllabus
                context_parts.append(syllabus_context)
                total_tokens += syllabus_tokens
                max_tokens -= syllabus_tokens
            else:
                # Truncate syllabus if too long
                max_syllabus_chars = (max_tokens // 3) * 4
                truncated_syllabus = syllabus_context[:max_syllabus_chars] + "..."
                context_parts.append(truncated_syllabus)
                total_tokens += max_tokens // 3
        
        # Add recent messages with strict token control
        for msg in reversed(messages):  # Start with most recent
            role = msg.get('type', 'user')
            content = msg.get('content', '')
            
            # Estimate tokens for this message
            message_tokens = len(content) // 4
            
            # Check if adding this message would exceed limit
            if total_tokens + message_tokens > max_tokens:
                logger.warning(f"Conversation context truncated: {total_tokens} tokens used")
                break
            
            # Add message to context
            if role == 'user':
                context_parts.append(f"User: {content}")
            elif role == 'assistant':
                context_parts.append(f"Assistant: {content}")
            
            total_tokens += message_tokens
        
        # Reverse to get chronological order
        context_parts.reverse()
        
        conversation_context = "\n".join(context_parts)
        logger.debug(f"Generated conversation context with ~{total_tokens} tokens")
        
        return conversation_context
    
    def clear_memory(self, session_id: str):
        """Clear conversation memory for a session."""
        try:
            # Clear from internal memory
            if session_id in self.conversation_memory:
                del self.conversation_memory[session_id]
                logger.info(f"Cleared internal conversation memory for session {session_id}")
            
            # Also try to clear external conversation file
            try:
                import json
                from pathlib import Path
                conversations_dir = Path("conversations")
                conversation_file = conversations_dir / f"{session_id}.json"
                
                if conversation_file.exists():
                    conversation_file.unlink()
                    logger.info(f"Cleared external conversation file for session {session_id}")
                    
            except Exception as e:
                logger.debug(f"Could not clear external conversation file: {e}")
                
        except Exception as e:
            logger.error(f"Error in clear_memory for session {session_id}: {e}")
            # Don't raise to prevent system crashes

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
                
                # Only filter out candidates with very low scores (relaxed threshold)
                if candidate["cross_score"] < MIN_CROSS_SCORE:
                    candidate["filtered_out"] = True
                    logger.debug(f"Filtered out candidate with cross_score: {candidate['cross_score']}")
            
            # Remove filtered candidates
            candidates = [c for c in candidates if not c.get("filtered_out", False)]
            
            # Sort by cross-encoder score
            candidates.sort(key=lambda x: x["cross_score"], reverse=True)
            
            logger.debug(f"Cross-encoder reranking: {len(candidates)} candidates remaining after filtering")
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
                
                # Only filter out candidates with very low contextual scores (relaxed threshold)
                if candidate['contextual_score'] < MIN_CONTEXTUAL_SCORE:
                    candidate["filtered_out"] = True
                    logger.debug(f"Filtered out candidate with contextual_score: {candidate['contextual_score']}")
            
            # Remove filtered candidates
            candidates = [c for c in candidates if not c.get("filtered_out", False)]
            
            # Sort by contextual score
            candidates.sort(key=lambda x: x['contextual_score'], reverse=True)
            
            logger.debug(f"Contextual filtering: {len(candidates)} candidates remaining after filtering")
            return candidates
            
        except Exception as e:
            logger.error(f"Error in contextual filtering: {e}")
            return candidates
    
    def generate_improved_prompt(self, query: str, chunks: List[Dict], conversation_context: str = "") -> str:
        """Generate an improved prompt with better token management to prevent overflow."""
        if not chunks:
            return f"Query: {query}\n\nI don't have enough relevant information to answer this query accurately."
        
        # Calculate available tokens for context (reserve space for prompt template)
        # Use more conservative limits to prevent model overflow
        max_total_tokens = min(MAX_TOTAL_TOKENS, 1500)  # Cap at 1500 to prevent overflow
        prompt_template_tokens = 200  # Estimate for prompt template
        available_tokens = max_total_tokens - prompt_template_tokens
        
        # Format conversation context with token limit
        context_section = ""
        if conversation_context:
            context_tokens = len(conversation_context.split())
            if context_tokens > available_tokens // 3:  # Use max 1/3 for conversation
                # Truncate conversation context
                words = conversation_context.split()[:available_tokens // 3]
                conversation_context = ' '.join(words) + "..."
            context_section = f"\nPrevious Conversation Context:\n{conversation_context}\n"
            available_tokens -= len(conversation_context.split())
        
        # Format chunks with strict token budget to prevent overflow
        # Use fewer chunks but ensure we stay within token limits
        max_chunks_for_model = 4  # Reduced from 8 to prevent overflow
        formatted_chunks = self.format_chunks_for_prompt(chunks, max_chunks=max_chunks_for_model, max_tokens=available_tokens)
            
        # Build the prompt with improved instructions
        prompt = f"""<s>[INST] {MAIN_PROMPT}

Question: {query}
{context_section}
Sources:
{formatted_chunks} [/INST]"""
        
        # Log token usage for debugging
        total_tokens = len(prompt.split())
        logger.debug(f"Generated prompt with ~{total_tokens} tokens")
        
        # Final validation to prevent token overflow
        if total_tokens > MAX_TOTAL_TOKENS:
            logger.warning(f"Prompt too long ({total_tokens} tokens), truncating")
            # Truncate prompt to fit within limits
            words = prompt.split()[:MAX_TOTAL_TOKENS]
            prompt = ' '.join(words) + "..."
            logger.debug(f"Truncated prompt to ~{len(words)} tokens")
        
        return prompt

    def _prepare_context(self, chunks: List[Dict], max_tokens: int = 800) -> str:
        """Prepare context from chunks with token limit."""
        context_parts = []
        total_tokens = 0
        
        for chunk in chunks:
            chunk_text = chunk.get('text', '').strip()
            if not chunk_text:
                continue
                
            # Estimate tokens (rough word count)
            chunk_tokens = len(chunk_text.split())
            if total_tokens + chunk_tokens > max_tokens:
                # Truncate chunk to fit
                words = chunk_text.split()
                allowed_words = max_tokens - total_tokens
                chunk_text = ' '.join(words[:allowed_words])
            
            context_parts.append(chunk_text)
            total_tokens += chunk_tokens
            
            if total_tokens >= max_tokens:
                break
                
        return "\n\n".join(context_parts)
    
    def format_chunks_for_prompt(self, chunks: List[Dict], max_chunks: int = 8, max_tokens: Optional[int] = None) -> str:
        """Format chunks for the prompt, limiting length and number with token management."""
        formatted_chunks = []
        total_tokens = 0
        
        for i, chunk in enumerate(chunks[:max_chunks]):
            chunk_text = chunk.get('text', '').strip()
            if not chunk_text:
                continue
            
            # Estimate tokens for this chunk
            chunk_tokens = len(chunk_text.split())
            
            # Check token limit if specified
            if max_tokens and total_tokens + chunk_tokens > max_tokens:
                # Truncate chunk to fit remaining token budget
                remaining_tokens = max_tokens - total_tokens
                words = chunk_text.split()[:remaining_tokens]
                chunk_text = ' '.join(words) + "..."
                chunk_tokens = len(words)
            
            # Truncate chunk if too long (150 words max per chunk - reduced to prevent overflow)
            if chunk_tokens > 150:
                words = chunk_text.split()[:150]
                chunk_text = ' '.join(words) + "..."
                chunk_tokens = 150
            
            formatted_chunks.append(chunk_text)
            total_tokens += chunk_tokens
            
            # Stop if we've reached token limit
            if max_tokens and total_tokens >= max_tokens:
                break
        
        return "\n\n".join(formatted_chunks)
    
    def _is_response_garbled(self, response: str) -> bool:
        """Check if response contains garbled or corrupted text."""
        if not response or not isinstance(response, str):
            return True
        
        # Check for repetitive patterns
        words = response.split()
        if len(words) < 10:
            return False
        
        # Check for excessive repetition
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
            if word_counts[word] > len(words) * 0.3:  # More than 30% repetition
                return True
        
        # Check for garbled text patterns
        garbled_patterns = [
            r'[A-Za-z]{20,}',  # Very long words
            r'[^\w\s\.\,\!\?]{3,}',  # Multiple special characters
            r'(.)\1{5,}',  # Repeated characters
            r'[A-Z]{10,}',  # All caps words
        ]
        
        for pattern in garbled_patterns:
            if re.search(pattern, response):
                return True
        
        return False
    
    def _generate_minimal_prompt(self, query: str, chunks: List[Dict]) -> str:
        """Generate a minimal prompt with very limited context."""
        context = ""
        if chunks:
            chunk_text = chunks[0].get('text', '')[:300]  # Limit to 300 chars
            context = f"\nRelevant information: {chunk_text}"
        
        return f"""<s>[INST] {MINIMAL_PROMPT}

{query}{context} [/INST]"""
    
    def _generate_fallback_response(self, query: str) -> str:
        """Generate a fallback response when all else fails."""
        return f"I apologize, but I'm having trouble generating a comprehensive response for your question about '{query}'. This could be due to limited relevant information in the available sources or technical issues. Please try rephrasing your question, asking about a different topic, or contact support if the issue persists."
    
    def query(self, query: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Process a query and return relevant results with improved prompt engineering."""
        if not self.initialized:
            self.initialize()
        
        # Input validation to prevent garbled text
        if not query or not isinstance(query, str):
            return {
                "error": "Invalid query provided",
                "sources": [],
                "response": "Please provide a valid question or query."
            }
        
        # Sanitize query to prevent encoding issues
        query = query.strip()
        if len(query) > 1000:  # Limit query length
            query = query[:1000] + "..."
        
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
            
            # Use all filtered candidates instead of limiting to top 5
            final_candidates = filtered_candidates
            logger.debug(f"Using all {len(final_candidates)} filtered candidates")
            
            # Generate improved prompt
            improved_prompt = self.generate_improved_prompt(
                query, final_candidates, conversation_context
            )
            
            # Get temperature from config
            from scaffold_core.config import LLM_TEMPERATURE
            temperature = LLM_TEMPERATURE
            
            # Generate response using LLM with comprehensive error handling
            llm_response = ""
            try:
                # First attempt with full context
                llm_response = llm.generate_response(improved_prompt, temperature=temperature)
                
                # Validate response quality
                if self._is_response_garbled(llm_response):
                    logger.warning("Detected garbled response, retrying with reduced context")
                    raise ValueError("Garbled response detected")
                    
            except Exception as e:
                logger.error(f"Error in LLM response generation: {e}")
                
                # Try with reduced context
                try:
                    logger.warning("Retrying with minimal context")
                    minimal_prompt = self._generate_minimal_prompt(query, final_candidates[:1])
                    llm_response = llm.generate_response(minimal_prompt, temperature=temperature)
                    
                    # Validate response again
                    if self._is_response_garbled(llm_response):
                        logger.error("Still getting garbled response, using fallback")
                        llm_response = self._generate_fallback_response(query)
                        
                except Exception as e2:
                    logger.error(f"Second attempt failed: {e2}")
                    llm_response = self._generate_fallback_response(query)
            
            # Validate and truncate response if necessary
            if len(llm_response) > 8000:  # Increased from 2000 to allow longer responses
                llm_response = llm_response[:8000] + "..."
                logger.warning("Response truncated due to length limit (8000 chars)")
            
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
    try:
        if not session_id:
            logger.warning("No session_id provided for memory clearing")
            return
        
        # Check if there's actually memory to clear (without full initialization)
        internal_memory_exists = False
        external_file_exists = False
        
        # Check internal memory if system is initialized
        try:
            if improved_enhanced_query_system.initialized:
                internal_memory_exists = session_id in improved_enhanced_query_system.conversation_memory
        except Exception:
            pass
        
        # Check external conversation file
        try:
            from pathlib import Path
            conversations_dir = Path("conversations")
            conversation_file = conversations_dir / f"{session_id}.json"
            external_file_exists = conversation_file.exists()
        except Exception:
            pass
        
        if not internal_memory_exists and not external_file_exists:
            logger.info(f"No memory found to clear for session: {session_id}")
            return
        
        # Clear memory if system is initialized
        try:
            if improved_enhanced_query_system.initialized:
                improved_enhanced_query_system.clear_memory(session_id)
                logger.info(f"Successfully cleared memory for session: {session_id}")
        except Exception as e:
            logger.debug(f"Could not clear internal memory: {e}")
        
    except Exception as e:
        logger.error(f"Error clearing conversation memory for session {session_id}: {e}")
        # Don't raise the exception to prevent system crashes

def get_conversation_memory(session_id: str) -> List[Dict]:
    """Get conversation memory for a session."""
    try:
        if not session_id:
            return []
        
        # Ensure the system is initialized
        if not improved_enhanced_query_system.initialized:
            logger.info("Initializing enhanced query system for memory retrieval")
            improved_enhanced_query_system.initialize()
        
        if session_id in improved_enhanced_query_system.conversation_memory:
            return improved_enhanced_query_system.conversation_memory[session_id]
        return []
        
    except Exception as e:
        logger.error(f"Error getting conversation memory for session {session_id}: {e}")
        return []

def clear_all_conversation_memory():
    """Clear all conversation memory for debugging."""
    try:
        # Check if there's actually memory to clear (without full initialization)
        internal_memory_count = 0
        external_file_count = 0
        
        # Check internal memory if system is initialized
        try:
            if improved_enhanced_query_system.initialized:
                internal_memory_count = len(improved_enhanced_query_system.conversation_memory)
        except Exception:
            pass
        
        # Check external conversation files
        try:
            from pathlib import Path
            conversations_dir = Path("conversations")
            if conversations_dir.exists():
                external_file_count = len(list(conversations_dir.glob("*.json")))
        except Exception:
            pass
        
        if internal_memory_count == 0 and external_file_count == 0:
            logger.info("No conversation memory found to clear")
            return
        
        # Clear internal memory if system is initialized
        try:
            if improved_enhanced_query_system.initialized:
                improved_enhanced_query_system.conversation_memory.clear()
                logger.info(f"Successfully cleared {internal_memory_count} internal conversation memory entries")
        except Exception as e:
            logger.debug(f"Could not clear internal memory: {e}")
        
        # Clear external conversation files
        try:
            from pathlib import Path
            conversations_dir = Path("conversations")
            if conversations_dir.exists():
                cleared_files = 0
                for conversation_file in conversations_dir.glob("*.json"):
                    conversation_file.unlink()
                    cleared_files += 1
                logger.info(f"Successfully cleared {cleared_files} external conversation files")
        except Exception as e:
            logger.debug(f"Could not clear external conversation files: {e}")
        
    except Exception as e:
        logger.error(f"Error clearing all conversation memory: {e}") 