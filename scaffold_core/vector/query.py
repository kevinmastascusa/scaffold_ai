import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

logger.debug("Starting query module initialization...")

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

# Constants
KEYWORD_FALLBACK = "environmental sustainability assessment case study"
KEYWORD_DOC = "assessment framework for environmental sustainability"

def embed_query(text):
    """Embed the query text using sentence transformers."""
    logger.debug(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    logger.debug(f"Embedding query text: {text[:100]}...")
    emb = model.encode([text], show_progress_bar=False)
    return np.array(emb).astype("float32")

def load_metadata(path):
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

def rerank_with_llm(query: str, chunks: list) -> str:
    """
    Rerank and filter chunks using the Hugging Face LLM.
    
    Args:
        query: The user's query
        chunks: List of text chunks to analyze
        
    Returns:
        Filtered and formatted answer from the LLM
    """
    logger.debug(f"Starting LLM reranking with {len(chunks)} chunks")
    
    # Construct the prompt
    prompt = f"""You are an expert assistant. Given a query and a set of text chunks, your task is to:
1) Carefully read the provided chunks and use them as the main source of information.
2) If the answer is directly stated or can be reasonably inferred or synthesized from the chunks, provide a concise, well-structured answer in your own words.
3) Always cite the most relevant chunk(s) by number (e.g., Chunk 3) or by source path if possible, to support your answer.
4) Only respond with 'Not found in the retrieved documents.' if there is truly no relevant information in any chunk.

Query: {query}

Here are the {len(chunks)} candidate chunks:\n\n"""
    
    # Add chunks to prompt
    for i, chunk in enumerate(chunks, start=1):
        snippet = chunk["text"].replace("\n", " ")
        prompt += f"{i}.) {snippet[:500]}…\nSource: {chunk.get('source','')}\n\n"

    logger.debug("Constructed prompt for LLM reranking")

    # Generate response
    try:
        logger.debug("Generating LLM response...")
        response = llm.generate_response(
            prompt,
            temperature=LLM_TEMPERATURE  # Use config temperature
        )
        logger.debug("Successfully generated LLM response")
        return response
    except Exception as e:
        logger.error(f"Error during LLM response generation: {str(e)}")
        raise

def main():
    """Main query function."""
    logger.info("Starting query process...")
    
    print("\nScaffold AI Query System")
    print("=" * 50)
    print("Loading models and initializing components...")
    
    try:
        # Pre-load models to catch any initialization errors
        logger.debug("Pre-loading embedding model...")
        model = SentenceTransformer(EMBEDDING_MODEL)
        logger.debug("Pre-loading cross-encoder model...")
        cross = CrossEncoder(CROSS_ENCODER_MODEL)
        logger.debug("Models loaded successfully")
        
        while True:
            print("\nEnter your query (or 'exit' to quit):")
            q = input("> ").strip()
            
            if q.lower() in ('exit', 'quit'):
                print("Goodbye!")
                break
                
            if not q:
                print("Please enter a query.")
                continue

            try:
                # Embed query
                logger.debug("Embedding query...")
                q_emb = embed_query(q)
                logger.debug("Query embedding complete")

                # Load index and metadata
                logger.debug("Loading FAISS index...")
                index_path = get_faiss_index_path()
                logger.debug(f"Using index path: {index_path}")
                index = faiss.read_index(str(index_path))
                logger.debug("FAISS index loaded successfully")
                
                meta = load_metadata(get_metadata_json_path())

                # Initial vector search
                logger.debug(f"Performing initial vector search with k={TOP_K_INITIAL}")
                distances, indices = index.search(q_emb, k=TOP_K_INITIAL)
                vector_ids = [int(i) for i in indices[0]]
                vector_scores = {int(i): float(s) for s, i in zip(distances[0], indices[0])}
                logger.debug(f"Found {len(vector_ids)} initial matches")

                # Add fallback results
                logger.debug("Adding fallback results...")
                fallback_ids = []
                for i, entry in enumerate(meta):
                    if isinstance(entry, dict) and "text" in entry:
                        if KEYWORD_FALLBACK in entry["text"].lower():
                            fallback_ids.append(i)
                logger.debug(f"Found {len(fallback_ids)} fallback matches")

                # Combine and deduplicate results
                combined_ids = []
                for cid in vector_ids + fallback_ids:
                    if cid not in combined_ids:
                        combined_ids.append(cid)
                combined_ids = combined_ids[:TOP_K_INITIAL]
                logger.debug(f"Combined and deduplicated to {len(combined_ids)} results")

                # Prepare candidates
                candidates = []
                for cid in combined_ids:
                    if cid < len(meta):
                        entry = meta[cid]
                        if isinstance(entry, dict):
                            candidates.append({
                                "chunk_id": cid,
                                "score": vector_scores.get(cid, 0.0),
                                "text": entry.get("text", ""),
                                "source": entry.get("source_path", "")
                            })

                # Show initial results
                top_initial = sorted(candidates, key=lambda x: -x["score"])[:TOP_K_INITIAL]
                print(f"\n=== TOP {TOP_K_INITIAL} RAW CHUNKS ===\n")
                for i, r in enumerate(top_initial, 1):
                    snippet = r["text"].replace("\n", " ")
                    print(f"--- #{i} (score={r['score']:.4f}) {r['source']} ---")
                    print(snippet[:500] + ("…" if len(r["text"]) > 500 else ""))
                    print()

                # Cross-encoder reranking
                logger.debug(f"Starting cross-encoder reranking with model: {CROSS_ENCODER_MODEL}")
                pairs = [[q, c["text"]] for c in candidates]
                logger.debug(f"Reranking {len(pairs)} pairs")
                cross_scores = cross.predict(pairs)
                for c, s in zip(candidates, cross_scores):
                    c["cross_score"] = float(s)
                candidates = sorted(candidates, key=lambda x: -x["cross_score"])[:TOP_K_FINAL]
                logger.debug(f"Reranking complete, selected top {TOP_K_FINAL} candidates")

                # Final LLM filtering and answer generation
                logger.debug("Starting final LLM filtering...")
                answer = rerank_with_llm(q, candidates)
                print("\n=== LLM-FILTERED ANSWER ===\n")
                print(answer)
                
            except Exception as e:
                logger.error(f"Error processing query: {str(e)}")
                print(f"\nError: {str(e)}")
                print("Please try again with a different query.")
                
    except Exception as e:
        logger.error(f"Error initializing query system: {str(e)}")
        print(f"\nFailed to initialize query system: {str(e)}")
        return

if __name__ == "__main__":
    main()
