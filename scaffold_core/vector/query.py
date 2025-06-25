import os
import json
import numpy as np
import faiss
import requests
from sentence_transformers import SentenceTransformer, CrossEncoder
import sys

# Add the parent directory to sys.path to import from main.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vector.main import (
    FAISS_INDEX_PATH,
    PROCESSED_JSON_PATH,
    EMBEDDING_MODEL_NAME,
    OLLAMA_MODEL,
    OLLAMA_ENDPOINT,
)

NUM_CANDIDATES       = 50   
KEYWORD_FALLBACK     = "environmental sustainability assessment case study"
KEYWORD_DOC          = "assessment framework for environmental sustainability"
CROSS_ENCODER_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CROSS_TOPK           = 10
cross = CrossEncoder(CROSS_ENCODER_MODEL)

def embed_query(text):
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    emb = model.encode([text], show_progress_bar=False)
    return np.array(emb).astype("float32")

def load_metadata(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def rerank_with_ollama(query, chunks):

    system_msg = {
        "role": "system",
        "content": (
            "You are a fact‐checking assistant. You will be given a user query "
            "and a list of text chunks. Your job is:\n"
            " 1) Discard any chunk that does not explicitly mention the details requested by the query.\n"
            " 2) If, after filtering, no chunk contains the requested details, respond exactly with "
            "“Not found in the retrieved documents.”\n"
            " 3) Otherwise, answer the user’s query by quoting directly from the remaining chunks, "
            "and do not add any information not present."
        )
    }
    user_msg = {
        "role": "user",
        "content": f"Query:\n\n{query}\n\nHere are the {len(chunks)} candidate chunks:"
    }

    chunk_msgs = []
    for i, c in enumerate(chunks, start=1):
        snippet = c["text"].replace("\n", " ")
        chunk_msgs.append({
            "role": "user",
            "content": f"{i}.) {snippet[:500]}…"
        })

    resp = requests.post(
        OLLAMA_ENDPOINT,
        headers={"Content-Type": "application/json"},
        json={
            "model": OLLAMA_MODEL,
            "messages": [system_msg, user_msg] + chunk_msgs,
            "stream": False
        }
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Ollama request failed [{resp.status_code}]:\n{resp.text}")

    return resp.json()["choices"][0]["message"]["content"]

def main():
    q = input("Enter your query: ").strip()
    if not q:
        print("No query entered. Exiting.")
        return

    q_emb = embed_query(q)

    index = faiss.read_index(FAISS_INDEX_PATH)
    meta  = load_metadata(PROCESSED_JSON_PATH)

    distances, indices = index.search(q_emb, k=NUM_CANDIDATES)
    vector_ids    = [int(i) for i in indices[0]]
    vector_scores = {int(i): float(s) for s, i in zip(distances[0], indices[0])}

    fallback_ids = [
        entry["chunk_id"]
        for entry in meta
        if KEYWORD_FALLBACK in entry["text"].lower()
    ]

    combined_ids = []
    for cid in vector_ids + fallback_ids:
        if cid not in combined_ids:
            combined_ids.append(cid)
    combined_ids = combined_ids[:NUM_CANDIDATES]

    candidates = []
    for cid in combined_ids:
        entry = meta[cid]
        candidates.append({
            "chunk_id": cid,
            "score":    vector_scores.get(cid, 0.0),
            "text":     entry["text"],
            "source":   entry.get("source_path", "")
        })

    top50 = sorted(candidates, key=lambda x: -x["score"])[:NUM_CANDIDATES]
    print(f"\n=== TOP {NUM_CANDIDATES} RAW CHUNKS ===\n")
    for i, r in enumerate(top50, 1):
        snippet = r["text"].replace("\n", " ")
        print(f"--- #{i} (score={r['score']:.4f}) {r['source']} ---")
        print(snippet[:500] + ("…" if len(r["text"]) > 500 else ""))
        print()

    doc_candidates = [c for c in candidates if KEYWORD_DOC in c["source"].lower()]
    if doc_candidates:
        candidates = doc_candidates
    pairs = [(q, c["text"]) for c in candidates]
    cross_scores = cross.predict(pairs)
    for c, s in zip(candidates, cross_scores):
        c["cross_score"] = float(s)
    candidates = sorted(candidates, key=lambda x: -x["cross_score"])[:CROSS_TOPK]

    answer = rerank_with_ollama(q, candidates)
    print("\n=== LLM‐FILTERED ANSWER ===\n")
    print(answer)

if __name__ == "__main__":
    main()
