import json
import faiss
from sentence_transformers import SentenceTransformer
import pprint

# Paths
metadata_path = "vector_outputs/scaffold_metadata_1.json"
index_path = "vector_outputs/scaffold_index_1.faiss"
embedding_model_name = "all-MiniLM-L6-v2"

print("\n--- Inspecting Metadata ---")
with open(metadata_path, "r", encoding="utf-8") as f:
    metadata = json.load(f)
    print(f"Total metadata entries: {len(metadata)}")
    for i, entry in enumerate(metadata[:3]):
        print(f"{i+1}. {entry}")

print("\n--- Inspecting FAISS Index ---")
index = faiss.read_index(index_path)
print(f"FAISS index vectors: {index.ntotal}")

if len(metadata) == index.ntotal:
    print("\n✅ Metadata and FAISS index are aligned!")
else:
    print(f"\n❌ Mismatch: {len(metadata)} metadata entries vs {index.ntotal} index vectors")

print("\n--- Running Test Search ---")
model = SentenceTransformer(embedding_model_name)
query = "life cycle assessment"
query_embedding = model.encode([query])
distances, indices = index.search(query_embedding, k=3)
print(f"Top 3 search results for query: '{query}'")
for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), 1):
    if idx < len(metadata):
        entry = metadata[idx]
        print(f"{rank}. idx={idx}, score={dist:.4f}")
        print(f"   Text: {entry.get('text', '')[:500]}...\n")
    else:
        print(f"{rank}. idx={idx}, score={dist:.4f}, [Index out of range in metadata]")

# Print the full metadata entry for the top result
print("\n--- Full metadata entry for top result (idx=4640) ---")
if 4640 < len(metadata):
    pprint.pprint(metadata[4640])
else:
    print("Index 4640 is out of range.") 