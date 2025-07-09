"""
Benchmarking script for Scaffold AI model selection.
Benchmarks embedding, cross-encoder, and LLM models for latency, memory, and basic output quality.
Outputs a summary table for comparison.
"""
import time
import torch
import tracemalloc
from scaffold_core.config import MODEL_REGISTRY
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import pipeline

TEST_SENTENCES = [
    "Sustainability in engineering education is crucial for the future.",
    "What are the key competencies for climate resilience?",
    "Describe a project-based learning approach for environmental literacy."
]

RESULTS = []

def benchmark_embedding(model_name):
    start = time.time()
    tracemalloc.start()
    model = SentenceTransformer(model_name)
    embeddings = model.encode(TEST_SENTENCES)
    mem = tracemalloc.get_traced_memory()[1] / 1e6
    tracemalloc.stop()
    elapsed = time.time() - start
    return elapsed, mem

def benchmark_cross_encoder(model_name):
    start = time.time()
    tracemalloc.start()
    model = CrossEncoder(model_name)
    scores = model.predict([(s, s) for s in TEST_SENTENCES])
    mem = tracemalloc.get_traced_memory()[1] / 1e6
    tracemalloc.stop()
    elapsed = time.time() - start
    return elapsed, mem

def benchmark_llm(model_name, token):
    start = time.time()
    tracemalloc.start()
    pipe = pipeline(
        "text-generation",
        model=model_name,
        tokenizer=model_name,
        token=token,
        device=0 if torch.cuda.is_available() else -1,
        max_length=64,
        do_sample=False
    )
    outputs = pipe(TEST_SENTENCES[0])
    mem = tracemalloc.get_traced_memory()[1] / 1e6
    tracemalloc.stop()
    elapsed = time.time() - start
    return elapsed, mem, outputs[0]['generated_text'][:80]

def main():
    from scaffold_core.config import HF_TOKEN, MIXTRAL_API_KEY
    print("\n--- EMBEDDING MODELS ---")
    for key, info in MODEL_REGISTRY['embedding'].items():
        t, m = benchmark_embedding(info['name'])
        print(f"{info['name']:<45} | {t:6.2f}s | {m:8.1f} MB | {info['desc']}")
    print("\n--- CROSS-ENCODER MODELS ---")
    for key, info in MODEL_REGISTRY['cross_encoder'].items():
        t, m = benchmark_cross_encoder(info['name'])
        print(f"{info['name']:<45} | {t:6.2f}s | {m:8.1f} MB | {info['desc']}")
    print("\n--- LLM MODELS ---")
    for key, info in MODEL_REGISTRY['llm'].items():
        token = HF_TOKEN if key != 'mixtral' else MIXTRAL_API_KEY
        try:
            t, m, out = benchmark_llm(info['name'], token)
            print(f"{info['name']:<45} | {t:6.2f}s | {m:8.1f} MB | {info['desc']} | Output: {out}")
        except Exception as e:
            print(f"{info['name']:<45} | ERROR: {e}")

if __name__ == "__main__":
    main()
