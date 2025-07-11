"""
Benchmarking script for Scaffold AI model selection.
Benchmarks embedding, cross-encoder, and LLM models for latency, memory,
and basic output quality. Outputs a summary table for comparison.
"""
import argparse
import os
import sys
import time
import tracemalloc
from typing import List, Tuple
import logging

import torch
from sentence_transformers import CrossEncoder, SentenceTransformer
from transformers.pipelines import pipeline

# Add project root to path to allow submodule imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scaffold_core.config import HF_TOKEN, MIXTRAL_API_KEY, MODEL_REGISTRY
from scaffold_core.log_config import setup_logging

logger = logging.getLogger(__name__)


TEST_SENTENCES: List[str] = [
    "Sustainability in engineering education is crucial for the future.",
    "What are the key competencies for climate resilience?",
    "Describe a project-based learning approach for environmental literacy."
]


def benchmark_embedding(model_name: str) -> Tuple[float, float]:
    """Benchmarks an embedding model."""
    start = time.time()
    tracemalloc.start()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer(model_name, device=device)
    model.encode(TEST_SENTENCES)
    mem = tracemalloc.get_traced_memory()[1] / 1e6
    tracemalloc.stop()
    elapsed = time.time() - start
    return elapsed, mem


def benchmark_cross_encoder(model_name: str) -> Tuple[float, float]:
    """Benchmarks a cross-encoder model."""
    start = time.time()
    tracemalloc.start()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CrossEncoder(model_name, device=device)
    model.predict([(s, s) for s in TEST_SENTENCES])
    mem = tracemalloc.get_traced_memory()[1] / 1e6
    tracemalloc.stop()
    elapsed = time.time() - start
    return elapsed, mem


def benchmark_llm(model_name: str, token: str) -> Tuple[float, float, str]:
    """Benchmarks a Language Model using ONNX Runtime."""
    from optimum.onnxruntime import ORTModelForCausalLM
    from transformers import AutoTokenizer

    start = time.time()
    tracemalloc.start()

    provider = "CUDAExecutionProvider" if torch.cuda.is_available() else "CPUExecutionProvider"
    logger.info(f"Using ONNX Runtime with provider: {provider}")

    # Optimum will automatically handle caching the exported ONNX model
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    model = ORTModelForCausalLM.from_pretrained(
        model_name,
        token=token,
        export=True,  # This will convert and save the model if not already cached
        provider=provider,
    )

    inputs = tokenizer(TEST_SENTENCES[0], return_tensors="pt")
    outputs = model.generate(**inputs, max_length=64, do_sample=False)
    generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    mem = tracemalloc.get_traced_memory()[1] / 1e6
    tracemalloc.stop()
    elapsed = time.time() - start

    # Clean the prompt from the beginning of the generated text
    if generated_text.startswith(TEST_SENTENCES[0]):
        generated_text = generated_text[len(TEST_SENTENCES[0]):].strip()

    return elapsed, mem, generated_text[:80]


def main():
    """Main function to run all benchmarks."""
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Benchmark different model types."
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=['embedding', 'cross_encoder', 'llm', 'all'],
        default='all',
        help="Type of model to benchmark."
    )
    args = parser.parse_args()


    if args.type in ['embedding', 'all']:
        logger.info("--- Starting EMBEDDING MODEL Benchmarks ---")
        for key, info in MODEL_REGISTRY['embedding'].items():
            t, m = benchmark_embedding(info['name'])
            logger.info(
                f"{info['name']:<45} | {t:6.2f}s | {m:8.1f} MB | "
                f"{info['desc']}"
            )

    if args.type in ['cross_encoder', 'all']:
        logger.info("--- Starting CROSS-ENCODER MODEL Benchmarks ---")
        for key, info in MODEL_REGISTRY['cross_encoder'].items():
            t, m = benchmark_cross_encoder(info['name'])
            logger.info(
                f"{info['name']:<45} | {t:6.2f}s | {m:8.1f} MB | "
                f"{info['desc']}"
            )

    if args.type in ['llm', 'all']:
        logger.info("--- Starting LLM MODEL Benchmarks ---")
        for key, info in MODEL_REGISTRY['llm'].items():
            token = HF_TOKEN if key != 'mixtral' else MIXTRAL_API_KEY
            try:
                t, m, out = benchmark_llm(info['name'], token)
                logger.info(
                    f"{info['name']:<45} | {t:6.2f}s | {m:8.1f} MB | "
                    f"{info['desc']}"
                )
                logger.info(f"{'':<47} | Output: {out}")
            except Exception as e:
                logger.error(
                    f"{info['name']:<45} | FAILED: {e}", exc_info=True
                )


if __name__ == "__main__":
    main()
