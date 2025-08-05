#!/usr/bin/env python3
"""
Model Benchmark Tool for Scaffold AI
This script benchmarks different LLM models to compare their performance.
"""

import os
import sys
import time
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Test queries for benchmarking
BENCHMARK_QUERIES = [
    "What is sustainability in engineering?",
    "Explain the concept of life cycle assessment.",
    "How does climate change impact engineering design?"
]

def benchmark_model(model_key):
    """Benchmark a specific model."""
    try:
        # Import after adding project root to path
        from scaffold_core.config import LLM_MODELS
        import scaffold_core.config as config
        from scaffold_core.llm import get_llm
        
        # Check if model exists
        if model_key not in LLM_MODELS:
            logger.error(f"Model key '{model_key}' not found in LLM_MODELS")
            return None
        
        # Get model info
        model_name = LLM_MODELS[model_key]["name"]
        model_desc = LLM_MODELS[model_key]["desc"]
        use_onnx = LLM_MODELS[model_key].get("use_onnx", False)
        
        # Store original model
        original_model = config.SELECTED_LLM_MODEL
        
        # Switch to benchmark model
        config.SELECTED_LLM_MODEL = model_name
        config.LLM_MODEL = model_name
        
        # Check if model has ONNX flag and update config
        if use_onnx:
            config.USE_ONNX = True
            logger.info(f"ONNX optimization enabled for model: {model_name}")
        else:
            config.USE_ONNX = False
        
        # Results dictionary
        results = {
            "model_key": model_key,
            "model_name": model_name,
            "model_desc": model_desc,
            "use_onnx": use_onnx,
            "timestamp": datetime.now().isoformat(),
            "queries": [],
            "load_time": 0,
            "total_time": 0,
            "average_time": 0
        }
        
        # Load model and measure time
        load_start = time.time()
        logger.info(f"Loading model: {model_name}")
        llm = get_llm()
        load_time = time.time() - load_start
        results["load_time"] = load_time
        logger.info(f"Model loaded in {load_time:.2f} seconds")
        
        # Run benchmark queries
        total_time = 0
        for i, query in enumerate(BENCHMARK_QUERIES, 1):
            logger.info(f"Running query {i}/{len(BENCHMARK_QUERIES)}: {query[:50]}...")
            
            # Measure query time
            start_time = time.time()
            response = llm.generate_response(query)
            query_time = time.time() - start_time
            total_time += query_time
            
            # Calculate metrics
            word_count = len(response.split())
            
            # Store query results
            query_result = {
                "query": query,
                "time": query_time,
                "words": word_count,
                "response": response[:500] + "..." if len(response) > 500 else response
            }
            results["queries"].append(query_result)
            
            logger.info(f"Query {i} completed in {query_time:.2f}s, {word_count} words")
        
        # Calculate averages
        results["total_time"] = total_time
        results["average_time"] = total_time / len(BENCHMARK_QUERIES) if BENCHMARK_QUERIES else 0
        
        # Restore original model
        config.SELECTED_LLM_MODEL = original_model
        config.LLM_MODEL = original_model
        
        return results
    
    except Exception as e:
        logger.error(f"Error benchmarking model {model_key}: {e}")
        return None

def main():
    """Main function to parse arguments and run benchmarks."""
    # Import after adding project root to path
    from scaffold_core.config import LLM_MODELS
    
    # Get available models
    model_keys = list(LLM_MODELS.keys())
    
    # Create argument parser
    parser = argparse.ArgumentParser(description="Benchmark LLM models")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=model_keys,
        help="Models to benchmark (space-separated list)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Benchmark all available models"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available models"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results.json",
        help="Output file for benchmark results"
    )
    
    # Parse arguments
    if "--list" in sys.argv:
        print("\nAvailable Models for Benchmarking:")
        print("-" * 50)
        for key, info in LLM_MODELS.items():
            onnx_flag = " [ONNX]" if info.get("use_onnx", False) else ""
            print(f"{key}{onnx_flag}: {info['desc']}")
        print("\nUsage: python benchmark_models.py --models MODEL1 MODEL2 ...")
        return
    
    args = parser.parse_args()
    
    if not args.models and not args.all:
        parser.print_help()
        return
    
    # Determine models to benchmark
    models_to_benchmark = model_keys if args.all else args.models
    
    print(f"\nBenchmarking {len(models_to_benchmark)} models:")
    for model in models_to_benchmark:
        print(f"- {model}")
    print()
    
    # Run benchmarks
    results = []
    for model_key in models_to_benchmark:
        print(f"\n{'=' * 50}")
        print(f"Benchmarking {model_key}")
        print(f"{'=' * 50}")
        
        model_results = benchmark_model(model_key)
        if model_results:
            results.append(model_results)
    
    # Save results
    if results:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nBenchmark results saved to {args.output}")
        
        # Print summary
        print("\nBenchmark Summary:")
        print("-" * 50)
        print(f"{'Model':<15} {'Load Time':<12} {'Avg Query Time':<15} {'Total Time':<12}")
        print("-" * 50)
        for result in results:
            model_key = result["model_key"]
            load_time = f"{result['load_time']:.2f}s"
            avg_time = f"{result['average_time']:.2f}s"
            total_time = f"{result['total_time']:.2f}s"
            print(f"{model_key:<15} {load_time:<12} {avg_time:<15} {total_time:<12}")
    else:
        print("\nNo benchmark results generated.")

if __name__ == "__main__":
    main()