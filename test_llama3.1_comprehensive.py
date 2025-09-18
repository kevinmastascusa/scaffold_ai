#!/usr/bin/env python3
"""
Comprehensive Llama 3.1 Test Suite
Tests Llama 3.1 8B and 70B models with proper prompt formatting
and ensures no response truncation.
"""

import os
import sys
import time
import json
import logging
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path

# Load environment variables from .env file first
def load_env_file():
    """Load environment variables from .env file."""
    env_file = Path(".env")
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value
        print("✓ Environment variables loaded from .env file")

# Load environment before importing any modules
load_env_file()

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scaffold_core.config import LLM_MODELS, LLM_TEMPERATURE, LLM_MAX_NEW_TOKENS
from scaffold_core.vector.enhanced_query_improved import improved_enhanced_query_system
from scaffold_core.log_config import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Test queries for sustainability curriculum
TEST_QUERIES = [
    "What is life cycle assessment and how can it be integrated into engineering education?",
    "How can sustainability principles be effectively integrated into fluid mechanics courses?",
    "What are the key competencies needed for climate resilience education in engineering?",
    "Describe a project-based learning approach for teaching environmental literacy in civil engineering.",
    "How can systems thinking be incorporated into sustainability education for mechanical engineers?"
]

# Llama 3.1 specific prompt formatting
def format_llama_prompt(query: str, context: str) -> str:
    """
    Format prompt specifically for Llama 3.1 models.
    Llama 3.1 uses a specific chat format with system, user, and assistant messages.
    """
    system_prompt = """You are an expert in sustainability education and engineering curriculum development. 
Your role is to provide comprehensive, well-structured responses based on the provided sources.
Always cite your sources clearly and provide detailed explanations.
Focus on practical applications and educational strategies."""

    user_prompt = f"""Based on the following sources, please answer this question: {query}

Sources:
{context}

Please provide a comprehensive answer that:
1. Directly addresses the question
2. Cites specific sources
3. Provides practical examples
4. Suggests educational strategies
5. Considers implementation challenges"""

    # Llama 3.1 chat format
    formatted_prompt = f"""<|system|>
{system_prompt}
<|user|>
{user_prompt}
<|assistant|>"""

    return formatted_prompt

def test_llama_model(model_key: str, model_name: str) -> Dict[str, Any]:
    """Test a specific Llama model with comprehensive evaluation."""
    logger.info(f"Testing {model_key}: {model_name}")
    
    # Get model parameters from config
    from scaffold_core.config import (
        LLM_MAX_NEW_TOKENS,
        LLM_TEMPERATURE,
        LLM_TOP_P,
        TOP_K_INITIAL,
        TOP_K_FINAL,
        USE_ONNX
    )
    
    results = {
        "model": model_key,
        "model_name": model_name,
        "timestamp": datetime.now().isoformat(),
        "parameters": {
            "max_new_tokens": LLM_MAX_NEW_TOKENS,
            "temperature": LLM_TEMPERATURE,
            "top_p": LLM_TOP_P,
            "top_k_initial": TOP_K_INITIAL,
            "top_k_final": TOP_K_FINAL,
            "use_onnx": USE_ONNX
        },
        "queries": []
    }
    
    # Initialize query system
    try:
        improved_enhanced_query_system.initialize()
        logger.info("Query system initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize query system: {e}")
        return results
    
    for i, query in enumerate(TEST_QUERIES, 1):
        logger.info(f"Processing query {i}/{len(TEST_QUERIES)}: {query[:100]}...")
        
        start_time = time.time()
        
        try:
            # Process query
            response_data = improved_enhanced_query_system.query(query)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Extract response details
            response_text = response_data.get("response", "")
            sources = response_data.get("sources", [])
            error = response_data.get("error")
            
            # Calculate metrics
            word_count = len(response_text.split())
            source_count = len(sources)
            
            # Check for truncation indicators
            truncation_indicators = ["...", "etc.", "and so on", "continues", "more"]
            is_truncated = any(response_text.endswith(indicator) for indicator in truncation_indicators) if response_text else False
            
            query_result = {
                "query": query,
                "response": response_text,
                "processing_time": processing_time,
                "word_count": word_count,
                "source_count": source_count,
                "is_truncated": is_truncated,
                "error": error,
                "sources": sources
            }
            
            results["queries"].append(query_result)
            
            logger.info(f"Query {i} completed in {processing_time:.2f}s, {word_count} words, {source_count} sources")
            
        except Exception as e:
            logger.error(f"Error processing query {i}: {e}")
            query_result = {
                "query": query,
                "error": str(e),
                "processing_time": 0,
                "word_count": 0,
                "source_count": 0,
                "is_truncated": False
            }
            results["queries"].append(query_result)
    
    return results

def run_comprehensive_test():
    """Run comprehensive test on all Llama 3.1 models."""
    logger.info("Starting comprehensive Llama 3.1 test suite")
    
    # Test TinyLlama-ONNX model for speed
    models_to_test = [
        ("tinyllama-onnx", LLM_MODELS["tinyllama-onnx"]["name"])
    ]
    
    all_results = []
    
    for model_key, model_name in models_to_test:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing {model_key.upper()}")
        logger.info(f"{'='*60}")
        
        # Update config to use this model
        from scaffold_core.config import SELECTED_LLM_MODEL
        import scaffold_core.config as config
        
        # Temporarily switch to this model
        original_model = config.SELECTED_LLM_MODEL
        config.SELECTED_LLM_MODEL = model_name
        config.LLM_MODEL = model_name
        
        try:
            results = test_llama_model(model_key, model_name)
            all_results.append(results)
        except Exception as e:
            logger.error(f"Failed to test {model_key}: {e}")
        finally:
            # Restore original model
            config.SELECTED_LLM_MODEL = original_model
            config.LLM_MODEL = original_model
    
    # Generate comprehensive report
    generate_test_report(all_results)

def generate_test_report(results: List[Dict[str, Any]]):
    """Generate a comprehensive test report."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"tinyllama_onnx_test_results_{timestamp}.json"
    summary_file = f"tinyllama_onnx_test_summary_{timestamp}.txt"
    
    # Save detailed results
    with open(report_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate summary
    summary_lines = [
        "ONNX-Optimized TinyLlama Test Results",
        "=" * 50,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "MODEL CONFIGURATION",
        "-" * 30
    ]
    
    # Add model configuration
    for result in results:
        model_name = result["model"]
        params = result.get("parameters", {})
        
        summary_lines.extend([
            f"\n{model_name.upper()} CONFIGURATION:",
            f"  Model: {result['model_name']}",
            f"  Max New Tokens: {params.get('max_new_tokens', 'N/A')}",
            f"  Temperature: {params.get('temperature', 'N/A')}",
            f"  Top-P: {params.get('top_p', 'N/A')}",
            f"  Initial Search Results: {params.get('top_k_initial', 'N/A')}",
            f"  Final Results: {params.get('top_k_final', 'N/A')}",
            f"  ONNX Optimization: {'Enabled' if params.get('use_onnx', False) else 'Disabled'}",
            ""
        ])
    
    summary_lines.extend([
        "\nMODEL PERFORMANCE SUMMARY",
        "-" * 30
    ])
    
    for result in results:
        model_name = result["model"]
        queries = result["queries"]
        
        # Calculate statistics
        total_time = sum(q.get("processing_time", 0) for q in queries)
        avg_time = total_time / len(queries) if queries else 0
        total_words = sum(q.get("word_count", 0) for q in queries)
        avg_words = total_words / len(queries) if queries else 0
        total_sources = sum(q.get("source_count", 0) for q in queries)
        avg_sources = total_sources / len(queries) if queries else 0
        truncated_count = sum(1 for q in queries if q.get("is_truncated", False))
        error_count = sum(1 for q in queries if q.get("error"))
        
        summary_lines.extend([
            f"\n{model_name.upper()}:",
            f"  Average Response Time: {avg_time:.2f}s",
            f"  Average Response Length: {avg_words:.0f} words",
            f"  Average Sources Used: {avg_sources:.1f}",
            f"  Truncated Responses: {truncated_count}/{len(queries)}",
            f"  Errors: {error_count}/{len(queries)}",
            ""
        ])
    
    # Add detailed query results
    summary_lines.extend([
        "\nDETAILED QUERY RESULTS",
        "-" * 30
    ])
    
    for result in results:
        model_name = result["model"]
        summary_lines.append(f"\n{model_name.upper()} Results:")
        
        for i, query_result in enumerate(result["queries"], 1):
            query = query_result["query"]
            time_taken = query_result.get("processing_time", 0)
            word_count = query_result.get("word_count", 0)
            source_count = query_result.get("source_count", 0)
            is_truncated = query_result.get("is_truncated", False)
            error = query_result.get("error")
            response = query_result.get("response", "")
            sources = query_result.get("sources", [])
            
            status = "ERROR" if error else ("TRUNCATED" if is_truncated else "SUCCESS")
            
            summary_lines.extend([
                f"\n  QUERY {i}: {query}",
                f"  Status: {status} | Time: {time_taken:.2f}s | Words: {word_count} | Sources: {source_count}",
                f"\n  RESPONSE:",
                f"{response}",
                f"\n  TOP SOURCES:"
            ])
            
            # Add top sources (limited to 10)
            for j, source in enumerate(sources[:10], 1):
                source_text = source.get("text", "")
                source_title = source.get("title", "Unknown")
                
                # Try to get source information from different possible locations
                if "source" in source:
                    source_info = source["source"]
                    source_file = source_info.get("name", source_info.get("id", "Unknown file"))
                    source_page = source_info.get("page", "")
                else:
                    source_file = source.get("file", "Unknown file")
                    source_page = source.get("page", "")
                
                # Create a better source title
                if source_title == "Unknown" and source_file != "Unknown file":
                    source_title = f"File: {source_file}"
                    if source_page:
                        source_title += f" (Page {source_page})"
                
                # Get text preview from different possible locations
                if "text_preview" in source:
                    source_preview = source["text_preview"]
                else:
                    source_preview = source_text[:100] + "..." if len(source_text) > 100 else source_text
                
                summary_lines.append(f"  {j}. {source_title}: {source_preview}")
            
            summary_lines.append("")  # Add blank line between queries
            
            if error:
                summary_lines.append(f"    Error: {error}")
    
    # Save summary
    with open(summary_file, 'w') as f:
        f.write('\n'.join(summary_lines))
    
    logger.info(f"Detailed results saved to: {report_file}")
    logger.info(f"Summary saved to: {summary_file}")
    
    # Print summary to console
    print('\n'.join(summary_lines))

if __name__ == "__main__":
    run_comprehensive_test() 