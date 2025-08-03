#!/usr/bin/env python3
"""
Comprehensive Mixtral Test Suite
Tests Mixtral with varied base prompts, temperatures, and environmental engineering curriculum queries.
"""

import os
import sys
import time
import json
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import model registries
from scaffold_core.config import EMBEDDING_MODELS, LLM_MODELS
from scaffold_core.config_manager import config_manager
from scaffold_core.vector.enhanced_query_improved import query_enhanced_improved

RESULTS_FILE = "Comprehensive Mixtral Test Results.txt"

# Environmental Engineering Curriculum Queries
CURRICULUM_QUERIES = [
    "What is life cycle assessment?",
    "How can sustainability be integrated into fluid mechanics?",
    "What are key competencies for climate education?",
    "How to teach environmental impact assessment?",
    "What are best practices for sustainable engineering design?",
    "How to integrate renewable energy into engineering curriculum?",
    "What are the core principles of green engineering?",
    "How to assess environmental risks in engineering projects?",
    "What are sustainable materials for construction?",
    "How to teach waste management in engineering courses?"
]

# Different base prompts to test
BASE_PROMPTS = [
    # Original prompt
    "You are a helpful AI assistant. Answer the user's question based on the provided context. If you cannot find the answer in the context, say so. Use the context to provide accurate and relevant information.",
    
    # More specific prompt
    "You are an environmental engineering curriculum assistant. Answer questions about sustainability, environmental engineering, and curriculum development based on the provided academic sources. Only use information from the provided context. If information is not available in the context, clearly state this.",
    
    # Factual prompt
    "You are a factual AI assistant for environmental engineering education. Provide accurate, evidence-based answers using only the information from the provided academic sources. Cite specific details from the sources when possible. If the answer cannot be found in the provided context, explicitly state this limitation.",
    
    # Educational prompt
    "You are an educational AI assistant specializing in environmental engineering curriculum. Help students and educators understand sustainability concepts, engineering principles, and curriculum development. Base your responses on the provided academic sources. If you cannot find relevant information in the context, acknowledge this limitation.",
    
    # Research-focused prompt
    "You are a research assistant for environmental engineering education. Provide comprehensive, well-sourced answers based on the provided academic literature. Use specific citations from the sources when possible. If the requested information is not available in the provided context, clearly indicate this."
]

# Temperature variations
TEMPERATURES = [0.1, 0.3, 0.5, 0.7, 0.9]

def modify_base_prompt(new_prompt):
    """Modify the base prompt in the enhanced query system."""
    try:
        # Import and modify the prompt in the enhanced query system
        import scaffold_core.vector.enhanced_query_improved
        
        # Store original prompt
        original_prompt = scaffold_core.vector.enhanced_query_improved.MAIN_PROMPT
        
        # Modify the prompt
        scaffold_core.vector.enhanced_query_improved.MAIN_PROMPT = new_prompt
        
        return original_prompt
    except Exception as e:
        print(f"Error modifying base prompt: {e}")
        return None

def restore_base_prompt(original_prompt):
    """Restore the original base prompt."""
    try:
        import scaffold_core.vector.enhanced_query_improved
        scaffold_core.vector.enhanced_query_improved.MAIN_PROMPT = original_prompt
    except Exception as e:
        print(f"Error restoring base prompt: {e}")

def run_comprehensive_test():
    """Run comprehensive tests with varied prompts, temperatures, and queries."""
    
    print("🚀 Comprehensive Mixtral Test Suite")
    print("=" * 60)
    
    # Set Mixtral as the LLM model
    config_manager.set_selected_model('llm', 'mixtral')
    
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        f.write(f"Comprehensive Mixtral Test Results\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        
        # Test each base prompt
        for prompt_idx, base_prompt in enumerate(BASE_PROMPTS, 1):
            f.write(f"=== TEST SET {prompt_idx}: Base Prompt Variation ===\n")
            f.write(f"Base Prompt: {base_prompt[:100]}...\n\n")
            
            # Store original prompt
            original_prompt = modify_base_prompt(base_prompt)
            
            # Test each temperature
            for temp_idx, temperature in enumerate(TEMPERATURES, 1):
                f.write(f"--- Temperature {temp_idx}: {temperature} ---\n")
                
                # Test each curriculum query
                for query_idx, query in enumerate(CURRICULUM_QUERIES, 1):
                    f.write(f"\nQuery {query_idx}: {query}\n")
                    
                    start = time.time()
                    try:
                        # Temporarily modify temperature
                        import scaffold_core.config
                        original_temp = scaffold_core.config.LLM_TEMPERATURE
                        scaffold_core.config.LLM_TEMPERATURE = temperature
                        
                        # Get response
                        result = query_enhanced_improved(query)
                        
                        # Restore temperature
                        scaffold_core.config.LLM_TEMPERATURE = original_temp
                        
                        answer = result.get('response', '[No response]')
                        sources = result.get('sources', [])
                        sources_count = len(sources)
                        search_stats = result.get('search_stats', {})
                        latency = time.time() - start
                        
                        f.write(f"Time Taken: {latency:.2f} seconds\n")
                        f.write(f"Temperature: {temperature}\n")
                        f.write(f"Sources Returned: {sources_count}\n")
                        f.write(f"Search Stats: {json.dumps(search_stats)}\n")
                        
                        # List names of final sources
                        if sources_count > 0:
                            source_names = []
                            for s in sources:
                                src = s.get('source', {})
                                name = src.get('name') or src.get('id') or str(src)[:60]
                                source_names.append(name)
                            f.write(f"Final Sources Used: {json.dumps(source_names, ensure_ascii=False)}\n")
                        
                        f.write(f"Answer:\n{answer}\n")
                        
                        if sources_count == 0:
                            f.write(f"[ISSUE] No sources returned for this query.\n")
                            
                    except Exception as e:
                        f.write(f"[ERROR] {str(e)}\n")
                    
                    f.write("-" * 40 + "\n")
                
                f.write("=" * 60 + "\n\n")
            
            # Restore original prompt
            if original_prompt:
                restore_base_prompt(original_prompt)
        
        f.write("=== SUMMARY ===\n")
        f.write(f"Total Test Sets: {len(BASE_PROMPTS)}\n")
        f.write(f"Temperatures Tested: {len(TEMPERATURES)}\n")
        f.write(f"Queries Tested: {len(CURRICULUM_QUERIES)}\n")
        f.write(f"Total Tests: {len(BASE_PROMPTS) * len(TEMPERATURES) * len(CURRICULUM_QUERIES)}\n")
    
    print(f"✅ Comprehensive results saved to {RESULTS_FILE}")

def run_focused_temperature_test():
    """Run focused tests with specific temperature ranges."""
    
    print("🌡️ Running Focused Temperature Tests")
    print("=" * 50)
    
    # Set Mixtral as the LLM model
    config_manager.set_selected_model('llm', 'mixtral')
    
    # Use the most effective base prompt
    best_prompt = BASE_PROMPTS[2]  # Factual prompt
    original_prompt = modify_base_prompt(best_prompt)
    
    # Focus on key queries
    key_queries = [
        "What is life cycle assessment?",
        "How can sustainability be integrated into fluid mechanics?",
        "What are key competencies for climate education?"
    ]
    
    # Test specific temperature ranges
    test_temperatures = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    results = []
    
    for temp in test_temperatures:
        print(f"Testing temperature: {temp}")
        
        for query in key_queries:
            start = time.time()
            try:
                # Temporarily modify temperature
                import scaffold_core.config
                original_temp = scaffold_core.config.LLM_TEMPERATURE
                scaffold_core.config.LLM_TEMPERATURE = temp
                
                # Get response
                result = query_enhanced_improved(query)
                
                # Restore temperature
                scaffold_core.config.LLM_TEMPERATURE = original_temp
                
                response_time = time.time() - start
                response = result.get('response', '')
                sources_count = len(result.get('sources', []))
                
                results.append({
                    'temperature': temp,
                    'query': query,
                    'response_time': response_time,
                    'response_length': len(response),
                    'sources_count': sources_count,
                    'response': response[:200] + "..." if len(response) > 200 else response
                })
                
                print(f"  Query: {query[:50]}...")
                print(f"  Time: {response_time:.2f}s, Sources: {sources_count}, Length: {len(response)} chars")
                
            except Exception as e:
                print(f"  Error: {e}")
    
    # Restore original prompt
    if original_prompt:
        restore_base_prompt(original_prompt)
    
    # Save focused results
    focused_file = "Focused Temperature Test Results.txt"
    with open(focused_file, "w", encoding="utf-8") as f:
        f.write(f"Focused Temperature Test Results\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        
        for result in results:
            f.write(f"Temperature: {result['temperature']}\n")
            f.write(f"Query: {result['query']}\n")
            f.write(f"Response Time: {result['response_time']:.2f}s\n")
            f.write(f"Response Length: {result['response_length']} chars\n")
            f.write(f"Sources Count: {result['sources_count']}\n")
            f.write(f"Response Preview: {result['response']}\n")
            f.write("-" * 40 + "\n")
    
    print(f"✅ Focused results saved to {focused_file}")

def main():
    """Main function to run comprehensive tests."""
    
    print("🚀 Starting Comprehensive Mixtral Test Suite")
    print("=" * 60)
    
    # Run comprehensive test
    run_comprehensive_test()
    
    # Run focused temperature test
    run_focused_temperature_test()
    
    print("🎉 All tests completed!")

if __name__ == "__main__":
    main() 