#!/usr/bin/env python3
"""
Full Test Suite for Scaffold AI
Runs 3 test queries on each available model and logs results to 'Full Test Results.txt'.
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

RESULTS_FILE = "Full Test Results.txt"

TEST_QUERIES = [
    "What is life cycle assessment?",
    "How can sustainability be integrated into fluid mechanics?",
    "What are key competencies for climate education?"
]

def run_test_suite():
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        f.write(f"Scaffold AI Full Test Suite Results\nGenerated: {datetime.now().isoformat()}\n\n")
        for llm_key, llm_info in LLM_MODELS.items():
            config_manager.set_selected_model('llm', llm_key)
            for emb_key, emb_info in EMBEDDING_MODELS.items():
                config_manager.set_selected_model('embedding', emb_key)
                f.write(f"=== LLM Model: {llm_key} ({llm_info['name']}) ===\n")
                f.write(f"Embedding Model: {emb_key} ({emb_info['name']})\n\n")
                for query in TEST_QUERIES:
                    start = time.time()
                    try:
                        result = query_enhanced_improved(query)
                        answer = result.get('response', '[No response]')
                        sources = result.get('sources', [])
                        sources_count = len(sources)
                        search_stats = result.get('search_stats', {})
                        latency = time.time() - start
                        f.write(f"Query: {query}\n")
                        f.write(f"Time Taken: {latency:.2f} seconds\n")
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
                        f.write(f"Answer:\n{answer}\n\n")
                        if sources_count == 0:
                            f.write(f"[ISSUE] No sources returned for this query.\n\n")
                    except Exception as e:
                        f.write(f"Query: {query}\n[ERROR] {str(e)}\n\n")
                f.write("-"*60 + "\n\n")

if __name__ == "__main__":
    run_test_suite()
