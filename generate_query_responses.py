#!/usr/bin/env python3
"""
Generate query responses using the enhanced search system
Tests multiple educational queries and saves results to output files
"""

import sys
import os
import json
import datetime
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from scaffold_core.vector.enhanced_query import query_enhanced

def main():
    print("🔍 Generating query responses with enhanced search system...")
    
    # Test queries for educational content about sustainability and LCA
    test_queries = [
        "What is life cycle assessment?",
        "How do you conduct a sustainability assessment in higher education?",
        "What are the main phases of life cycle assessment?",
        "How can universities implement environmental sustainability frameworks?",
        "What are the key indicators for measuring environmental impact?",
        "How do you assess the carbon footprint of educational institutions?",
        "What methods are used for environmental impact assessment?",
        "How can life cycle thinking be integrated into curriculum design?",
        "What are best practices for sustainable campus management?",
        "How do you measure and report sustainability performance in education?"
    ]
    
    # Create outputs directory
    outputs_dir = Path("query_outputs")
    outputs_dir.mkdir(exist_ok=True)
    
    # Generate timestamp for file naming
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results = []
    
    print(f"\n📝 Processing {len(test_queries)} queries...")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Query {i}/{len(test_queries)} ---")
        print(f"Query: {query}")
        
        try:
            # Generate response using enhanced query system
            result = query_enhanced(query)
            
            # Extract key information
            response_data = {
                "query_id": i,
                "query": query,
                "response": result["response"],
                "candidates_found": result["search_stats"]["final_candidates"],
                "search_stats": result["search_stats"],
                "timestamp": datetime.datetime.now().isoformat(),
                "top_sources": [
                    {
                        "source": candidate.get("source", "Unknown"),
                        "score": candidate.get("cross_score", candidate.get("score", 0))
                    }
                    for candidate in result["candidates"][:3]
                ]
            }
            
            results.append(response_data)
            
            print(f"✅ Response generated ({len(result['response'])} chars)")
            print(f"📊 Found {result['search_stats']['final_candidates']} candidates")
            print(f"📄 Response preview: {result['response'][:150]}...")
            
        except Exception as e:
            print(f"❌ Error processing query: {str(e)}")
            error_data = {
                "query_id": i,
                "query": query,
                "error": str(e),
                "timestamp": datetime.datetime.now().isoformat()
            }
            results.append(error_data)
    
    # Save detailed results to JSON
    json_output_path = outputs_dir / f"query_responses_{timestamp}.json"
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # Save human-readable report
    report_output_path = outputs_dir / f"query_report_{timestamp}.txt"
    with open(report_output_path, "w", encoding="utf-8") as f:
        f.write(f"Enhanced Query System Test Report\n")
        f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Queries: {len(test_queries)}\n")
        f.write("=" * 80 + "\n\n")
        
        for result in results:
            f.write(f"Query {result['query_id']}: {result['query']}\n")
            f.write("-" * 60 + "\n")
            
            if "error" in result:
                f.write(f"ERROR: {result['error']}\n")
            else:
                f.write(f"Candidates Found: {result['candidates_found']}\n")
                f.write(f"Response Length: {len(result['response'])} characters\n")
                f.write(f"Top Sources: {', '.join([src['source'] for src in result['top_sources']])}\n")
                f.write(f"\nResponse:\n{result['response']}\n")
            
            f.write("\n" + "=" * 80 + "\n\n")
    
    # Print summary
    successful_queries = len([r for r in results if "error" not in r])
    print(f"\n🎉 Summary:")
    print(f"   Total queries: {len(test_queries)}")
    print(f"   Successful: {successful_queries}")
    print(f"   Failed: {len(test_queries) - successful_queries}")
    print(f"   JSON output: {json_output_path}")
    print(f"   Report output: {report_output_path}")

if __name__ == "__main__":
    main() 