#!/usr/bin/env python3
"""
Script to generate query results and save them to a text file or print as JSON.
This script runs a query through the enhanced query system and saves
or prints the results.
"""

import json
import os
import sys
import datetime
from pathlib import Path
import argparse

print("DEBUG: Script starting...")

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
print(f"DEBUG: Added project root to path: {project_root}")

def run_query_and_save(query_text, output_filename=None, output_json=False):
    """
    Run a query and save results to a text file or print as JSON.
    
    Args:
        query_text (str): The query to run
        output_filename (str, optional): Output filename. If None, auto-generates.
        output_json (bool): If True, print JSON to stdout instead of saving text file.
    """
    
    print(f"Running query: {query_text}")
    
    try:
        print("DEBUG: About to import query_enhanced...")
        from scaffold_core.vector.enhanced_query import query_enhanced
        print("DEBUG: Successfully imported query_enhanced")
        
        print("Processing query...")
        result = query_enhanced(query_text)
        print("DEBUG: Query completed successfully")
        
        if output_json:
            print("DEBUG: About to print JSON...")
            print(json.dumps(result, indent=2, ensure_ascii=False))
            print("DEBUG: JSON printed successfully")
            return None
        
        # Generate output filename if not provided
        if output_filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_query = "".join(c for c in query_text[:30] if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_query = safe_query.replace(' ', '_')
            output_filename = f"query_results_{safe_query}_{timestamp}.txt"
        
        output_dir = Path("query_outputs")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / output_filename
        formatted_output = format_query_results(result, query_text)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(formatted_output)
        print(f"Results saved to: {output_path}")
        return str(output_path)
    except Exception as e:
        print(f"Error running query: {e}")
        import traceback
        traceback.print_exc()
        return None

def format_query_results(result, original_query):
    """
    Format query results into a readable text format.
    
    Args:
        result (dict): The query result from enhanced_query
        original_query (str): The original query text
    
    Returns:
        str: Formatted text output
    """
    
    output = []
    
    # Header
    output.append("=" * 80)
    output.append("SCAFFOLD AI QUERY RESULTS")
    output.append("=" * 80)
    output.append(f"Query: {original_query}")
    output.append(f"Timestamp: {result.get('timestamp', datetime.datetime.now().isoformat())}")
    output.append("")
    
    # Search Statistics
    output.append("SEARCH STATISTICS")
    output.append("-" * 40)
    stats = result.get('search_stats', {})
    output.append(f"Initial candidates found: {stats.get('initial_candidates', 0)}")
    output.append(f"After reranking: {stats.get('reranked_candidates', 0)}")
    output.append(f"After filtering: {stats.get('filtered_candidates', 0)}")
    output.append(f"Final candidates used: {stats.get('final_candidates', 0)}")
    output.append("")
    
    # Main Response
    output.append("AI RESPONSE")
    output.append("-" * 40)
    response = result.get('response', 'No response generated')
    output.append(response)
    output.append("")
    
    # Sources
    output.append("SOURCES USED")
    output.append("-" * 40)
    sources = result.get('sources', [])
    
    if not sources:
        output.append("No sources found.")
    else:
        for i, source in enumerate(sources, 1):
            output.append(f"Source {i}:")
            output.append(f"  Score: {source.get('score', 'N/A'):.4f}")
            output.append(f"  Name: {source.get('source', {}).get('name', 'Unknown')}")
            output.append(f"  ID: {source.get('source', {}).get('id', 'Unknown')}")
            output.append(f"  Path: {source.get('source', {}).get('raw_path', 'Unknown')}")
            output.append(f"  Preview: {source.get('text_preview', 'No preview available')}")
            output.append("")
    
    # Detailed Candidates (if available)
    candidates = result.get('candidates', [])
    if candidates:
        output.append("DETAILED CANDIDATE ANALYSIS")
        output.append("-" * 40)
        for i, candidate in enumerate(candidates, 1):
            output.append(f"Candidate {i}:")
            output.append(f"  Chunk ID: {candidate.get('chunk_id', 'N/A')}")
            output.append(f"  Score: {candidate.get('score', 'N/A'):.4f}")
            if 'cross_score' in candidate:
                output.append(f"  Cross-encoder score: {candidate.get('cross_score', 'N/A'):.4f}")
            if 'contextual_score' in candidate:
                output.append(f"  Contextual score: {candidate.get('contextual_score', 'N/A')}")
            output.append(f"  Search type: {candidate.get('search_type', 'Unknown')}")
            output.append(f"  Text: {candidate.get('text', 'No text available')[:500]}...")
            output.append("")
    
    # Footer
    output.append("=" * 80)
    output.append("End of Query Results")
    output.append("=" * 80)
    
    return "\n".join(output)

def main():
    """Main function to run queries and save results."""
    print("DEBUG: main() function starting...")
    
    try:
        parser = argparse.ArgumentParser(description="Run a query and output results as text or JSON.")
        parser.add_argument('--json', action='store_true', help='Print raw JSON result to stdout')
        parser.add_argument('--query', type=str, help='Query to run (if not provided, will prompt)')
        args = parser.parse_args()
        print(f"DEBUG: Parsed arguments - json: {args.json}, query: {args.query}")

        if args.query:
            print("DEBUG: Running with provided query...")
            run_query_and_save(args.query, output_json=args.json)
            return

        # Fallback to interactive mode
        example_queries = [
            "What is sustainability in engineering education?",
            "How are climate change concepts integrated into engineering curricula?",
            "What are the key competencies for environmental literacy in engineering?",
            "How do engineering programs address sustainability challenges?",
            "What methods are used to teach sustainability in civil engineering?"
        ]
        print("Scaffold AI Query Results Generator")
        print("=" * 50)
        print("\nAvailable example queries:")
        for i, query in enumerate(example_queries, 1):
            print(f"{i}. {query}")
        print("\nOptions:")
        print("0. Run all example queries")
        print("C. Enter custom query")
        choice = input("\nEnter your choice (0-5 or C): ").strip().upper()
        if choice == "0":
            print(f"\nRunning {len(example_queries)} queries...")
            for i, query in enumerate(example_queries, 1):
                print(f"\n--- Query {i}/{len(example_queries)} ---")
                output_file = run_query_and_save(query)
                if output_file:
                    print(f"Query {i} completed: {output_file}")
                else:
                    print(f"Query {i} failed")
        elif choice == "C":
            custom_query = input("Enter your query: ").strip()
            if custom_query:
                output_file = run_query_and_save(custom_query)
                if output_file:
                    print(f"Query completed: {output_file}")
                else:
                    print("Query failed")
            else:
                print("No query entered")
        elif choice.isdigit() and 1 <= int(choice) <= len(example_queries):
            query_index = int(choice) - 1
            query = example_queries[query_index]
            output_file = run_query_and_save(query)
            if output_file:
                print(f"Query completed: {output_file}")
            else:
                print("Query failed")
        else:
            print("Invalid choice")
            
    except Exception as e:
        print(f"Fatal error in main(): {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    try:
        print("DEBUG: Script entry point reached")
        main()
        print("DEBUG: Script completed successfully")
    except Exception as e:
        print(f"Fatal error at script level: {e}")
        import traceback
        traceback.print_exc() 