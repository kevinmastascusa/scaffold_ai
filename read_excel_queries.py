#!/usr/bin/env python3
"""
Script to read queries from the Excel file and test them with the vector system.
"""

import pandas as pd
import sys
import os

# Add the project root to the Python path
sys.path.append('/Users/kevinmastascusa/GITHUB/scaffold_ai')

def read_excel_queries():
    """Read queries from the Excel file."""
    try:
        # Read the Excel file
        df = pd.read_excel('Prompt Examples.xlsx')
        
        print("Excel file contents:")
        print("=" * 50)
        print(f"Columns: {list(df.columns)}")
        print(f"Shape: {df.shape}")
        print("\nFirst few rows:")
        print(df.head())
        
        # Look for query-related columns
        query_columns = []
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['query', 'prompt', 'question', 'text']):
                query_columns.append(col)
        
        print(f"\nPotential query columns: {query_columns}")
        
        # Extract queries from the first column that looks like it contains text
        queries = []
        for col in df.columns:
            if df[col].dtype == 'object':  # String/text column
                non_null_values = df[col].dropna()
                if len(non_null_values) > 0:
                    print(f"\nColumn '{col}' contains {len(non_null_values)} non-null values")
                    print("Sample values:")
                    for i, value in enumerate(non_null_values.head(5)):
                        print(f"  {i+1}. {str(value)[:100]}...")
                    queries.extend(non_null_values.tolist())
                    break
        
        return queries
        
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return []

def test_queries_with_vector_system(queries):
    """Test the queries with the vector system."""
    if not queries:
        print("No queries found to test.")
        return
    
    print(f"\nTesting {len(queries)} queries with vector system...")
    print("=" * 50)
    
    # Import the enhanced query system
    try:
        from scaffold_core.vector.enhanced_query import EnhancedQuerySystem
        
        # Initialize the system
        eqs = EnhancedQuerySystem()
        eqs.initialize()
        
        for i, query in enumerate(queries[:5], 1):  # Test first 5 queries
            print(f"\nQuery {i}: {query}")
            print("-" * 30)
            
            try:
                # Perform semantic search
                candidates = eqs.semantic_search(query, k=3)
                print(f"Found {len(candidates)} candidates")
                
                if candidates:
                    for j, candidate in enumerate(candidates, 1):
                        print(f"  Candidate {j}:")
                        print(f"    Score: {candidate.get('score', 'N/A')}")
                        print(f"    Text: {candidate.get('text', 'N/A')[:200]}...")
                        print(f"    Source: {candidate.get('source', 'N/A')}")
                else:
                    print("  No candidates found")
                    
            except Exception as e:
                print(f"  Error processing query: {e}")
                
    except Exception as e:
        print(f"Error initializing vector system: {e}")

if __name__ == "__main__":
    print("Reading queries from Excel file...")
    queries = read_excel_queries()
    
    if queries:
        print(f"\nFound {len(queries)} queries")
        test_queries_with_vector_system(queries)
    else:
        print("No queries found in the Excel file.") 