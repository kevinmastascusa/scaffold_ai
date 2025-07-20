#!/usr/bin/env python3
"""
Script to extract full queries from the Excel file by combining columns.
"""

import pandas as pd
import sys
import os

def extract_full_queries():
    """Extract full queries from the Excel file by combining columns."""
    try:
        # Read the Excel file
        df = pd.read_excel('Prompt Examples.xlsx')
        
        print("Excel file contents:")
        print("=" * 50)
        print(f"Columns: {list(df.columns)}")
        print(f"Shape: {df.shape}")
        
        # Look for rows that contain query content
        queries = []
        
        # Process each row
        for index, row in df.iterrows():
            # Combine all non-null values in the row
            row_values = []
            for col in df.columns:
                value = row[col]
                if pd.notna(value) and str(value).strip():
                    row_values.append(str(value).strip())
            
            # If we have values, combine them into a query
            if row_values:
                full_query = ' '.join(row_values)
                if len(full_query) > 10:  # Only include substantial queries
                    queries.append(full_query)
                    print(f"Query {len(queries)}: {full_query}")
        
        return queries
        
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return []

def test_queries_with_simple_search(queries):
    """Test queries with a simple vector search without LLM."""
    if not queries:
        print("No queries found to test.")
        return
    
    print(f"\nTesting {len(queries)} queries with simple vector search...")
    print("=" * 50)
    
    # Import only the vector components
    try:
        import sys
        sys.path.append('/Users/kevinmastascusa/GITHUB/scaffold_ai')
        
        # Import the sentence transformer directly
        from sentence_transformers import SentenceTransformer
        import faiss
        import json
        import numpy as np
        
        # Load the embedding model
        print("Loading embedding model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load the FAISS index and metadata
        print("Loading FAISS index...")
        index_path = "vector_outputs/scaffold_index_1.faiss"
        metadata_path = "vector_outputs/scaffold_metadata_1.json"
        
        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            print("FAISS index or metadata not found!")
            return
        
        # Load the index
        index = faiss.read_index(index_path)
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        print(f"Index loaded with {index.ntotal} vectors")
        print(f"Metadata contains {len(metadata)} entries")
        
        # Test each query
        for i, query in enumerate(queries[:5], 1):  # Test first 5 queries
            print(f"\nQuery {i}: {query}")
            print("-" * 30)
            
            try:
                # Encode the query
                query_embedding = model.encode([query])
                
                # Search the index
                scores, indices = index.search(query_embedding, k=3)
                
                print(f"Found {len(indices[0])} candidates")
                
                for j, (score, idx) in enumerate(zip(scores[0], indices[0])):
                    if idx < len(metadata):
                        candidate = metadata[idx]
                        print(f"  Candidate {j+1}:")
                        print(f"    Score: {score:.4f}")
                        print(f"    Text: {candidate.get('text', 'N/A')[:200]}...")
                        print(f"    Source: {candidate.get('source', 'N/A')}")
                    else:
                        print(f"  Candidate {j+1}: Index {idx} out of range")
                        
            except Exception as e:
                print(f"  Error processing query: {e}")
                
    except Exception as e:
        print(f"Error in vector search: {e}")

if __name__ == "__main__":
    print("Extracting queries from Excel file...")
    queries = extract_full_queries()
    
    if queries:
        print(f"\nFound {len(queries)} queries")
        test_queries_with_simple_search(queries)
    else:
        print("No queries found in the Excel file.") 