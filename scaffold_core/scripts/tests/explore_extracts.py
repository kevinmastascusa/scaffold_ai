#!/usr/bin/env python3
"""
Script to explore the extracted PDF text data.
Run this to get summaries and search through the extracted text.
"""

import json
import os
from collections import Counter
import re

def load_extracts(extract_type="full_text"):
    """Load either full_text or chunked extracts"""
    if extract_type == "full_text":
        file_path = "outputs/full_text_extracts.json"
    else:
        file_path = "outputs/chunked_text_extracts.json"
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def summarize_extracts(extracts):
    """Print a summary of the extracted data"""
    if not extracts:
        return
    
    print(f"Total extracts: {len(extracts)}")
    
    # Count by folder
    folders = Counter(extract['folder'] for extract in extracts)
    print(f"\nDocuments by folder:")
    for folder, count in folders.items():
        print(f"  {folder}: {count}")
    
    # Word count statistics
    word_counts = [extract.get('word_count', 0) for extract in extracts if 'word_count' in extract]
    if word_counts:
        print(f"\nWord count statistics:")
        print(f"  Total words: {sum(word_counts):,}")
        print(f"  Average per document: {sum(word_counts)/len(word_counts):.0f}")
        print(f"  Largest document: {max(word_counts):,} words")
        print(f"  Smallest document: {min(word_counts):,} words")

def search_text(extracts, search_term, context_chars=200):
    """Search for a term across all extracts and show context"""
    if not extracts:
        return
    
    results = []
    for extract in extracts:
        text = extract.get('full_text', extract.get('text', ''))
        if search_term.lower() in text.lower():
            # Find all occurrences
            for match in re.finditer(re.escape(search_term), text, re.IGNORECASE):
                start = max(0, match.start() - context_chars)
                end = min(len(text), match.end() + context_chars)
                context = text[start:end]
                
                results.append({
                    'document': extract['document_id'],
                    'folder': extract['folder'],
                    'context': context,
                    'position': match.start()
                })
    
    print(f"\nFound {len(results)} occurrences of '{search_term}':")
    for i, result in enumerate(results[:10]):  # Show first 10 results
        print(f"\n{i+1}. {result['document']} ({result['folder']})")
        print(f"   ...{result['context']}...")
        if i >= 9 and len(results) > 10:
            print(f"\n   ... and {len(results) - 10} more results")
            break

def list_documents(extracts, show_details=True):
    """List all documents with their metadata"""
    if not extracts:
        return
    
    print(f"\nDocument list:")
    for i, extract in enumerate(extracts, 1):
        print(f"{i:3d}. {extract['document_id']}")
        if show_details:
            folder = extract.get('folder', 'Unknown')
            word_count = extract.get('word_count', 'Unknown')
            pages = extract.get('metadata', {}).get('total_pages', 'Unknown')
            print(f"     Folder: {folder}")
            print(f"     Words: {word_count}, Pages: {pages}")
            if 'metadata' in extract and extract['metadata'].get('title'):
                print(f"     Title: {extract['metadata']['title']}")
            print()

def main():
    print("PDF Text Extract Explorer")
    print("=" * 50)
    
    # Load full text extracts
    extracts = load_extracts("full_text")
    if not extracts:
        print("No full text extracts found. Make sure ChunkTest.py has been run.")
        return
    
    while True:
        print("\nOptions:")
        print("1. Summary of extracts")
        print("2. List all documents")
        print("3. Search text")
        print("4. Show sample document")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            summarize_extracts(extracts)
        
        elif choice == "2":
            list_documents(extracts)
        
        elif choice == "3":
            search_term = input("Enter search term: ").strip()
            if search_term:
                search_text(extracts, search_term)
        
        elif choice == "4":
            if extracts:
                print(f"\nSample document: {extracts[0]['document_id']}")
                print(f"Folder: {extracts[0]['folder']}")
                text = extracts[0].get('full_text', '')
                print(f"First 500 characters:")
                print(text[:500] + "..." if len(text) > 500 else text)
        
        elif choice == "5":
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
