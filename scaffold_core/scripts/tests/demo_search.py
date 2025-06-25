#!/usr/bin/env python3
"""
Example usage of extracted PDF text for searching and analysis
"""

import json
import re
from collections import Counter

def search_documents(search_term, max_results=10):
    """Search across all documents and return matching results"""
    with open("outputs/full_text_extracts.json", 'r', encoding='utf-8') as f:
        extracts = json.load(f)
    
    results = []
    for extract in extracts:
        text = extract.get('full_text', '')
        if search_term.lower() in text.lower():
            # Find context around the match
            match = re.search(re.escape(search_term), text, re.IGNORECASE)
            if match:
                start = max(0, match.start() - 150)
                end = min(len(text), match.end() + 150)
                context = text[start:end].strip()
                
                results.append({
                    'document': extract['document_id'],
                    'folder': extract['folder'],
                    'word_count': extract.get('word_count', 0),
                    'context': context
                })
    
    return results[:max_results]

def get_most_common_words(min_length=5, top_n=20):
    """Find most common words across all documents"""
    with open("outputs/full_text_extracts.json", 'r', encoding='utf-8') as f:
        extracts = json.load(f)
    
    all_words = []
    for extract in extracts:
        text = extract.get('full_text', '').lower()
        # Remove punctuation and split into words
        words = re.findall(r'\b[a-z]{' + str(min_length) + ',}\b', text)
        all_words.extend(words)
    
    # Filter out common words
    stop_words = {'the', 'and', 'that', 'have', 'for', 'not', 'with', 'you', 
                  'this', 'but', 'his', 'from', 'they', 'she', 'her', 'been', 
                  'than', 'its', 'are', 'was', 'one', 'our', 'had', 'would',
                  'there', 'what', 'your', 'when', 'him', 'more', 'can', 'will',
                  'who', 'oil', 'has', 'may', 'use', 'could', 'which', 'their',
                  'said', 'each', 'into', 'were', 'about', 'other'}
    
    filtered_words = [word for word in all_words if word not in stop_words]
    
    return Counter(filtered_words).most_common(top_n)

def main():
    print("PDF Text Search and Analysis Demo")
    print("=" * 50)
    
    # Example searches
    search_terms = [
        "sustainability",
        "climate change", 
        "engineering education",
        "student engagement"
    ]
    
    for term in search_terms:
        print(f"\n🔍 Searching for: '{term}'")
        print("-" * 30)
        
        results = search_documents(term, max_results=3)
        
        if results:
            print(f"Found {len(results)} documents mentioning '{term}':")
            for i, result in enumerate(results, 1):
                print(f"\n{i}. {result['document']}")
                print(f"   📁 {result['folder']}")
                print(f"   📄 {result['word_count']:,} words")
                print(f"   💬 ...{result['context']}...")
        else:
            print(f"No documents found mentioning '{term}'")
    
    # Show most common terms
    print(f"\n📊 Most Common Terms (5+ letters)")
    print("-" * 30)
    common_words = get_most_common_words()
    
    for word, count in common_words[:15]:
        print(f"{word:20} {count:4d} occurrences")
    
    print(f"\n✨ Summary")
    print(f"Your extracted corpus contains rich information about:")
    top_5_words = [word for word, count in common_words[:5]]
    print(f"• {', '.join(top_5_words)}")
    
    print(f"\n💡 Usage Ideas:")
    print("• Build a semantic search system using embeddings")
    print("• Analyze themes and trends across documents")
    print("• Create summaries of key topics")
    print("• Extract specific information (authors, methodologies, findings)")
    print("• Compare content across different folders/sources")

if __name__ == "__main__":
    main()
