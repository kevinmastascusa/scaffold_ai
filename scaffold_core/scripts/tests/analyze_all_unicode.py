#!/usr/bin/env python3
"""
Script to comprehensively analyze unicode in ALL documents from vector and original outputs.
"""

import json
import os
import unicodedata
from collections import Counter, defaultdict
import re

def analyze_unicode_characters(text):
    """Analyze unicode characters in full text"""
    char_categories = defaultdict(int)
    problematic_chars = []
    control_chars = []
    
    for char in text:
        category = unicodedata.category(char)
        char_categories[category] += 1
        
        # Check for problematic characters
        if ord(char) > 127:  # Non-ASCII
            char_name = unicodedata.name(char, f"U+{ord(char):04X}")
            if char not in [c[0] for c in problematic_chars]:  # Avoid duplicates
                problematic_chars.append((char, char_name, ord(char)))
        
        # Check for control characters
        if category.startswith('C'):
            if char not in [c[0] for c in control_chars]:
                control_chars.append((char, unicodedata.name(char, f"U+{ord(char):04X}"), ord(char)))
    
    return {
        'categories': dict(char_categories),
        'non_ascii_chars': problematic_chars,
        'control_chars': control_chars
    }

def comprehensive_unicode_analysis(data, data_name):
    """Analyze unicode quality across ALL documents"""
    
    print(f"\n" + "=" * 80)
    print(f"COMPREHENSIVE UNICODE ANALYSIS - {data_name.upper()}")
    print("=" * 80)
    
    total_unicode_issues = 0
    total_text_length = 0
    all_unique_chars = set()
    documents_with_issues = 0
    
    print(f"Analyzing ALL {len(data)} documents...")
    
    for i, item in enumerate(data):
        # Get text content
        text = item.get('text', item.get('full_text', ''))
        if not text:
            continue
            
        doc_id = item.get('document_id', f'item_{i}')
        folder = item.get('folder', 'Unknown')
        
        unicode_analysis = analyze_unicode_characters(text)
        total_text_length += len(text)
        
        doc_issues = len(unicode_analysis['non_ascii_chars'])
        if doc_issues > 0:
            documents_with_issues += 1
            
            # Show first few problematic documents for reference
            if documents_with_issues <= 10:
                print(f"\nDocument {i+1}: {doc_id}")
                print(f"  Folder: {folder}")
                print(f"  Text length: {len(text):,} characters")
                print(f"  Non-ASCII characters: {doc_issues}")
                for char, name, code in unicode_analysis['non_ascii_chars'][:5]:
                    print(f"    '{char}' ({name}) - U+{code:04X}")
                if len(unicode_analysis['non_ascii_chars']) > 5:
                    print(f"    ... and {len(unicode_analysis['non_ascii_chars']) - 5} more")
        
        # Add unique characters to global set
        for char, name, code in unicode_analysis['non_ascii_chars']:
            all_unique_chars.add((char, name, code))
        
        total_unicode_issues += doc_issues
        
        # Show progress
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(data)} documents... ({documents_with_issues} with unicode issues so far)")
    
    print(f"\n" + "=" * 60)
    print(f"{data_name} FINAL RESULTS:")
    print("=" * 60)
    print(f"📊 Documents analyzed: {len(data):,}")
    print(f"📊 Documents with unicode issues: {documents_with_issues:,}")
    print(f"📊 Total text analyzed: {total_text_length:,} characters")
    print(f"📊 Total unicode character instances: {total_unicode_issues:,}")
    print(f"📊 Unique non-ASCII character types: {len(all_unique_chars)}")
    
    if all_unique_chars:
        print(f"\n🔍 ALL UNIQUE NON-ASCII CHARACTERS FOUND:")
        for char, name, code in sorted(all_unique_chars, key=lambda x: x[2]):
            print(f"    '{char}' ({name}) - U+{code:04X}")
    else:
        print(f"\n✅ NO NON-ASCII CHARACTERS FOUND - PERFECT ASCII TEXT!")
    
    return total_unicode_issues, len(all_unique_chars)

def main():
    """Main analysis function"""
    
    print("COMPREHENSIVE UNICODE ANALYSIS - ALL DOCUMENTS")
    print("=" * 80)
    
    # Define file paths
    files_to_analyze = [
        (r"c:\Users\dlaev\OneDrive\Documents\GitHub\scaffold_ai\outputs\chunked_text_extracts.json", "Original Chunked"),
        (r"c:\Users\dlaev\OneDrive\Documents\GitHub\scaffold_ai\outputs\full_text_extracts.json", "Original Full Text"),
        (r"c:\Users\dlaev\OneDrive\Documents\GitHub\scaffold_ai\vector_outputs\processed_1.json", "Vector Output")
    ]
    
    results = {}
    
    for file_path, file_name in files_to_analyze:
        if os.path.exists(file_path):
            print(f"✓ Found: {file_name}")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                total_issues, unique_chars = comprehensive_unicode_analysis(data, file_name)
                results[file_name] = {
                    'documents': len(data),
                    'total_issues': total_issues,
                    'unique_chars': unique_chars
                }
                
            except Exception as e:
                print(f"✗ Error analyzing {file_name}: {e}")
        else:
            print(f"✗ Missing: {file_name}")
    
    # Final comparison
    if len(results) > 1:
        print(f"\n" + "=" * 80)
        print("FINAL COMPARISON SUMMARY")
        print("=" * 80)
        
        for name, stats in results.items():
            print(f"{name}:")
            print(f"  📄 Documents: {stats['documents']:,}")
            print(f"  🔤 Unicode issues: {stats['total_issues']:,}")
            print(f"  🆔 Unique chars: {stats['unique_chars']}")
            print()
    
    print("ANALYSIS COMPLETE!")

if __name__ == "__main__":
    main()
