#!/usr/bin/env python3
"""
Script to check unicode and compare vector output with existing extraction output.
"""

import json
import os
import unicodedata
from collections import Counter, defaultdict
import re

def analyze_unicode_characters(text, sample_size=1000):
    """Analyze unicode characters in text"""
    # Take a sample for performance
    sample_text = text[:sample_size] if len(text) > sample_size else text
    
    char_categories = defaultdict(int)
    problematic_chars = []
    control_chars = []
    math_symbols = []
    
    # Math symbol ranges
    math_ranges = [
        (0x2200, 0x22FF),  # Mathematical Operators
        (0x2300, 0x23FF),  # Miscellaneous Technical
        (0x27C0, 0x27EF),  # Miscellaneous Mathematical Symbols-A
        (0x2980, 0x29FF),  # Miscellaneous Mathematical Symbols-B
        (0x2A00, 0x2AFF),  # Supplemental Mathematical Operators
    ]
    
    for char in sample_text:
        category = unicodedata.category(char)
        char_categories[category] += 1
        
        # Check for problematic characters
        if ord(char) > 127:  # Non-ASCII
            char_name = unicodedata.name(char, f"U+{ord(char):04X}")
            if char not in [c[0] for c in problematic_chars]:  # Avoid duplicates
                problematic_chars.append((char, char_name, ord(char)))
                
                # Check if it's a math symbol
                code_point = ord(char)
                is_math = any(start <= code_point <= end for start, end in math_ranges)
                if is_math and char not in [c[0] for c in math_symbols]:
                    math_symbols.append((char, char_name, ord(char)))
                    
        # Check for control characters
        if category.startswith('C'):
            if char not in [c[0] for c in control_chars]:
                control_chars.append((char, unicodedata.name(char, f"U+{ord(char):04X}"), ord(char)))
    
    return {
        'categories': dict(char_categories),
        'non_ascii_chars': problematic_chars,  # All unique non-ASCII chars
        'control_chars': control_chars,  # All unique control chars
        'math_symbols': math_symbols  # Mathematical symbols found
    }

def compare_extractions(original_path, vector_path):
    """Compare original extraction with vector output"""
    
    print("=" * 80)
    print("EXTRACTION COMPARISON ANALYSIS")
    print("=" * 80)
    
    # Load original data
    try:
        with open(original_path, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
        print(f"✓ Loaded original data: {len(original_data)} items from {original_path}")
    except Exception as e:
        print(f"✗ Error loading original data: {e}")
        return None
    
    # Load vector data with enhanced error handling
    try:
        with open(vector_path, 'r', encoding='utf-8') as f:
            vector_data = json.load(f)
        
        # Handle different data structures
        if isinstance(vector_data, dict):
            if 'documents' in vector_data:
                # Math-aware format - convert to list
                doc_list = []
                for doc_path, doc_info in vector_data['documents'].items():
                    if doc_info.get('extraction_metadata', {}).get('success', False):
                        chunks = doc_info.get('chunks', [])
                        for chunk in chunks:
                            chunk_item = {
                                'document_id': doc_info['filename'],
                                'text': chunk['text'],
                                'math_analysis': chunk.get('math_analysis', {}),
                                'unicode_analysis': chunk.get('unicode_analysis', {}),
                                'chunk_metadata': chunk.get('chunk_metadata', {})
                            }
                            doc_list.append(chunk_item)
                vector_data = doc_list
                print(f"✓ Converted math-aware data: {len(vector_data)} chunks from {vector_path}")
            elif 'processing_metadata' in vector_data:
                # Other structured format
                print(f"✓ Loaded structured data from {vector_path}")
            else:
                print(f"✓ Loaded vector data: {len(vector_data)} items from {vector_path}")
        else:
            print(f"✓ Loaded vector data: {len(vector_data)} items from {vector_path}")
            
    except UnicodeDecodeError as e:
        print(f"✗ Unicode error loading vector data: {e}")
        print("Trying with different encoding...")
        try:
            with open(vector_path, 'r', encoding='utf-8-sig') as f:
                vector_data = json.load(f)
            print(f"✓ Loaded vector data with utf-8-sig: {len(vector_data)} items")
        except Exception as e2:
            print(f"✗ Still failed: {e2}")
            return None
    except Exception as e:
        print(f"✗ Error loading vector data: {e}")
        return None    
    print("\n" + "=" * 40)
    print("DATA STRUCTURE COMPARISON")
    print("=" * 40)
    
    # Compare data structures - handle empty data
    if original_data and vector_data and len(original_data) > 0 and len(vector_data) > 0:
        orig_keys = set(original_data[0].keys()) if original_data else set()
        vector_keys = set(vector_data[0].keys()) if vector_data else set()
        
        print(f"Original data keys: {sorted(orig_keys)}")
        print(f"Vector data keys: {sorted(vector_keys)}")
        print(f"Common keys: {sorted(orig_keys & vector_keys)}")
        print(f"Original-only keys: {sorted(orig_keys - vector_keys)}")
        print(f"Vector-only keys: {sorted(vector_keys - orig_keys)}")
    else:
        print("⚠️ Cannot compare data structures - one or both datasets are empty")
        if not original_data or len(original_data) == 0:
            print("  - Original data is empty")
        if not vector_data or len(vector_data) == 0:
            print("  - Vector data is empty")
    
    print("\n" + "=" * 40)
    print("DOCUMENT COVERAGE COMPARISON")
    print("=" * 40)
    
    # Get document lists
    orig_docs = set()
    vector_docs = set()
    
    for item in original_data:
        doc_id = item.get('document_id', item.get('filename', 'unknown'))
        orig_docs.add(doc_id)
    
    for item in vector_data:
        doc_id = item.get('document_id', item.get('filename', 'unknown'))
        vector_docs.add(doc_id)
    
    print(f"Documents in original: {len(orig_docs)}")
    print(f"Documents in vector: {len(vector_docs)}")
    print(f"Common documents: {len(orig_docs & vector_docs)}")
    
    missing_in_vector = orig_docs - vector_docs
    missing_in_original = vector_docs - orig_docs
    
    if missing_in_vector:
        print(f"Missing in vector ({len(missing_in_vector)}): {list(missing_in_vector)[:5]}...")
    
    if missing_in_original:
        print(f"Missing in original ({len(missing_in_original)}): {list(missing_in_original)[:5]}...")
    
    return original_data, vector_data

def analyze_unicode_quality(data, data_name, num_samples=10):
    """Analyze unicode quality in the data"""
    
    print(f"\n" + "=" * 60)
    print(f"UNICODE ANALYSIS - {data_name.upper()}")
    print("=" * 60)
    
    total_unicode_issues = 0
    total_text_length = 0
    
    # Sample documents for analysis
    sample_size = min(num_samples, len(data))
    samples = data[:sample_size] if len(data) > sample_size else data
    
    for i, item in enumerate(samples):
        # Get text content
        text = item.get('text', item.get('full_text', ''))
        if not text:
            continue
            
        doc_id = item.get('document_id', f'item_{i}')
        folder = item.get('folder', 'Unknown')
        
        print(f"\nDocument {i+1}: {doc_id}")
        print(f"Folder: {folder}")
        unicode_analysis = analyze_unicode_characters(text)
        total_text_length += len(text)
        
        print(f"  Text length: {len(text):,} characters")
        print(f"  Character categories: {unicode_analysis['categories']}")
        
        if unicode_analysis['math_symbols']:
            print(f"  Mathematical symbols found: {len(unicode_analysis['math_symbols'])}")
            # Show math symbols
            for char, name, code in unicode_analysis['math_symbols'][:5]:
                print(f"    '{char}' ({name}) - U+{code:04X}")
            if len(unicode_analysis['math_symbols']) > 5:
                print(f"    ... and {len(unicode_analysis['math_symbols']) - 5} more math symbols")
        
        if unicode_analysis['non_ascii_chars']:
            non_math_chars = [char for char in unicode_analysis['non_ascii_chars'] 
                            if char not in [c[0] for c in unicode_analysis['math_symbols']]]
            if non_math_chars:
                print(f"  Non-math Unicode characters: {len(non_math_chars)}")
                # Show non-math Unicode characters
                chars_to_show = non_math_chars[:5] if len(non_math_chars) > 5 else non_math_chars
                for char, name, code in chars_to_show:
                    print(f"    '{char}' ({name}) - U+{code:04X}")
                if len(non_math_chars) > 5:
                    print(f"    ... and {len(non_math_chars) - 5} more characters")
            total_unicode_issues += len(unicode_analysis['non_ascii_chars'])
        
        if unicode_analysis['control_chars']:
            print(f"  Control characters found: {len(unicode_analysis['control_chars'])}")
            # Show all control characters
            for char, name, code in unicode_analysis['control_chars']:
                print(f"    {repr(char)} ({name}) - U+{code:04X}")
                print(f"    {repr(char)} ({name}) - U+{code:04X}")
        
        # Check for common text extraction issues
        text_sample = text[:500]
        if '�' in text_sample:
            print("  ⚠️  WARNING: Replacement characters (�) found - encoding issues!")
        if re.search(r'[^\x00-\x7F]{10,}', text_sample):
            print("  ⚠️  WARNING: Long sequences of non-ASCII characters found")
    
    print(f"\n{data_name} Summary:")
    print(f"  Total text analyzed: {total_text_length:,} characters")
    print(f"  Unicode issues found: {total_unicode_issues}")
    
    return total_unicode_issues

def compare_text_samples(original_data, vector_data):
    """Compare actual text content between original and vector data"""
    
    print(f"\n" + "=" * 60)
    print("TEXT CONTENT COMPARISON")
    print("=" * 60)
    
    # Find a common document to compare
    orig_docs = {item.get('document_id', 'unknown'): item for item in original_data}
    vector_docs = {item.get('document_id', 'unknown'): item for item in vector_data}
    
    common_docs = set(orig_docs.keys()) & set(vector_docs.keys())
    
    if common_docs:
        sample_doc = list(common_docs)[0]
        orig_item = orig_docs[sample_doc]
        
        # Find first vector chunk for this document
        vector_items = [item for item in vector_data if item.get('document_id') == sample_doc]
        
        if vector_items:
            vector_item = vector_items[0]
            
            orig_text = orig_item.get('text', orig_item.get('full_text', ''))[:500]
            vector_text = vector_item.get('text', '')[:500]
            
            print(f"Sample document: {sample_doc}")
            print(f"Original text preview (first 500 chars):")
            print(f"  '{orig_text[:200]}...'")
            print(f"Vector text preview (first 500 chars):")
            print(f"  '{vector_text[:200]}...'")
            
            # Check if they match
            if orig_text.strip() == vector_text.strip():
                print("✓ Text content appears to match")
            else:
                print("⚠️  Text content differs - may be due to chunking differences")
        else:
            print(f"No vector chunks found for {sample_doc}")
    else:
        print("No common documents found for text comparison")

def analyze_math_aware_quality(data, data_name, num_samples=10):
    """Analyze math-aware extraction quality"""
    
    print(f"\n" + "=" * 60)
    print(f"MATH-AWARE ANALYSIS - {data_name.upper()}")
    print("=" * 60)
    
    total_math_content = 0
    total_unicode_chars = 0
    total_text_length = 0
    math_documents = 0
    
    # Handle different data structures
    if isinstance(data, dict) and 'documents' in data:
        # New math-aware format
        documents = data['documents']
        metadata = data.get('processing_metadata', {})
        
        print(f"Processing metadata:")
        print(f"  Total files: {metadata.get('total_files', 'Unknown')}")
        print(f"  Successful extractions: {metadata.get('successful_extractions', 'Unknown')}")
        print(f"  Math documents: {metadata.get('total_math_documents', 'Unknown')}")
        print(f"  Unicode documents: {metadata.get('total_unicode_documents', 'Unknown')}")
        
        # Convert to list for processing
        doc_list = []
        for doc_path, doc_data in documents.items():
            if doc_data.get('extraction_metadata', {}).get('success', False):
                # Handle chunked data
                if 'chunks' in doc_data:
                    for i, chunk in enumerate(doc_data['chunks']):
                        chunk_item = {
                            'document_id': doc_data['filename'],
                            'text': chunk['text'],
                            'chunk_id': i,
                            'math_analysis': chunk.get('math_analysis', {}),
                            'unicode_analysis': chunk.get('unicode_analysis', {}),
                            'chunk_metadata': chunk.get('chunk_metadata', {})
                        }
                        doc_list.append(chunk_item)
                else:
                    # Handle full text data
                    doc_item = {
                        'document_id': doc_data['filename'],
                        'text': doc_data.get('full_text', ''),
                        'math_analysis': doc_data.get('math_analysis', {}),
                        'unicode_analysis': doc_data.get('unicode_analysis', {})
                    }
                    doc_list.append(doc_item)
        
        data = doc_list
    
    # Sample documents for analysis
    sample_size = min(num_samples, len(data))
    samples = data[:sample_size] if len(data) > sample_size else data
    
    for i, item in enumerate(samples):
        # Get text content
        text = item.get('text', '')
        if not text:
            continue
            
        doc_id = item.get('document_id', f'item_{i}')
        total_text_length += len(text)
        
        print(f"\nDocument {i+1}: {doc_id}")
        
        # Analyze math content if available
        math_analysis = item.get('math_analysis', {})
        if math_analysis:
            print(f"  Math Analysis:")
            print(f"    Has math: {math_analysis.get('has_math', False)}")
            print(f"    Math symbols: {len(math_analysis.get('math_symbols', []))}")
            print(f"    Math density: {math_analysis.get('math_density', 0):.4f}")
            print(f"    Equations: {len(math_analysis.get('equations', []))}")
            print(f"    Formulas: {len(math_analysis.get('formulas', []))}")
            print(f"    Units: {len(math_analysis.get('units', []))}")
            print(f"    Statistics: {len(math_analysis.get('statistics', []))}")
            
            # Show some math symbols if found
            math_symbols = math_analysis.get('math_symbols', [])
            if math_symbols:
                print(f"    Sample math symbols: {math_symbols[:10]}")
            
            # Show some equations if found
            equations = math_analysis.get('equations', [])
            if equations:
                print(f"    Sample equations:")
                for eq in equations[:3]:
                    print(f"      '{eq.get('text', eq)}'")
            
            if math_analysis.get('has_math', False):
                math_documents += 1
                total_math_content += math_analysis.get('math_symbol_count', 0)
        
        # Analyze Unicode content if available
        unicode_analysis = item.get('unicode_analysis', {})
        if unicode_analysis:
            unicode_chars = unicode_analysis.get('unicode_chars', 0)
            total_unicode_chars += unicode_chars
            print(f"  Unicode Analysis:")
            print(f"    Unicode characters: {unicode_chars}")
            print(f"    Total characters: {unicode_analysis.get('total_chars', 0)}")
            if unicode_analysis.get('total_chars', 0) > 0:
                unicode_ratio = unicode_chars / unicode_analysis['total_chars']
                print(f"    Unicode ratio: {unicode_ratio:.4f}")
            
            # Show Unicode categories
            categories = unicode_analysis.get('unicode_categories', {})
            if categories:
                print(f"    Unicode categories: {categories}")
        
        # Check chunk metadata if available
        chunk_metadata = item.get('chunk_metadata', {})
        if chunk_metadata:
            print(f"  Chunk Metadata:")
            print(f"    Has math: {chunk_metadata.get('has_math', False)}")
            print(f"    Math density: {chunk_metadata.get('math_density', 0):.4f}")
            print(f"    Unicode ratio: {chunk_metadata.get('unicode_ratio', 0):.4f}")
    
    print(f"\n{data_name} Summary:")
    print(f"  Total text analyzed: {total_text_length:,} characters")
    print(f"  Documents with math: {math_documents}")
    print(f"  Total math symbols: {total_math_content}")
    print(f"  Total Unicode characters: {total_unicode_chars}")
    
    return {
        'total_text_length': total_text_length,
        'math_documents': math_documents,
        'total_math_content': total_math_content,
        'total_unicode_chars': total_unicode_chars
    }

def compare_math_content(original_data, math_aware_data):
    """Compare mathematical content between original and math-aware extractions"""
    
    print(f"\n" + "=" * 60)
    print("MATHEMATICAL CONTENT COMPARISON")
    print("=" * 60)
    
    # Find a common document to compare
    orig_docs = {}
    math_docs = {}
    
    # Build document mappings
    for item in original_data:
        doc_id = item.get('document_id', 'unknown')
        if doc_id not in orig_docs:
            orig_docs[doc_id] = []
        orig_docs[doc_id].append(item)
    
    for item in math_aware_data:
        doc_id = item.get('document_id', 'unknown')
        if doc_id not in math_docs:
            math_docs[doc_id] = []
        math_docs[doc_id].append(item)
    
    common_docs = set(orig_docs.keys()) & set(math_docs.keys())
    
    if common_docs:
        sample_doc = list(common_docs)[0]
        print(f"Comparing mathematical content in: {sample_doc}")
        
        # Get text samples
        orig_text = ""
        math_text = ""
        
        for item in orig_docs[sample_doc][:3]:  # First 3 chunks
            orig_text += item.get('text', '') + " "
        
        for item in math_docs[sample_doc][:3]:  # First 3 chunks
            math_text += item.get('text', '') + " "
        
        orig_text = orig_text.strip()
        math_text = math_text.strip()
        
        print(f"\nOriginal extraction (first 300 chars):")
        print(f"  '{orig_text[:300]}...'")
        print(f"\nMath-aware extraction (first 300 chars):")
        print(f"  '{math_text[:300]}...'")
        
        # Count mathematical symbols
        math_symbol_chars = set('±×÷∞∂∇∆∑∏∫√≤≥≠≈≡∈∉⊂⊃∪∩∧∨¬∀∃∴∵→←↔⇒⇔αβγδεζηθικλμνξπρστυφχψω')
        
        orig_math_count = sum(1 for char in orig_text if char in math_symbol_chars)
        math_aware_count = sum(1 for char in math_text if char in math_symbol_chars)
        
        print(f"\nMathematical symbol preservation:")
        print(f"  Original extraction: {orig_math_count} math symbols")
        print(f"  Math-aware extraction: {math_aware_count} math symbols")
        
        if math_aware_count > orig_math_count:
            improvement = math_aware_count - orig_math_count
            print(f"  ✓ Math-aware extraction preserved {improvement} additional math symbols")
        elif math_aware_count == orig_math_count:
            print(f"  = Both extractions preserved the same number of math symbols")
        else:
            print(f"  ⚠️ Math-aware extraction has fewer math symbols")
        
        # Check for specific mathematical patterns
        patterns = [
            (r'[a-zA-Z]\s*=\s*\d+\.?\d*', 'variable assignments'),
            (r'p\s*[<>=]\s*0\.\d+', 'statistical p-values'),
            (r'R²\s*=\s*0\.\d+', 'R-squared values'),
            (r'\d+\.?\d*\s*[×*]\s*10\^?[-\d]+', 'scientific notation')
        ]
        
        print(f"\nMathematical pattern preservation:")
        for pattern, description in patterns:
            orig_matches = len(re.findall(pattern, orig_text, re.IGNORECASE))
            math_matches = len(re.findall(pattern, math_text, re.IGNORECASE))
            
            print(f"  {description}:")
            print(f"    Original: {orig_matches}, Math-aware: {math_matches}")
    else:
        print("No common documents found for mathematical content comparison")

def main():
    """Main analysis function"""
    
    print("Unicode and Extraction Comparison Analysis (Including Math-Aware)")
    print("=" * 80)
    
    # Define file paths - including math-aware outputs
    original_chunked_path = r"c:\Users\dlaev\OneDrive\Documents\GitHub\scaffold_ai\outputs\chunked_text_extracts.json"
    original_full_path = r"c:\Users\dlaev\OneDrive\Documents\GitHub\scaffold_ai\outputs\full_text_extracts.json"
    vector_path = r"c:\Users\dlaev\OneDrive\Documents\GitHub\scaffold_ai\vector_outputs\processed_1.json"
    math_full_path = r"c:\Users\dlaev\OneDrive\Documents\GitHub\scaffold_ai\math_outputs\math_aware_full_extracts.json"
    math_chunked_path = r"c:\Users\dlaev\OneDrive\Documents\GitHub\scaffold_ai\math_outputs\math_aware_chunked_extracts.json"
    
    # Check which files exist
    files_to_check = [
        (original_chunked_path, "Original Chunked"),
        (original_full_path, "Original Full Text"),
        (vector_path, "Vector Output"),
        (math_full_path, "Math-Aware Full Text"),
        (math_chunked_path, "Math-Aware Chunked")
    ]
    
    available_files = []
    for file_path, file_name in files_to_check:
        if os.path.exists(file_path):
            print(f"✓ Found: {file_name} at {file_path}")
            available_files.append((file_path, file_name))
        else:
            print(f"✗ Missing: {file_name} at {file_path}")
    
    if len(available_files) < 2:
        print("\nNeed at least 2 files to compare. Please run the extraction scripts first.")
        return
    
    # Compare original chunked with vector output if both exist
    orig_chunked_path = None
    vector_data_path = None
    math_chunked_file = None
    math_full_file = None
    
    for file_path, file_name in available_files:
        if "original" in file_name.lower() and "chunked" in file_name.lower():
            orig_chunked_path = file_path
        elif "vector" in file_name.lower():
            vector_data_path = file_path
        elif "math-aware" in file_name.lower() and "chunked" in file_name.lower():
            math_chunked_file = file_path
        elif "math-aware" in file_name.lower() and "full" in file_name.lower():
            math_full_file = file_path
    
    # Compare original vs vector
    if orig_chunked_path and vector_data_path:
        print(f"\n" + "=" * 60)
        print("COMPARING: ORIGINAL CHUNKED vs VECTOR OUTPUT")
        print("=" * 60)
        result = compare_extractions(orig_chunked_path, vector_data_path)
        
        if result:
            original_data, vector_data = result
            
            # Unicode analysis
            analyze_unicode_quality(original_data, "Original Chunked", num_samples=5)
            analyze_unicode_quality(vector_data, "Vector Output", num_samples=5)
            
            # Text comparison
            compare_text_samples(original_data, vector_data)
    
    # Compare original vs math-aware
    if orig_chunked_path and math_chunked_file:
        print(f"\n" + "=" * 60)
        print("COMPARING: ORIGINAL CHUNKED vs MATH-AWARE CHUNKED")
        print("=" * 60)
        result = compare_extractions(orig_chunked_path, math_chunked_file)
        
        if result:
            original_data, math_data = result
            
            # Enhanced analysis for math-aware data
            analyze_unicode_quality(original_data, "Original Chunked", num_samples=5)
            analyze_math_aware_quality(math_data, "Math-Aware Chunked", num_samples=5)
            
            # Compare mathematical content preservation
            compare_math_content(original_data, math_data)
    
    # Analyze any other available files individually
    for file_path, file_name in available_files:
        if file_path not in [orig_chunked_path, vector_data_path, math_chunked_file]:
            try:
                print(f"\n" + "=" * 60)
                print(f"INDIVIDUAL ANALYSIS: {file_name.upper()}")
                print("=" * 60)
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if "math-aware" in file_name.lower():
                    analyze_math_aware_quality(data, file_name, num_samples=3)
                else:
                    analyze_unicode_quality(data, file_name, num_samples=3)
            except Exception as e:
                print(f"Error analyzing {file_name}: {e}")
    
    print(f"\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
