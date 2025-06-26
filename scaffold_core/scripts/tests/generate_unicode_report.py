#!/usr/bin/env python3
"""
Generate a comprehensive unicode and text analysis report for chunked_text_extracts.json.
Writes results to outputs/unicode_report.txt
"""
import sys
import os
import re
import unicodedata
from collections import defaultdict, Counter

# Add workspace root to Python path for module imports
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, root_dir)

import json
from scaffold_core.config import CHUNKED_TEXT_EXTRACTS_JSON, UNICODE_REPORT_TXT

# Paths are now defined in config.py
CHUNKS_JSON = str(CHUNKED_TEXT_EXTRACTS_JSON)
REPORT_TXT = str(UNICODE_REPORT_TXT)


def analyze_unicode_content(text):
    """Analyze Unicode content in text."""
    total_chars = len(text)
    unicode_chars = 0
    unicode_categories = defaultdict(int)
    scripts = set()
    problematic_chars = set()
    
    for char in text:
        if ord(char) > 127:  # Non-ASCII
            unicode_chars += 1
            category = unicodedata.category(char)
            unicode_categories[category] += 1
            
            # Check for problematic characters
            if category in ['Cc', 'Cf', 'Cs', 'Co', 'Cn']:
                problematic_chars.add(char)
            
            # Get script information
            try:
                script = unicodedata.name(char).split()[-1]
                scripts.add(script)
            except:
                pass
    
    return {
        'total_chars': total_chars,
        'unicode_chars': unicode_chars,
        'unicode_ratio': unicode_chars / total_chars if total_chars > 0 else 0,
        'unicode_categories': dict(unicode_categories),
        'scripts': list(scripts),
        'problematic_chars': list(problematic_chars)
    }


def find_combined_words(text):
    """Find combined words in text."""
    combined_words = []
    
    # Patterns for combined words
    patterns = [
        r'\b[a-z]+[A-Z][a-z]+\b',  # camelCase
        r'\b[A-Z]+[a-z]+[A-Z][a-z]+\b',  # PascalCase
        r'\b[a-z]+[A-Z]{2,}[a-z]*\b',  # words with multiple caps
        r'\b[A-Z]+[a-z]+[A-Z]+\b',  # mixed case
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            if len(match) > 8:  # Only longer combined words
                combined_words.append(match)
    
    return combined_words


def find_sustainability_combined_words(text):
    """Find sustainability-related combined words."""
    sustainability_patterns = [
        r'\b\w*sustainability\w*\b',
        r'\b\w*environmental\w*\b', 
        r'\b\w*engineering\w*\b',
        r'\b\w*education\w*\b',
        r'\b\w*curriculum\w*\b',
        r'\b\w*pedagogy\w*\b'
    ]
    
    sustainability_combined = []
    for pattern in sustainability_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            if len(match) > 12:  # Likely combined
                sustainability_combined.append(match)
    
    return sustainability_combined


def main():
    print("Generating comprehensive Unicode and text analysis report...")
    
    # Load chunks
    with open(CHUNKS_JSON, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    print(f"Loaded {len(chunks)} chunks for analysis...")
    
    # Initialize counters
    total_chars = 0
    total_unicode_chars = 0
    all_unicode_categories = defaultdict(int)
    all_scripts = set()
    all_problematic_chars = set()
    all_combined_words = []
    all_sustainability_combined = []
    document_stats = {}
    
    # Analyze each chunk
    for i, chunk in enumerate(chunks):
        if i % 1000 == 0:
            print(f"Processing chunk {i+1}/{len(chunks)}...")
        
        doc_id = chunk.get('document_id', 'unknown')
        chunk_id = chunk.get('chunk_id', f'chunk_{i}')
        page = chunk.get('start_page', 'unknown')
        text = chunk.get('text', '')
        
        # Unicode analysis
        uni_info = analyze_unicode_content(text)
        
        # Combined words analysis
        combined_words = find_combined_words(text)
        sustainability_combined = find_sustainability_combined_words(text)
        
        # Update global counters
        total_chars += uni_info['total_chars']
        total_unicode_chars += uni_info['unicode_chars']
        
        for cat, count in uni_info['unicode_categories'].items():
            all_unicode_categories[cat] += count
        
        all_scripts.update(uni_info['scripts'])
        all_problematic_chars.update(uni_info['problematic_chars'])
        all_combined_words.extend(combined_words)
        all_sustainability_combined.extend(sustainability_combined)
        
        # Update document stats
        if doc_id not in document_stats:
            document_stats[doc_id] = {
                'chunks': 0,
                'total_chars': 0,
                'unicode_chars': 0,
                'combined_words': [],
                'sustainability_combined': []
            }
        
        doc_stats = document_stats[doc_id]
        doc_stats['chunks'] += 1
        doc_stats['total_chars'] += uni_info['total_chars']
        doc_stats['unicode_chars'] += uni_info['unicode_chars']
        doc_stats['combined_words'].extend(combined_words)
        doc_stats['sustainability_combined'].extend(sustainability_combined)
    
    # Generate report
    with open(REPORT_TXT, 'w', encoding='utf-8') as out:
        out.write('COMPREHENSIVE UNICODE AND TEXT ANALYSIS REPORT\n')
        out.write('=' * 60 + '\n\n')
        
        # Overall statistics
        out.write('OVERALL STATISTICS\n')
        out.write('-' * 20 + '\n')
        out.write(f'Total chunks analyzed: {len(chunks):,}\n')
        out.write(f'Total characters: {total_chars:,}\n')
        out.write(f'Total Unicode characters: {total_unicode_chars:,}\n')
        out.write(f'Unicode percentage: {total_unicode_chars/total_chars*100:.2f}%\n')
        out.write(f'Total documents: {len(document_stats)}\n\n')
        
        # Unicode categories
        out.write('UNICODE CATEGORIES\n')
        out.write('-' * 20 + '\n')
        for category, count in sorted(all_unicode_categories.items()):
            category_name = {
                'Ll': 'Lowercase Letter',
                'Lu': 'Uppercase Letter', 
                'Lt': 'Titlecase Letter',
                'Lm': 'Modifier Letter',
                'Lo': 'Other Letter',
                'Mn': 'Nonspacing Mark',
                'Mc': 'Spacing Mark',
                'Me': 'Enclosing Mark',
                'Nd': 'Decimal Number',
                'Nl': 'Letter Number',
                'No': 'Other Number',
                'Pc': 'Connector Punctuation',
                'Pd': 'Dash Punctuation',
                'Ps': 'Open Punctuation',
                'Pe': 'Close Punctuation',
                'Pi': 'Initial Punctuation',
                'Pf': 'Final Punctuation',
                'Po': 'Other Punctuation',
                'Sm': 'Math Symbol',
                'Sc': 'Currency Symbol',
                'Sk': 'Modifier Symbol',
                'So': 'Other Symbol',
                'Zs': 'Space Separator',
                'Zl': 'Line Separator',
                'Zp': 'Paragraph Separator',
                'Cc': 'Control',
                'Cf': 'Format',
                'Cs': 'Surrogate',
                'Co': 'Private Use',
                'Cn': 'Unassigned'
            }.get(category, category)
            out.write(f'{category} ({category_name}): {count:,}\n')
        out.write('\n')
        
        # Scripts
        if all_scripts:
            out.write('UNICODE SCRIPTS\n')
            out.write('-' * 20 + '\n')
            for script in sorted(all_scripts):
                out.write(f'{script}\n')
            out.write('\n')
        
        # Problematic characters
        if all_problematic_chars:
            out.write('PROBLEMATIC CHARACTERS\n')
            out.write('-' * 20 + '\n')
            for char in sorted(all_problematic_chars):
                out.write(f'U+{ord(char):04X}: {repr(char)}\n')
            out.write('\n')
        
        # Combined words analysis
        out.write('COMBINED WORDS ANALYSIS\n')
        out.write('-' * 20 + '\n')
        combined_word_counts = Counter(all_combined_words)
        out.write(f'Total combined words found: {len(all_combined_words):,}\n')
        out.write(f'Unique combined words: {len(combined_word_counts):,}\n\n')
        
        out.write('Most common combined words:\n')
        for word, count in combined_word_counts.most_common(50):
            out.write(f'  "{word}": {count} occurrences\n')
        out.write('\n')
        
        # Sustainability combined words
        sustainability_counts = Counter(all_sustainability_combined)
        out.write('SUSTAINABILITY-RELATED COMBINED WORDS\n')
        out.write('-' * 40 + '\n')
        out.write(f'Total sustainability combined words: {len(all_sustainability_combined):,}\n')
        out.write(f'Unique sustainability combined words: {len(sustainability_counts):,}\n\n')
        
        for word, count in sustainability_counts.most_common(100):
            out.write(f'  "{word}": {count} occurrences\n')
        out.write('\n')
        
        # Document-level analysis
        out.write('DOCUMENT-LEVEL ANALYSIS\n')
        out.write('-' * 20 + '\n')
        out.write('Documents with highest Unicode content:\n')
        doc_unicode_ratios = []
        for doc_id, stats in document_stats.items():
            if stats['total_chars'] > 0:
                ratio = stats['unicode_chars'] / stats['total_chars']
                doc_unicode_ratios.append((doc_id, ratio, stats))
        
        for doc_id, ratio, stats in sorted(doc_unicode_ratios, key=lambda x: x[1], reverse=True)[:20]:
            out.write(f'  {doc_id}: {ratio*100:.2f}% Unicode ({stats["unicode_chars"]:,}/{stats["total_chars"]:,} chars)\n')
        out.write('\n')
        
        out.write('Documents with most combined words:\n')
        doc_combined_counts = []
        for doc_id, stats in document_stats.items():
            combined_count = len(stats['combined_words'])
            doc_combined_counts.append((doc_id, combined_count, stats))
        
        for doc_id, count, stats in sorted(doc_combined_counts, key=lambda x: x[1], reverse=True)[:20]:
            out.write(f'  {doc_id}: {count} combined words\n')
        out.write('\n')
        
        # Recommendations
        out.write('RECOMMENDATIONS\n')
        out.write('-' * 20 + '\n')
        
        if total_unicode_chars / total_chars > 0.05:
            out.write('⚠️  High Unicode content detected. Consider Unicode normalization.\n')
        else:
            out.write('✅ Unicode content is within acceptable range.\n')
        
        if len(all_combined_words) > 1000:
            out.write('⚠️  Many combined words detected. Implement word separation.\n')
            out.write('   Key terms to fix:\n')
            for word, count in sustainability_counts.most_common(10):
                out.write(f'     - "{word}" ({count} occurrences)\n')
        else:
            out.write('✅ Combined words are minimal.\n')
        
        if all_problematic_chars:
            out.write('⚠️  Problematic Unicode characters found. Consider cleaning.\n')
        else:
            out.write('✅ No problematic Unicode characters detected.\n')
        
        out.write('\nReport generated successfully!\n')
    
    print(f'Comprehensive report written to: {REPORT_TXT}')


if __name__ == '__main__':
    main()
