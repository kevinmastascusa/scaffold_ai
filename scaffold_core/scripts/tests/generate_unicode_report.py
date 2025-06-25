#!/usr/bin/env python3
"""
Generate a unicode analysis report for each chunk in chunked_text_extracts.json.
Writes results to outputs/unicode_report.txt
"""
import sys
import os

# Add workspace root to Python path for module imports
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, root_dir)

import json
from scaffold_core.scripts.chunk.math.ChunkTest_Math import MathAwarePDFProcessor

# Paths
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
CHUNKS_JSON = os.path.join(ROOT_DIR, 'outputs', 'chunked_text_extracts.json')
REPORT_TXT = os.path.join(ROOT_DIR, 'outputs', 'unicode_report.txt')


def main():
    # Load chunks
    with open(CHUNKS_JSON, 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    processor = MathAwarePDFProcessor()

    with open(REPORT_TXT, 'w', encoding='utf-8') as out:
        out.write('Unicode Analysis Report\n')
        out.write('=' * 40 + '\n\n')

        for i, chunk in enumerate(chunks, 1):
            doc_id = chunk.get('document_id')
            chunk_id = chunk.get('chunk_id')
            page = chunk.get('start_page')
            text = chunk.get('text', '')

            uni_info = processor.analyze_unicode_content(text)

            out.write(f'[{i}] Document: {doc_id}, Chunk: {chunk_id}, Page: {page}\n')
            out.write(f"  Total chars: {uni_info['total_chars']}, Unicode chars: {uni_info['unicode_chars']}\n")
            out.write('  Categories:\n')
            for cat, count in uni_info.get('unicode_categories', {}).items():
                out.write(f'    {cat}: {count}\n')
            out.write('  Scripts: ' + ', '.join(sorted(uni_info.get('scripts', []))) + '\n')
            out.write('-' * 40 + '\n')

    print(f'Unicode report written to: {REPORT_TXT}')


if __name__ == '__main__':
    main()
