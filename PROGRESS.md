# Project Progress Log

**Date:** June 24, 2025

## Summary of Recent Changes

- Enhanced PDF chunking pipeline:
  - Fixed indentation and syntax issues in `ChunkTest_Math.py`.
  - Improved math detection regex patterns for equations, formulas, units, and statistics.
  - Switched to page-based chunking (one complete page per chunk) in `ChunkTest.py`.

- Vectorization pipeline updates:
  - Pointed `main.py` and `transformVector.py` to use precomputed `chunked_text_extracts.json`.
  - Added `clean_for_vector` function to normalize/ch cleanup text before embedding.
  - Corrected FAISS `index.add` signature to include the vector count.
  - Renamed `transormVector.py` to `transformVector.py` and updated references in test scripts.

- Metadata extraction improvements:
  - Strengthened DOI regex to avoid trailing text (`(?=\s|$)` lookahead).
  - Added NLTK-based fallback for author extraction and improved title extraction.

- Testing and utilities:
  - Created `generate_unicode_report.py` to output `unicode_report.txt` for chunked extracts.
  - Updated import paths and `sys.path` insertions for test scripts.
  - Cleaned up `test_vector_setup.py` to inline chunk conversion logic and remove deprecated imports.

---

## Next Steps / TODOs

- Fix combined words in `chunked_text_extracts.json` (e.g., "environmentalsustainability").
- Integrate full Unicode analysis into page-based chunks for `ChunkTest.py` outputs.
- Complete math-aware chunking improvements in `ChunkTest_Math.py` and re-integrate with vector pipeline.
- Add vector query examples in the README.
