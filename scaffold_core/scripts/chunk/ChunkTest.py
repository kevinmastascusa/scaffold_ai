"""
Module: ChunkTest.py
Page-based PDF chunking (one complete page per chunk).

Preferred chunking strategies (in order):
1. Page-based chunking (this module) - recommended. Outputs: outputs/chunked_text_extracts.json
2. Math-aware chunking (ChunkTest_Math.py) - in progress. Outputs: math_outputs/math_aware_chunked_extracts.json
3. Word-based chunking (vector/chunk.py) - legacy fallback.

Vector pipeline is now configured to use this page-based JSON output.
"""

import os
import json
import fitz
import unicodedata
import re


# Update paths to work with current workspace
PDF_INPUT_DIR  = r"c:\Users\dlaev\OneDrive\Documents\GitHub\scaffold_ai\data"
OUTPUT_DIR     = r"c:\Users\dlaev\OneDrive\Documents\GitHub\scaffold_ai\outputs"
CHUNK_SIZE     = 500  # in words (unused - now using complete page chunking)
CHUNK_OVERLAP  = 50   # in words (unused - now using complete page chunking)
OUTPUT_NAME    = "processed_test.json"

# NOTE: This script now uses page-based chunking (one chunk per complete page)
# instead of overlapping word-based chunks



def extract_text_from_pdf(path):
    """
    Extracts the entire text of a PDF via PyMuPDF.
    Recommended for full-document extraction; for page-based chunks use `extract_text_by_page`.
    """
    text = ""
    doc = fitz.open(path)
    for page in doc:
        text += page.get_text() + "\n"
    return text

def chunk_text(text, chunk_size, overlap):
    """
    Legacy overlapping word-based chunking: splits text into windows of `chunk_size` words with `overlap`.
    Not used when page-based chunking is preferred.
    """
    words = text.split()
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        yield " ".join(words[i : i + chunk_size])

def extract_text_by_page(path):
    """
    Extracts and cleans text from each page of a PDF for page-based chunking.
    Returns a list of dicts with 'page_number' and 'text'.
    """
    doc = fitz.open(path)
    page_chunks = []

    for i, page in enumerate(doc):
        text = page.get_text()
        text = clean_text(text)
        if text.strip():  # Skip empty pages
            page_chunks.append({
                "page_number": i + 1,
                "text": text
            })

    return page_chunks

def parse_metadata_from_text(text):
    """
    Parses DOI and possible authors from the first page text metadata block.
    Returns a dict with 'doi' and 'authors_extracted'.
    """
    metadata = {}

    # Extract DOI with stricter regex and lookahead for whitespace/end
    doi_match = re.search(r'\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+(?=\s|$)', text)
    if doi_match:
        metadata["doi"] = doi_match.group(0)

    # Try extracting possible author block from first lines
    lines = text.split('\n')
    possible_authors = []
    for line in lines[:20]:
        if re.search(r'\bby\b', line, re.IGNORECASE) or re.search(r'\b[A-Z][a-z]+ [A-Z][a-z]+', line):
            if len(line) < 150 and '@' not in line and 'doi' not in line.lower():
                possible_authors.append(line.strip())

    if possible_authors:
        metadata["authors_extracted"] = " ".join(possible_authors)
    else:
        # Fallback: Named entity recognition for PERSON
        try:
            from nltk import word_tokenize, pos_tag, ne_chunk
            tokens = word_tokenize(text[:500])
            tags = pos_tag(tokens)
            chunks = ne_chunk(tags)
            persons = [" ".join([leaf[0] for leaf in tree.leaves()])
                       for tree in chunks if hasattr(tree, 'label') and tree.label() == 'PERSON']
            if persons:
                metadata["authors_extracted"] = ', '.join(persons[:2])
        except Exception:
            pass

    # Extract title: first reasonable line
    for line in lines:
        line = line.strip()
        if 20 < len(line) < 120 and 'doi' not in line.lower():
            metadata["title"] = line
            break

    return metadata

def validate_metadata(meta_dict):
    """
    Validates and normalizes raw metadata dict.
    Ensures title, author, subject fields are non-empty strings or None.
    """
    def safe(val):
        return val.strip() if isinstance(val, str) and len(val.strip()) > 2 else None

    return {
        "title": safe(meta_dict.get("title")),
        "author": safe(meta_dict.get("author")),
        "subject": safe(meta_dict.get("subject"))
    }


def clean_text(text):
    """
    Cleans and normalizes text:
    - Applies NFKC normalization
    - Removes control, symbol, and non-ASCII characters
    - Collapses whitespace
    """
    # Normalize to NFC form (standardizes combined characters)
    text = unicodedata.normalize("NFKC", text)

    # Remove control characters, private-use, surrogates, non-ASCII (except for basic Latin + Latin-1 Supplement)
    text = ''.join(
        c for c in text
        if unicodedata.category(c)[0] not in ('C', 'S')  # Control & Symbol characters
        and ord(c) >= 32
    )

    # Optional: remove extra whitespace and junk
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def main():
    """
    Main function with options for full text or chunked extraction.
    """
    print("PDF Text Extraction Tool")
    print("=" * 50)
    print(f"Input directory: {PDF_INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    
    # Extract full text from all PDFs (recommended for full text needs)
    print("1. Extracting FULL TEXT from all PDFs...")
    full_text_results = extract_full_text_recursive(
        input_dir=PDF_INPUT_DIR, 
        output_dir=OUTPUT_DIR, 
        extract_full_text=True
    )
    
    print("\n" + "=" * 50)
    
    # Also create chunked version (useful for embeddings/search)
    print("2. Creating CHUNKED version for search/embeddings...")
    chunked_results = extract_full_text_recursive(
        input_dir=PDF_INPUT_DIR, 
        output_dir=OUTPUT_DIR, 
        extract_full_text=False
    )
    
    print("\n" + "=" * 50)
    print("SUMMARY:")
    print(f"  Full text extracts: {len(full_text_results)} documents")
    print(f"  Chunked extracts: {len(chunked_results)} chunks")
    print(f"  Output directory: {OUTPUT_DIR}")
    print("\nFiles created:")
    print(f"  - full_text_extracts.json (complete document text)")
    print(f"  - chunked_text_extracts.json (text chunks with overlap)")
    print("\nProcessing complete!")



def extract_full_text_recursive(input_dir, output_dir, extract_full_text=True):
    """
    Recursively process PDFs in `input_dir`.

    If `extract_full_text` is True, extracts full document text per PDF.
    If False, uses page-based chunking to output one chunk per page.

    Saves JSON to `output_dir` and returns list of extracts.
    """
    all_extracts = []
    processed_count = 0
    
    print(f"Starting recursive PDF processing from: {input_dir}")
    
    for root, dirs, files in os.walk(input_dir):
        relative_path = os.path.relpath(root, input_dir)
        folder_name = relative_path.replace(os.sep, "_") if relative_path != "." else "root"
        
        print(f"\nProcessing folder: {relative_path}")
        
        pdf_files = [f for f in files if f.lower().endswith('.pdf')]
        if not pdf_files:
            print(f"  No PDF files found in {relative_path}")
            continue
            
        print(f"  Found {len(pdf_files)} PDF files")
        
        for fname in pdf_files:
            full_path = os.path.join(root, fname)
            print(f"    Processing: {fname}")
            
            try:
                doc = fitz.open(full_path)
                raw_meta = validate_metadata(doc.metadata)
                
                if extract_full_text:
                    # Extract full text from entire document
                    full_text = extract_text_from_pdf(full_path)
                    full_text = clean_text(full_text)
                    
                    # Parse metadata from first page
                    first_page_text = clean_text(doc[0].get_text())
                    text_meta = parse_metadata_from_text(first_page_text)
                    
                    extract_info = {
                        "document_id": fname,
                        "full_text": full_text,
                        "word_count": len(full_text.split()),
                        "source_path": full_path,
                        "folder": relative_path,
                        "metadata": {
                            "folder": relative_path,
                            "filename": fname,
                            "title": raw_meta["title"] or os.path.splitext(fname)[0],
                            "authors": raw_meta["author"] or text_meta.get("authors_extracted"),
                            "doi": text_meta.get("doi"),
                            "subject": raw_meta["subject"],
                            "total_pages": len(doc)                        }
                    }
                    all_extracts.append(extract_info)
                    
                else:
                    # Use page-based chunking logic - one chunk per complete page
                    first_page_text = clean_text(doc[0].get_text())
                    text_meta = parse_metadata_from_text(first_page_text)
                    pages = extract_text_by_page(full_path)
                    
                    for page in pages:
                        # Create one chunk per complete page instead of multiple overlapping chunks
                        all_extracts.append({
                            "document_id": fname,
                            "chunk_id": f"{fname}_page_{page['page_number']}",
                            "text": page["text"],
                            "start_page": page["page_number"],
                            "end_page": page["page_number"],
                            "source_path": full_path,
                            "metadata": {
                                "folder": relative_path,
                                "filename": fname,
                                "title": raw_meta["title"] or os.path.splitext(fname)[0],
                                "authors": raw_meta["author"] or text_meta.get("authors_extracted"),
                                "doi": text_meta.get("doi"),
                                "subject": raw_meta["subject"],
                                "page_number": page["page_number"],
                                "chunk_type": "complete_page"
                            }
                            })
                
                processed_count += 1
                doc.close()
                
            except Exception as e:
                print(f"    [ERROR] Failed to process {fname}: {e}")
                continue
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    if extract_full_text:
        output_file = "full_text_extracts.json"
        print(f"\nSaving full text extracts for {processed_count} documents...")
    else:
        output_file = "chunked_text_extracts.json"
        print(f"\nSaving chunked text extracts for {processed_count} documents...")
    
    output_path = os.path.join(output_dir, output_file)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_extracts, f, ensure_ascii=False, indent=2)
    
    print(f"Results saved to: {output_path}")
    print(f"Total documents processed: {processed_count}")
    print(f"Total extracts/chunks: {len(all_extracts)}")
    
    return all_extracts

if __name__ == "__main__":
    main()
