import os
import json
import fitz
import unicodedata
import re
from pathlib import Path

# Import central configuration
from scaffold_core.config import (
    DATA_DIR, VECTOR_OUTPUTS_DIR, ITERATION, CHUNK_SIZE, CHUNK_OVERLAP,
    CHUNKED_TEXT_EXTRACTS_JSON, ensure_directories
)

# Update paths to use central configuration
PDF_INPUT_DIR = str(DATA_DIR)
OUTPUT_DIR = str(VECTOR_OUTPUTS_DIR)

# Optional import for PDF processing (fallback only)
try:
    from PyPDF2 import PdfReader
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False
    print("PyPDF2 not available - will use existing chunked data only")

def extract_text_per_page(path):
    if not HAS_PYPDF2:
        raise ImportError("PyPDF2 not available for PDF processing")
    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    # Normalize each page to fix OCR artifacts at the source
    try:
        from scaffold_core.text_clean import normalize_extracted_text
        pages = [normalize_extracted_text(p) for p in pages]
    except Exception:
        pass
    return pages

def chunk_text_by_words(pages, chunk_size, overlap):
    chunks = []
    words_with_page = []

    for i, text in enumerate(pages):
        words = text.split()
        words_with_page.extend([(w, i + 1) for w in words])

    step = chunk_size - overlap
    for i in range(0, len(words_with_page), step):
        word_slice = words_with_page[i : i + chunk_size]
        if not word_slice:
            continue
        words_only = [w for w, _ in word_slice]
        page_nums = [p for _, p in word_slice]
        chunk_text = " ".join(words_only)

        match = re.search(r'(?m)^(?P<h>\d+(\.\d+)+\s+.+)$', chunk_text)
        if match:
            heading = match.group('h')
            chunk_text = f"[{heading}]\n{chunk_text}"

        chunks.append({
            "text": chunk_text,
            "start_page": min(page_nums),
            "end_page": max(page_nums),
        })

    return chunks

def build_metadata_stub(filename):
    return {
        "title":       None,
        "authors":     None,
        "institution": None,
        "publication": None,
        "doi":         None,
        "url":         None,
        "filename":    filename
    }

def convert_existing_chunks_to_vector_format():
    """Convert existing chunked_text_extracts.json to vector format"""
    try:
        # Load existing chunked data
        existing_path = os.path.join(OUTPUT_DIR, "chunked_text_extracts.json")
        if not os.path.exists(existing_path):
            print(f"No existing chunked data found at {existing_path}")
            return None
            
        with open(existing_path, 'r', encoding='utf-8') as f:
            existing_chunks = json.load(f)
        
        print(f"Converting {len(existing_chunks)} existing chunks to vector format...")
        
        vector_chunks = []
        for i, chunk in enumerate(existing_chunks):
            # Extract folder from the existing data
            folder = chunk.get('folder', 'Unknown')
            metadata = chunk.get('metadata', {})
            
            vector_chunk = {
                "document_id": chunk.get('document_id', f"doc_{i}"),
                "chunk_id": i,
                "text": chunk.get('text', ''),
                "source_path": chunk.get('source_path', ''),
                "start_page": chunk.get('start_page', 1),
                "end_page": chunk.get('end_page', 1),
                "folder": folder,
                "metadata": {
                    "title": metadata.get('title'),
                    "authors": metadata.get('authors'),
                    "institution": metadata.get('institution'),
                    "publication": metadata.get('publication'),
                    "doi": metadata.get('doi'),
                    "url": metadata.get('url'),
                    "filename": chunk.get('document_id', f"doc_{i}")
                }
            }
            vector_chunks.append(vector_chunk)
        
        return vector_chunks
        
    except Exception as e:
        print(f"Error converting existing chunks: {e}")
        return None

def main():
    print("Processing PDFs from scratch...")
    all_chunks = []
    chunk_id = 0
    processed_files = 0
    total_files = 0

    # Count total PDF files first
    for root, _, files in os.walk(PDF_INPUT_DIR):
        for fname in files:
            if fname.lower().endswith(".pdf"):
                total_files += 1

    print(f"Found {total_files} PDF files to process...")

    for root, _, files in os.walk(PDF_INPUT_DIR):
        for fname in files:
            if not fname.lower().endswith(".pdf"):
                continue
            
            full_path = os.path.join(root, fname)
            processed_files += 1
            
            try:
                print(f"Processing {processed_files}/{total_files}: {fname}")
                
                if not HAS_PYPDF2:
                    print("  ✗ PyPDF2 not available - skipping PDF processing")
                    continue
                    
                pages = extract_text_per_page(full_path)
                chunks = chunk_text_by_words(pages, CHUNK_SIZE, CHUNK_OVERLAP)
                
                # Get folder name from path structure
                rel_path = os.path.relpath(root, PDF_INPUT_DIR)
                folder = rel_path if rel_path != '.' else 'Root'
                
                print(f"  ✓ Extracted {len(chunks)} chunks from {len(pages)} pages")

                base_metadata = build_metadata_stub(fname)

                for chunk in chunks:
                    all_chunks.append({
                        "document_id": fname,
                        "chunk_id":     chunk_id,
                        "text":         chunk["text"],
                        "source_path":  full_path,
                        "start_page":   chunk["start_page"],
                        "end_page":     chunk["end_page"],
                        "folder":       folder,
                        "metadata":     base_metadata
                    })
                    chunk_id += 1
                    
            except Exception as e:
                print(f"  ✗ Error processing {fname}: {e}")
                continue

    if all_chunks:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_path = os.path.join(OUTPUT_DIR, f"processed_{ITERATION}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=2)

        print(f"\n✓ Successfully processed {processed_files} files")
        print(f"✓ Generated {len(all_chunks)} chunks → {out_path}")
    else:
        print(f"\n✗ No chunks generated. Check if PyPDF2 is installed or if PDFs are readable.")

if __name__ == "__main__":
    main()
