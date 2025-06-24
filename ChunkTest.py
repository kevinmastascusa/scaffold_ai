import os
import json
import fitz

PDF_INPUT_DIR  = r"E:\Scaffold AI E\Current PDFs"
OUTPUT_DIR     = r"E:\Scaffold AI E\Outputs"
CHUNK_SIZE     = 500  # in words
CHUNK_OVERLAP  = 50   # in words
OUTPUT_NAME    = "processed_test.json"

def extract_text_from_pdf(path):
    """Extracts full text from a PDF using PyMuPDF."""
    text = ""
    doc = fitz.open(path)
    for page in doc:
        text += page.get_text() + "\n"
    return text

def chunk_text(text, chunk_size, overlap):
    """Chunks text into overlapping word windows."""
    words = text.split()
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        yield " ".join(words[i : i + chunk_size])

def main():
    all_chunks = []
    chunk_id = 0

    for root, _, files in os.walk(PDF_INPUT_DIR):
        for fname in files:
            if not fname.lower().endswith(".pdf"):
                continue
            full_path = os.path.join(root, fname)
            try:
                raw_text = extract_text_from_pdf(full_path)
            except Exception as e:
                print(f"[!] Failed to extract {fname}: {e}")
                continue

            for chunk in chunk_text(raw_text, CHUNK_SIZE, CHUNK_OVERLAP):
                all_chunks.append({
                    "document_id": fname,
                    "chunk_id": chunk_id,
                    "text": chunk,
                    "source_path": full_path,
                    "metadata": {
                        "folder": root,
                        "filename": fname
                    }
                })
                chunk_id += 1

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_NAME)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"Processed {chunk_id} chunks → {output_path}")

if __name__ == "__main__":
    main()
