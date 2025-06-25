import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

import json
import re
import unicodedata
from pathlib import Path
import PyPDF2
import fitz  # PyMuPDF for better math extraction
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import sys
from huggingface_hub import hf_hub_download

# Add the root directory to the Python path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
sys.path.append(root_dir)

# Simple math-preserving clean_text function
def clean_text_preserve_math(text):
    """Clean text while preserving mathematical symbols and formulas."""
    import unicodedata
    
    # Define mathematical Unicode ranges to preserve
    math_ranges = [
        (0x2190, 0x21FF),  # Arrows
        (0x2200, 0x22FF),  # Mathematical Operators
        (0x2300, 0x23FF),  # Miscellaneous Technical
        (0x25A0, 0x25FF),  # Geometric Shapes
        (0x2600, 0x26FF),  # Miscellaneous Symbols
        (0x27C0, 0x27EF),  # Miscellaneous Mathematical Symbols-A
        (0x2980, 0x29FF),  # Miscellaneous Mathematical Symbols-B
        (0x2A00, 0x2AFF),  # Supplemental Mathematical Operators
        (0x1D400, 0x1D7FF), # Mathematical Alphanumeric Symbols
    ]
    
    def is_math_char(char):
        code_point = ord(char)
        return any(start <= code_point <= end for start, end in math_ranges)
    
    # Only do minimal normalization - do NOT use NFKC which converts math symbols
    # Just remove control characters but preserve math symbols
    cleaned = ''
    for char in text:
        category = unicodedata.category(char)
        # Keep all printable characters, math symbols, and whitespace
        if category[0] != 'C' or is_math_char(char):
            cleaned += char
        elif category in ['Zs', 'Zl', 'Zp']:  # Whitespace
            cleaned += char
    
    return cleaned

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class MathAwarePDFProcessor:
    def __init__(self):
        # Initialize the sentence transformer model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.math_patterns = {            'equations': [
                # Basic equations: x = 5, E = mc², F = ma
                r'\b[a-zA-Z][₀-₉]*\s*[=≈≠]\s*[^.,;!?\n]{1,50}',
                # Mathematical functions: sin(x), log(y), exp(z)
                r'(?:sin|cos|tan|log|ln|exp|sqrt|abs)\s*\([^)]{1,30}\)',
                # Variables with subscripts/superscripts
                r'\b[a-zA-Z][₀-₉⁰-⁹]*\s*[=≈≠]\s*\d+\.?\d*',
                # Greek letters in equations - improved pattern
                r'[αβγδεζηθικλμνξοπρστυφχψω]\s*[=≈≠]\s*[0-9]+\.?[0-9]*%?',
                r'[αβγδεζηθικλμνξοπρστυφχψω]\s*[=≈≠]\s*\.[0-9]+',
                # Statistical equations like α = 0.05
                r'α\s*=\s*[0-9]*\.?[0-9]+%?',
                r'β\s*=\s*[0-9]*\.?[0-9]+%?',
                # More general math variable patterns
                r'\b[a-zA-Z]+\s*[=≈≠]\s*[0-9]+\.?[0-9]*%?',
            ],
            'formulas': [
                # Integration and summation
                r'[∫∑∏]\s*[^.,;!?\n]{1,80}',
                # Square roots and powers
                r'√[^.,;!?\s]{1,30}',
                r'\b[a-zA-Z]+[²³⁴⁵⁶⁷⁸⁹⁰¹]',
                r'\b[a-zA-Z]+\^[0-9]+',
                r'\b[a-zA-Z]+_[0-9]+',
                # Fractions and ratios
                r'\d+\.?\d*/\d+\.?\d*',
                # Mathematical expressions with operators
                r'[^.,;!?\n]*[±×÷≤≥≠≈∞][^.,;!?\n]*',
            ],
            'units': [
                # Scientific units with numbers - more comprehensive
                r'\d+\.?\d*\s*(?:kg|g|mg|μg|ng|lb|oz|t|ton)(?=\s|$|[.,;!?])',  # Mass
                r'\d+\.?\d*\s*(?:m|cm|mm|μm|nm|km|ft|in|yd|mi)(?=\s|$|[.,;!?])', # Length
                r'\d+\.?\d*\s*(?:L|mL|μL|gal|qt|pt|fl\.?oz)(?=\s|$|[.,;!?])',   # Volume
                r'\d+\.?\d*\s*(?:J|kJ|MJ|cal|kcal|BTU|Wh|kWh)(?=\s|$|[.,;!?])', # Energy
                r'\d+\.?\d*\s*(?:W|kW|MW|hp)(?=\s|$|[.,;!?])',                  # Power
                r'\d+\.?\d*\s*(?:V|mV|kV|A|mA|Ω|kΩ|MΩ)(?=\s|$|[.,;!?])',       # Electrical
                r'\d+\.?\d*\s*(?:Hz|kHz|MHz|GHz|rpm)(?=\s|$|[.,;!?])',         # Frequency
                r'\d+\.?\d*\s*(?:Pa|kPa|MPa|psi|atm|bar|mmHg)(?=\s|$|[.,;!?])', # Pressure
                r'\d+\.?\d*\s*°[CF](?=\s|$|[.,;!?])',                          # Temperature
                r'\d+\.?\d*\s*(?:mol|mmol|μmol)(?=\s|$|[.,;!?])',              # Amount
            ],            'statistics': [
                # Statistical measures
                r'(?:mean|median|mode|average|std|variance|σ²?)\s*[=:≈]\s*\d+\.?\d*',
                # Correlation and regression
                r'(?:r|R²?|correlation)\s*[=≈]\s*[01]?\.\d+',
                # P-values and significance - improved patterns
                r'(?:p|P)\s*[<>=≤≥≈]\s*0\.\d+',
                r'p\s*<\s*0\.0[0-9]+',
                # Alpha significance levels - improved
                r'α\s*[=≈]\s*[0-9]*\.?[0-9]+%?',
                r'significance\s+level.*α\s*[=≈]\s*[0-9]*\.?[0-9]+%?',
                # Sample sizes
                r'(?:n|N|sample\s+size)\s*[=:]\s*\d+',
                # Confidence intervals and error bars
                r'±\s*\d+\.?\d*',
                r'\d+\.?\d*\s*±\s*\d+\.?\d*',
                # Statistical ranges
                r'\d+\.?\d*\s*[-–—]\s*\d+\.?\d*',
                # Percentages in statistics
                r'\d+\.?\d*\s*%\s*(?:confidence|error|uncertainty)',
                # Cronbach's alpha and other reliability measures
                r'(?:pre|post)\s+α\s*[=≈]\s*\.[0-9]+',
                r'α\s*[=≈]\s*\.[0-9]+\)',
            ]
        }
        # Extended math symbols set for better detection
        self.math_symbols = set("∫∑∏√αβγδεζηθικλμνξοπρστυφχψω±×÷≤≥≠≈∞∂∇∆ΔΩπΠΣΦΨΞΘΛΓμνρστφχψωΘ²³⁴⁵⁶⁷⁸⁹⁰¹₀₁₂₃₄₅₆₇₈₉")
        self.math_unicode_ranges = [(0x2200, 0x22FF)]  # Math symbols range

    def extract_metadata(self, filename):
        # Improved DOI regex and fallback for subject/authors
        doi_pattern = r'10\.\d{4,9}/[-._;()/:A-Z0-9]+(?=\s|$)'
        subject_fallback = "Unknown Subject"
        authors_fallback = "Unknown Authors"

        metadata = {
            "folder": os.path.basename(os.path.dirname(filename)),
            "filename": os.path.basename(filename),
            "title": None,
            "authors": authors_fallback,
            "doi": None,
            "subject": subject_fallback,
            "page_number": None
        }

        # Extract DOI
        with open(filename, "r", encoding="utf-8") as f:
            content = f.read()
            doi_match = re.search(doi_pattern, content)
            if doi_match:
                metadata["doi"] = doi_match.group(0)

        return metadata

    def is_math_symbol(self, char):
        """Check if a character is a mathematical symbol"""
        if char in self.math_symbols:
            return True
        
        code_point = ord(char)
        for start, end in self.math_unicode_ranges:
            if start <= code_point <= end:
                return True
        
        return False

    def detect_math_content(self, text):
        """Detect mathematical content in text"""
        math_content = {
            'has_math': False,
            'math_symbols': [],
            'equations': [],
            'formulas': [],
            'units': [],
            'statistics': [],
            'math_symbol_count': 0,
            'math_density': 0.0
        }
        
        # Find math symbols
        math_chars = [char for char in text if self.is_math_symbol(char)]
        math_content['math_symbols'] = list(set(math_chars))
        math_content['math_symbol_count'] = len(math_chars)
          # Calculate math density
        if len(text) > 0:
            math_content['math_density'] = len(math_chars) / len(text)
        
        # Find mathematical patterns with better error handling
        for category, patterns in self.math_patterns.items():
            matches = []
            for pattern in patterns:
                try:
                    found_matches = re.finditer(pattern, text, re.IGNORECASE | re.UNICODE)
                    for match in found_matches:
                        match_text = match.group().strip()
                        if match_text:  # Only add non-empty matches
                            matches.append({
                                'text': match_text,
                                'start': match.start(),
                                'end': match.end()
                            })
                except re.error:
                    continue  # Skip invalid regex patterns
            math_content[category] = matches
        
        # More precise math detection logic
        has_unicode_math_symbols = math_content['math_symbol_count'] > 0
        has_meaningful_equations = len(math_content['equations']) > 0
        has_statistical_content = len(math_content['statistics']) > 0
        has_units_with_numbers = len(math_content['units']) > 0
        has_formulas = len(math_content['formulas']) > 0
        
        # Set has_math based on more specific criteria with lower threshold
        math_content['has_math'] = (
            has_unicode_math_symbols or 
            has_meaningful_equations or 
            has_statistical_content or 
            has_units_with_numbers or 
            has_formulas or
            math_content['math_density'] > 0.0001  # Lower threshold for better detection
        )
        
        return math_content

    def analyze_unicode_content(self, text):
        """Analyze Unicode content in text"""
        unicode_info = {
            'total_chars': len(text),
            'unicode_chars': 0,
            'unicode_categories': {},
            'scripts': set(),
            'normalization_forms': {},
            'problematic_chars': []
        }
        
        for char in text:
            # Count Unicode characters (non-ASCII)
            if ord(char) > 127:
                unicode_info['unicode_chars'] += 1
                
                # Categorize Unicode characters
                category = unicodedata.category(char)
                unicode_info['unicode_categories'][category] = unicode_info['unicode_categories'].get(category, 0) + 1
                
                # Detect scripts
                try:
                    script = unicodedata.name(char).split()[0]
                    unicode_info['scripts'].add(script)
                except ValueError:
                    pass
                
                # Check for problematic characters
                if unicodedata.category(char) in ['Cc', 'Cf', 'Co', 'Cs']:
                    unicode_info['problematic_chars'].append({
                        'char': char,
                        'code_point': ord(char),
                        'category': category,
                        'name': unicodedata.name(char, 'UNKNOWN')
                    })
        
        # Convert sets to lists for JSON serialization
        unicode_info['scripts'] = list(unicode_info['scripts'])        # Test different normalization forms
        norm_forms = ['NFC', 'NFD', 'NFKC', 'NFKD']
        for form_name in norm_forms:
            try:
                normalized = unicodedata.normalize(form_name, text)  # type: ignore
                unicode_info['normalization_forms'][form_name] = {
                    'length': len(normalized),
                    'differs_from_original': normalized != text
                }
            except Exception:
                unicode_info['normalization_forms'][form_name] = {
                    'length': 0,
                    'differs_from_original': False
                }
        
        return unicode_info

    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF using PyMuPDF for better math preservation"""
        try:
            # Use PyMuPDF (fitz) instead of PyPDF2 for better math extraction
            pdf_document = fitz.open(pdf_path)
            
            extraction_info = {
                'filename': os.path.basename(pdf_path),
                'num_pages': len(pdf_document),
                'full_text': '',
                'pages': [],
                'unicode_analysis': {},
                'math_analysis': {},
                'extraction_metadata': {
                    'success': True,
                    'error': None,
                    'total_chars': 0,
                    'total_unicode_chars': 0,
                    'total_math_symbols': 0,
                    'extraction_method': 'PyMuPDF'                }
            }
            
            full_text = ""
            total_unicode_chars = 0
            total_math_symbols = 0
            
            for page_num in range(len(pdf_document)):
                try:
                    page = pdf_document[page_num]
                    # Extract text with better math symbol preservation
                    page_text = ""
                    try:
                        page_text = page.get_text()
                        if not isinstance(page_text, str):
                            page_text = ""
                    except Exception as e:
                        print(f"Error extracting text from page {page_num + 1}: {e}")
                        page_text = ""
                    # Ensure page_text is a string
                    if page_text is None:
                        page_text = ""
                    full_text += page_text + "\n"
                    
                    # Analyze this page
                    page_unicode = self.analyze_unicode_content(page_text)
                    page_math = self.detect_math_content(page_text)
                    
                    total_unicode_chars += page_unicode.get('unicode_chars', 0)
                    total_math_symbols += page_math.get('math_symbol_count', 0)
                    
                    extraction_info['pages'].append({
                        'page_number': page_num + 1,
                        'text': page_text,
                        'char_count': len(page_text),
                        'unicode_analysis': page_unicode,
                        'math_analysis': page_math
                    })
                    
                except Exception as e:
                    print(f"Error extracting page {page_num + 1} from {pdf_path}: {e}")
                    extraction_info['pages'].append({
                        'page_number': page_num + 1,
                        'error': str(e)
                    })
            
            # Close the PDF document
            pdf_document.close()
            
            extraction_info['full_text'] = full_text
            extraction_info['unicode_analysis'] = self.analyze_unicode_content(full_text)
            extraction_info['math_analysis'] = self.detect_math_content(full_text)
            
            # Update metadata
            extraction_info['extraction_metadata']['total_chars'] = len(full_text)
            extraction_info['extraction_metadata']['total_unicode_chars'] = total_unicode_chars
            extraction_info['extraction_metadata']['total_math_symbols'] = total_math_symbols
            
            return extraction_info
                
        except Exception as e:
            return {
                'filename': os.path.basename(pdf_path),
                'extraction_metadata': {
                    'success': False,
                    'error': str(e)
                }
            }

    def smart_chunk_text(self, text, max_chunk_size=500, overlap=50):
        """Create chunks with math-awareness and Unicode preservation"""
        if not text.strip():
            return []
        # Clean text using the math-preserving Unicode cleaner
        cleaned_text = clean_text_preserve_math(text)

        # Minimal normalization - preserve mathematical content
        normalized_text = cleaned_text  # No aggressive normalization

        # Use NLTK sentence tokenization for better boundaries
        try:
            sentences = sent_tokenize(normalized_text)
        except:
            # Fallback to simple sentence splitting
            sentences = re.split(r'[.!?]+\s+', normalized_text)

        chunks = []
        current_chunk = ""
        current_size = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence_size = len(sentence)

            # If adding this sentence would exceed max size, finalize current chunk
            if current_size + sentence_size > max_chunk_size and current_chunk:
                # Analyze the chunk before adding it
                chunk_math = self.detect_math_content(current_chunk)
                chunk_unicode = self.analyze_unicode_content(current_chunk)

                chunks.append({
                    'text': current_chunk.strip(),
                    'size': len(current_chunk),
                    'unicode_analysis': chunk_unicode,
                    'math_analysis': chunk_math,
                    'chunk_metadata': {
                        'has_math': chunk_math['has_math'],
                        'math_density': chunk_math['math_density'],
                        'unicode_ratio': chunk_unicode['unicode_chars'] / chunk_unicode['total_chars'] if chunk_unicode['total_chars'] > 0 else 0
                    }
                })

                # Start new chunk with overlap
                if overlap > 0 and len(current_chunk) > overlap:
                    overlap_text = current_chunk[-overlap:]
                    current_chunk = overlap_text + " " + sentence
                    current_size = len(current_chunk)
                else:
                    current_chunk = sentence
                    current_size = sentence_size
            else:
                # Add sentence to current chunk
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
                current_size = len(current_chunk)

        # Add the last chunk if it exists
        if current_chunk.strip():
            chunk_math = self.detect_math_content(current_chunk)
            chunk_unicode = self.analyze_unicode_content(current_chunk)

            chunks.append({
                'text': current_chunk.strip(),
                'size': len(current_chunk),
                'unicode_analysis': chunk_unicode,
                'math_analysis': chunk_math,
                'chunk_metadata': {
                    'has_math': chunk_math['has_math'],
                    'math_density': chunk_math['math_density'],
                    'unicode_ratio': chunk_unicode['unicode_chars'] / chunk_unicode['total_chars'] if chunk_unicode['total_chars'] > 0 else 0
                }
            })

        return chunks

    def chunk_by_pages(self, pdf_path):
        """Create chunks based on PDF pages instead of text size"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                chunks = []
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        
                        if not page_text.strip():
                            continue
                          # Clean the text preserving math symbols
                        cleaned_text = clean_text_preserve_math(page_text)
                        normalized_text = cleaned_text  # No aggressive normalization
                        
                        # Analyze the page for math and unicode content
                        page_math = self.detect_math_content(normalized_text)
                        page_unicode = self.analyze_unicode_content(normalized_text)
                        
                        # Create chunk for this page
                        chunks.append({
                            'text': normalized_text.strip(),
                            'size': len(normalized_text),
                            'page_number': page_num + 1,
                            'unicode_analysis': page_unicode,
                            'math_analysis': page_math,
                            'chunk_metadata': {
                                'has_math': page_math['has_math'],
                                'math_density': page_math['math_density'],
                                'unicode_ratio': page_unicode['unicode_chars'] / page_unicode['total_chars'] if page_unicode['total_chars'] > 0 else 0,
                                'chunk_type': 'page_based'
                            },
                            # Include all math detection results in the chunk
                            'has_math': page_math['has_math'],
                            'math_symbols': page_math['math_symbols'],
                            'equations': page_math['equations'],
                            'formulas': page_math['formulas'],
                            'units': page_math['units'],
                            'statistics': page_math['statistics'],
                            'math_symbol_count': page_math['math_symbol_count'],
                            'math_density': page_math['math_density']
                        })
                        
                    except Exception as e:
                        print(f"Error processing page {page_num + 1} from {pdf_path}: {e}")
                        continue
                
                return chunks
                
        except Exception as e:
            print(f"Error reading PDF {pdf_path}: {e}")
            return []

    def process_all_pdfs(self, data_folder):
        """Process all PDFs in the data folder with math and Unicode awareness"""
        results = {
            'processing_metadata': {
                'total_files': 0,
                'successful_extractions': 0,
                'failed_extractions': 0,
                'total_math_documents': 0,
                'total_unicode_documents': 0,
                'processing_timestamp': None
            },
            'documents': {}
        }
        
        pdf_files = []
        for root, dirs, files in os.walk(data_folder):
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdf_files.append(os.path.join(root, file))
        
        results['processing_metadata']['total_files'] = len(pdf_files)
        
        print(f"Found {len(pdf_files)} PDF files to process with math-aware extraction...")
        
        for pdf_path in pdf_files:
            print(f"Processing: {os.path.basename(pdf_path)}")
            
            # Extract text with analysis
            extraction_result = self.extract_text_from_pdf(pdf_path)
            
            if extraction_result['extraction_metadata']['success']:
                results['processing_metadata']['successful_extractions'] += 1
                
                # Check if document has significant math or Unicode content
                if extraction_result.get('math_analysis', {}).get('has_math', False):
                    results['processing_metadata']['total_math_documents'] += 1
                
                if extraction_result.get('unicode_analysis', {}).get('unicode_chars', 0) > 10:
                    results['processing_metadata']['total_unicode_documents'] += 1
                  # Create chunks by pages instead of by text size
                chunks = self.chunk_by_pages(pdf_path)
                extraction_result['chunks'] = chunks
                extraction_result['chunk_count'] = len(chunks)
                
                # Calculate document-level statistics
                extraction_result['document_statistics'] = {
                    'total_chunks': len(chunks),
                    'chunks_with_math': sum(1 for chunk in chunks if chunk['chunk_metadata']['has_math']),
                    'chunks_with_unicode': sum(1 for chunk in chunks if chunk['chunk_metadata']['unicode_ratio'] > 0.01),
                    'avg_math_density': np.mean([chunk['chunk_metadata']['math_density'] for chunk in chunks]) if chunks else 0,
                    'avg_unicode_ratio': np.mean([chunk['chunk_metadata']['unicode_ratio'] for chunk in chunks]) if chunks else 0
                }
                
            else:
                results['processing_metadata']['failed_extractions'] += 1
            
            # Store result
            relative_path = os.path.relpath(pdf_path, data_folder)
            results['documents'][relative_path] = extraction_result
        
        # Add timestamp
        from datetime import datetime
        results['processing_metadata']['processing_timestamp'] = datetime.now().isoformat()
        
        return results

    def process_chunk(self, chunk, filename, start_page, end_page):
        metadata = self.extract_metadata(filename)
        return {
            "document_id": metadata["filename"],
            "chunk_id": f"{metadata['filename']}_{start_page}_{end_page}",
            "text": chunk,
            "start_page": start_page,
            "end_page": end_page,
            "source_path": filename,
            "metadata": metadata
        }

def main():
    # Create output directories
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('math_outputs', exist_ok=True)
    
    # Initialize processor
    processor = MathAwarePDFProcessor()
    
    # Process all PDFs in the data folder
    data_folder = 'data'
    
    print("Starting math-aware PDF processing...")
    results = processor.process_all_pdfs(data_folder)
    
    # Save full extraction results with math and Unicode analysis
    full_output_path = 'math_outputs/math_aware_full_extracts.json'
    with open(full_output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Math-aware full extraction results saved to: {full_output_path}")
    
    # Create chunked-only output for comparison
    chunked_results = {
        'processing_metadata': results['processing_metadata'].copy(),
        'documents': {}
    }
    
    for doc_path, doc_data in results['documents'].items():
        if doc_data['extraction_metadata']['success']:
            chunked_results['documents'][doc_path] = {
                'filename': doc_data['filename'],
                'chunk_count': doc_data.get('chunk_count', 0),
                'chunks': doc_data.get('chunks', []),
                'document_statistics': doc_data.get('document_statistics', {}),
                'math_analysis_summary': {
                    'has_math': doc_data.get('math_analysis', {}).get('has_math', False),
                    'math_symbol_count': doc_data.get('math_analysis', {}).get('math_symbol_count', 0),
                    'math_density': doc_data.get('math_analysis', {}).get('math_density', 0)
                },
                'unicode_analysis_summary': {
                    'unicode_chars': doc_data.get('unicode_analysis', {}).get('unicode_chars', 0),
                    'total_chars': doc_data.get('unicode_analysis', {}).get('total_chars', 0),
                    'unicode_ratio': doc_data.get('unicode_analysis', {}).get('unicode_chars', 0) / max(doc_data.get('unicode_analysis', {}).get('total_chars', 1), 1)
                }
            }
        else:
            chunked_results['documents'][doc_path] = {
                'filename': doc_data['filename'],
                'extraction_metadata': doc_data['extraction_metadata']
            }
    
    # Save chunked results
    chunked_output_path = 'math_outputs/math_aware_chunked_extracts.json'
    with open(chunked_output_path, 'w', encoding='utf-8') as f:
        json.dump(chunked_results, f, indent=2, ensure_ascii=False)
    
    print(f"Math-aware chunked results saved to: {chunked_output_path}")
    
    # Create summary report
    summary = {
        'extraction_summary': results['processing_metadata'],
        'math_content_summary': {
            'documents_with_math': results['processing_metadata']['total_math_documents'],
            'documents_with_unicode': results['processing_metadata']['total_unicode_documents'],
            'math_percentage': (results['processing_metadata']['total_math_documents'] / 
                              max(results['processing_metadata']['successful_extractions'], 1)) * 100,
            'unicode_percentage': (results['processing_metadata']['total_unicode_documents'] / 
                                 max(results['processing_metadata']['successful_extractions'], 1)) * 100
        },
        'top_math_documents': [],
        'top_unicode_documents': []
    }
    
    # Find documents with highest math content
    math_docs = []
    unicode_docs = []
    
    for doc_path, doc_data in results['documents'].items():
        if doc_data['extraction_metadata']['success']:
            math_density = doc_data.get('math_analysis', {}).get('math_density', 0)
            unicode_ratio = doc_data.get('unicode_analysis', {}).get('unicode_chars', 0) / max(doc_data.get('unicode_analysis', {}).get('total_chars', 1), 1)
            
            if math_density > 0:
                math_docs.append((doc_path, math_density, doc_data.get('math_analysis', {}).get('math_symbol_count', 0)))
            
            if unicode_ratio > 0:
                unicode_docs.append((doc_path, unicode_ratio, doc_data.get('unicode_analysis', {}).get('unicode_chars', 0)))
    
    # Sort and get top documents
    math_docs.sort(key=lambda x: x[1], reverse=True)
    unicode_docs.sort(key=lambda x: x[1], reverse=True)
    
    summary['top_math_documents'] = [
        {'document': doc[0], 'math_density': doc[1], 'math_symbol_count': doc[2]}
        for doc in math_docs[:10]
    ]
    
    summary['top_unicode_documents'] = [
        {'document': doc[0], 'unicode_ratio': doc[1], 'unicode_char_count': doc[2]}
        for doc in unicode_docs[:10]
    ]
    
    # Save summary
    summary_output_path = 'math_outputs/math_unicode_summary.json'
    with open(summary_output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"Summary report saved to: {summary_output_path}")
    
    # Print processing summary
    print("\n" + "="*50)
    print("MATH-AWARE PROCESSING SUMMARY")
    print("="*50)
    print(f"Total files processed: {results['processing_metadata']['total_files']}")
    print(f"Successful extractions: {results['processing_metadata']['successful_extractions']}")
    print(f"Failed extractions: {results['processing_metadata']['failed_extractions']}")
    print(f"Documents with math content: {results['processing_metadata']['total_math_documents']}")
    print(f"Documents with Unicode content: {results['processing_metadata']['total_unicode_documents']}")
    print(f"Math content percentage: {summary['math_content_summary']['math_percentage']:.1f}%")
    print(f"Unicode content percentage: {summary['math_content_summary']['unicode_percentage']:.1f}%")
    print("\nOutputs created:")
    print(f"  - {full_output_path}")
    print(f"  - {chunked_output_path}")
    print(f"  - {summary_output_path}")
    print("\nUse these files to compare with standard extraction results!")

if __name__ == "__main__":
    main()
