"""
Central configuration file for Scaffold AI project.
All paths are defined relative to the workspace root for portability.
"""

import os
from pathlib import Path
import torch
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Get the workspace root directory (where this config file is located, going up to scaffold_ai/)
WORKSPACE_ROOT = Path(__file__).parent.parent.absolute()

# Core directories
DATA_DIR = WORKSPACE_ROOT / "data"
OUTPUTS_DIR = WORKSPACE_ROOT / "outputs"
VECTOR_OUTPUTS_DIR = WORKSPACE_ROOT / "vector_outputs"
MATH_OUTPUTS_DIR = WORKSPACE_ROOT / "math_outputs"

# Input/Output file paths
CHUNKED_TEXT_EXTRACTS_JSON = OUTPUTS_DIR / "chunked_text_extracts.json"
FULL_TEXT_EXTRACTS_JSON = OUTPUTS_DIR / "full_text_extracts.json"
UNICODE_REPORT_TXT = OUTPUTS_DIR / "unicode_report.txt"

# Vector processing files
PROCESSED_JSON_PATH = CHUNKED_TEXT_EXTRACTS_JSON  # Point to precomputed chunks
VECTOR_PROCESSED_JSON = VECTOR_OUTPUTS_DIR / "processed_1.json"

# Math processing files
MATH_AWARE_FULL_EXTRACTS_JSON = MATH_OUTPUTS_DIR / "math_aware_full_extracts.json"
MATH_AWARE_CHUNKED_EXTRACTS_JSON = MATH_OUTPUTS_DIR / "math_aware_chunked_extracts.json"

# Vector index and metadata files (will be created dynamically)
def get_faiss_index_path(iteration: int = 1) -> Path:
    """Get the FAISS index path for a given iteration."""
    return VECTOR_OUTPUTS_DIR / f"scaffold_index_{iteration}.faiss"

def get_metadata_json_path(iteration: int = 1) -> Path:
    """Get the metadata JSON path for a given iteration."""
    return VECTOR_OUTPUTS_DIR / f"scaffold_metadata_{iteration}.json"

# Processing parameters
CHUNK_SIZE = 1000  # in words (unused - now using complete page chunking)
CHUNK_OVERLAP = 200  # in words (unused - now using complete page chunking)
ITERATION = 1  # bump this when you want a fresh run

# Model configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# LLM configuration
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"  # Official Mistral v0.2 model
LLM_TASK = "text-generation"  # Task type for the pipeline
LLM_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available
LLM_MAX_LENGTH = 2048  # Increased for more detailed responses
LLM_TEMPERATURE = 0.3  # Lower temperature for more focused responses
LLM_TOP_P = 0.9  # Slightly lower for faster generation
LLM_BATCH_SIZE = 1  # Batch size for processing
LLM_LOAD_IN_8BIT = False  # Use 8-bit quantization for faster loading (disabled for Windows)
LLM_LOAD_IN_4BIT = False  # 4-bit for even faster loading (disabled for Windows)

# Hugging Face token
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
if not HF_TOKEN:
    logger.warning("HUGGINGFACE_TOKEN environment variable not set. Some models may not be accessible.")

# Cross-encoder model for reranking
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# FAISS index configuration
FAISS_INDEX_TYPE = "IndexFlatL2"

# Search configuration
TOP_K_INITIAL = 25  # Reduced for faster processing
TOP_K_FINAL = 5     # Reduced for faster processing

# Ensure directories exist
def ensure_directories():
    """Create all necessary directories if they don't exist."""
    directories = [
        DATA_DIR,
        OUTPUTS_DIR,
        VECTOR_OUTPUTS_DIR,
        MATH_OUTPUTS_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"✓ Ensured directory exists: {directory}")

# Legacy path compatibility (for backward compatibility)
# These will be removed once all files are updated to use the new config
PDF_INPUT_DIR = str(DATA_DIR)
OUTPUT_DIR = str(OUTPUTS_DIR)
ROOT_DIR = str(WORKSPACE_ROOT)