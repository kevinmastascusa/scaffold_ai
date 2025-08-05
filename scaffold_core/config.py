"""
Central configuration file for Scaffold AI project.
All paths are defined relative to the workspace root for portability.

Compatible with Python 3.12.10+
Updated: January 2025
"""

import logging
import os
from pathlib import Path

import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the workspace root directory
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
PROCESSED_JSON_PATH = CHUNKED_TEXT_EXTRACTS_JSON
VECTOR_PROCESSED_JSON = VECTOR_OUTPUTS_DIR / "processed_1.json"

# Math processing files
MATH_AWARE_FULL_EXTRACTS_JSON = (
    MATH_OUTPUTS_DIR / "math_aware_full_extracts.json"
)
MATH_AWARE_CHUNKED_EXTRACTS_JSON = (
    MATH_OUTPUTS_DIR / "math_aware_chunked_extracts.json"
)


def get_faiss_index_path(iteration: int = 1) -> Path:
    """Get the FAISS index path for a given iteration."""
    return VECTOR_OUTPUTS_DIR / f"scaffold_index_{iteration}.faiss"


def get_metadata_json_path(iteration: int = 1) -> Path:
    """Get the metadata JSON path for a given iteration."""
    return VECTOR_OUTPUTS_DIR / f"scaffold_metadata_{iteration}.json"


# Processing parameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
ITERATION = 1

# -------------------
# MODEL REGISTRIES & SELECTION
# -------------------

# Embedding Models Registry
EMBEDDING_MODELS = {
    "miniLM": {
        "name": "all-MiniLM-L6-v2",
        "desc": "Recommended: Fast, high-quality, widely supported."
    },
    "mpnet": {
        "name": "all-mpnet-base-v2",
        "desc": "Larger, higher quality, slower."
    },
    "distiluse": {
        "name": "distiluse-base-multilingual-cased-v2",
        "desc": "Multilingual support."
    },
}
SELECTED_EMBEDDING_MODEL = EMBEDDING_MODELS["miniLM"]["name"]

# Cross-Encoder Models Registry
CROSS_ENCODER_MODELS = {
    "miniLM": {
        "name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "desc": "Recommended: Fast, accurate reranker."
    },
    "mpnet": {
        "name": "cross-encoder/ms-marco-MiniLM-L-12-v2",
        "desc": "Larger, more accurate, slower."
    },
}
SELECTED_CROSS_ENCODER_MODEL = CROSS_ENCODER_MODELS["miniLM"]["name"]

# LLM Models Registry
LLM_MODELS = {
    "mistral": {
        "name": "mistralai/Mistral-7B-Instruct-v0.2",
        "desc": "Recommended: Good balance of quality and speed."
    },
    "mixtral": {
        "name": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "desc": "Large Mixture-of-Experts model, high quality, higher resource usage."
    },
    "tinyllama": {
        "name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "desc": "Very fast, low resource, lower quality."
    },
    "tinyllama-onnx": {
        "name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "desc": "ONNX optimized version for maximum CPU speed.",
        "use_onnx": True
    },
    "llama3.1-8b": {
        "name": os.getenv("LLAMA3_8B_PATH") or "meta-llama/Llama-3.1-8B-Instruct",
        "desc": "Meta's latest 8B model with excellent reasoning and instruction following."
    },
    "llama3.1-70b": {
        "name": os.getenv("LLAMA3_70B_PATH") or "meta-llama/Llama-3.1-70B-Instruct",
        "desc": "Meta's flagship 70B model with state-of-the-art performance, requires significant resources."
    },
}
SELECTED_LLM_MODEL = LLM_MODELS["tinyllama-onnx"]["name"]

# Model registry for tracking status/compatibility
MODEL_REGISTRY = {
    "embedding": EMBEDDING_MODELS,
    "cross_encoder": CROSS_ENCODER_MODELS,
    "llm": LLM_MODELS
}

# -------------------
# Model selection variables (used throughout the codebase)
# -------------------
EMBEDDING_MODEL = SELECTED_EMBEDDING_MODEL
CROSS_ENCODER_MODEL = SELECTED_CROSS_ENCODER_MODEL
LLM_MODEL = SELECTED_LLM_MODEL

# Check if selected model has ONNX flag
for model_key, model_info in LLM_MODELS.items():
    if model_info["name"] == SELECTED_LLM_MODEL and model_info.get("use_onnx", False):
        USE_ONNX = True
        logger.info(f"ONNX optimization enabled for model: {SELECTED_LLM_MODEL}")

# -------------------
# Model API Keys
# -------------------
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
MIXTRAL_API_KEY = os.getenv(
    "MIXTRAL_API_KEY",
    os.getenv("HUGGINGFACE_TOKEN", "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")
)

# Use Mixtral key if Mixtral model is selected
if LLM_MODEL == LLM_MODELS.get("mixtral", {}).get("name"):
    HF_TOKEN = MIXTRAL_API_KEY
    logger.info("Using Mixtral API key for Mixtral model.")

if not HF_TOKEN:
    logger.warning(
        "HUGGINGFACE_TOKEN environment variable not set. "
        "Some models may not be accessible."
    )

# FAISS index configuration
FAISS_INDEX_TYPE = "IndexFlatL2"

# Search configuration
TOP_K_INITIAL = 30
TOP_K_FINAL = 3  # Increased from 3 to provide more context

# -------------------
# LLM pipeline/task configuration
# -------------------
LLM_TASK = "text-generation"
LLM_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LLM_MAX_LENGTH = 4096  # Increased for Llama 3.1 to prevent truncation
LLM_MAX_NEW_TOKENS = 2048  # Increased for Llama 3.1 to allow longer responses
LLM_TEMPERATURE = 0.3
LLM_TOP_P = 0.9
LLM_BATCH_SIZE = 1
LLM_LOAD_IN_8BIT = False
LLM_LOAD_IN_4BIT = True

# Python 3.12.10 optimizations
TORCH_COMPILE = True
CUDA_OPTIMIZATIONS = False
USE_ONNX = False  # Will be set to True if selected model has use_onnx=True

# Response quality settings
ENABLE_TRUNCATION_DETECTION = True
MIN_RESPONSE_WORDS = 50  # Minimum expected response length
MAX_RESPONSE_WORDS = 4000  # Increased for Llama 3.1 to allow longer responses


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


# Legacy path compatibility
PDF_INPUT_DIR = str(DATA_DIR)
OUTPUT_DIR = str(OUTPUTS_DIR)
ROOT_DIR = str(WORKSPACE_ROOT)