"""
Vector processing module for Scaffold AI.
Contains functionality for chunking, embedding, and querying text data.
"""

from .query import main as query_main
from .chunk import main as chunk_main
from .transformVector import main as transform_main

__all__ = ['query_main', 'chunk_main', 'transform_main'] 