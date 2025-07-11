#!/usr/bin/env python3
"""
Citation Handling for Scaffold AI
Provides a structured way to manage, format, and reference source documents.
"""

import hashlib
import os


class Citation:
    """
    Represents a source citation with a clean name and a unique ID.
    """
    def __init__(self, source_path: str):
        """
        Initializes a Citation object from a raw source file path.

        Args:
            source_path: The raw file path to the source document.
        """
        if not source_path or not isinstance(source_path, str):
            raise ValueError("A valid source path is required.")

        self.raw_path: str = source_path
        self.clean_name: str = self._format_name(source_path)
        self.id: str = self._generate_id(source_path)

    def _format_name(self, path: str) -> str:
        """
        Formats the raw file path into a clean, human-readable name.
        Example: 'data/Sust_in_Eng.pdf' -> 'Sust in Eng'
        """
        base_name = os.path.basename(path)
        name_without_ext = os.path.splitext(base_name)[0]
        # Replace underscores and hyphens with spaces
        return name_without_ext.replace('_', ' ').replace('-', ' ').strip()

    def _generate_id(self, path: str) -> str:
        """
        Generates a short, unique, and persistent identifier for the source.
        This ensures a stable reference even if the file is renamed.
        """
        # Using a partial SHA-256 hash for a short but unique ID
        return hashlib.sha256(path.encode()).hexdigest()[:8]

    def to_dict(self) -> dict:
        """
        Returns a dictionary representation of the citation.
        """
        return {
            'id': self.id,
            'name': self.clean_name,
            'raw_path': self.raw_path
        }

    def __repr__(self) -> str:
        return f"Citation(id='{self.id}', name='{self.clean_name}')" 