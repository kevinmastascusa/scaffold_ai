#!/usr/bin/env python3
"""
Setup script for Scaffold AI project.
Ensures all necessary directories exist and validates the workspace structure.
"""

import os
import sys
from pathlib import Path

# Add scaffold_core to Python path
sys.path.insert(0, str(Path(__file__).parent / "scaffold_core"))

from scaffold_core.config import ensure_directories, WORKSPACE_ROOT

# Version: 0.1.0-prebeta

def main():
    """Main setup function"""
    print("🌱 Scaffold AI Setup")
    print("=" * 50)
    
    # Ensure all directories exist
    print("Creating necessary directories...")
    ensure_directories()
    
    # Validate workspace structure
    print("\nValidating workspace structure...")
    
    required_dirs = [
        WORKSPACE_ROOT / "data",
        WORKSPACE_ROOT / "outputs", 
        WORKSPACE_ROOT / "vector_outputs",
        WORKSPACE_ROOT / "math_outputs"
    ]
    
    for directory in required_dirs:
        if directory.exists():
            print(f"✓ {directory.name}/")
        else:
            print(f"✗ {directory.name}/ (will be created)")
    
    # Check for data files
    data_dir = WORKSPACE_ROOT / "data"
    if data_dir.exists():
        pdf_files = list(data_dir.rglob("*.pdf"))
        print(f"\n📄 Found {len(pdf_files)} PDF files in data/ directory")
        
        if pdf_files:
            print("Sample files:")
            for pdf in pdf_files[:5]:  # Show first 5 files
                print(f"  - {pdf.name}")
            if len(pdf_files) > 5:
                print(f"  ... and {len(pdf_files) - 5} more")
        else:
            print("⚠️  No PDF files found in data/ directory")
            print("   Please add your PDF documents to the data/ directory")
    else:
        print("⚠️  data/ directory not found")
    
    print("\n✅ Setup complete!")
    print("\nNext steps:")
    print("1. Add your PDF documents to the data/ directory")
    print("2. Run the chunking script: python scaffold_core/scripts/chunk/ChunkTest.py")
    print("3. Run the vector processing: python scaffold_core/vector/main.py")
    print("\nFor more information, see README.md")

if __name__ == "__main__":
    main() 