#!/usr/bin/env python3
"""
Startup script for Scaffold AI Enhanced UI
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        'flask',
        'flask_cors',
        'sentence_transformers',
        'transformers',
        'torch',
        'faiss',
        'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nPlease install missing packages with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def check_data_files():
    """Check if required data files exist."""
    # Get project root directory
    project_root = Path(__file__).parent.parent
    
    required_files = [
        "vector_outputs/scaffold_index_1.faiss",
        "vector_outputs/scaffold_metadata_1.json"
    ]
    
    missing_files = []
    
    for file_path in required_files:
        full_path = project_root / file_path
        if not full_path.exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required data files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print("\nPlease ensure the vector index and metadata files are available.")
        return False
    
    return True

def main():
    """Main startup function."""
    parser = argparse.ArgumentParser(description='Scaffold AI Enhanced UI Startup')
    parser.add_argument('--port', type=int, default=5002, help='Port to run the server on')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to run the server on')
    parser.add_argument('--skip-checks', action='store_true', help='Skip dependency and data file checks')
    
    args = parser.parse_args()
    
    print("🚀 Scaffold AI Enhanced UI - Startup Check")
    print("=" * 50)
    
    # Check Python version
    python_version = sys.version_info
    print(f"✅ Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if not args.skip_checks:
        # Check dependencies
        print("📦 Checking Dependencies...")
        if not check_dependencies():
            return 1
        
        # Check data files
        print("📁 Checking Data Files...")
        if not check_data_files():
            return 1
    
    print("🎉 All checks passed! Starting the Enhanced UI...")
    print("=" * 50)
    
    # Set environment variables
    if 'HUGGINGFACE_TOKEN' not in os.environ:
        print("⚠️  HUGGINGFACE_TOKEN not set. Some models may not be accessible.")
    
    # Import and run the enhanced app
    try:
        from frontend.app_enhanced import app
        
        print(f"📍 Access the Enhanced UI at: http://localhost:{args.port}")
        print(f"📊 Feedback dashboard at: http://localhost:{args.port}/feedback")
        print(f"💡 Press Ctrl+C to stop the server")
        
        app.run(host=args.host, port=args.port, debug=False, use_reloader=False, threaded=True)
        
    except ImportError as e:
        print(f"❌ Error importing enhanced app: {e}")
        return 1
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return 1

if __name__ == '__main__':
    exit(main()) 