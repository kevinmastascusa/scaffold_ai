#!/usr/bin/env python3
"""
Scaffold AI Pilot UI Startup Script
Checks dependencies and starts the Flask application.
"""

import os
import sys

# Add project root to path before other imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from scaffold_core.log_config import setup_logging


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required. You have:", sys.version)
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True


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
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} - Missing")

    if missing_packages:
        print(f"\n🔧 Missing packages: {', '.join(missing_packages)}")
        print("Please install requirements: pip install -r requirements.txt")
        return False

    return True


def check_data_files():
    """Check if necessary data files exist."""
    # Add project root for imports
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
    from scaffold_core.config import (
        get_faiss_index_path, get_metadata_json_path
    )

    required_files = [
        get_faiss_index_path(),
        get_metadata_json_path()
    ]

    missing_files = []

    for file_path in required_files:
        if not file_path.exists():
            missing_files.append(str(file_path))
            print(f"❌ {file_path} - Missing")
        else:
            print(f"✅ {file_path}")

    if missing_files:
        print(f"\n📁 Missing data files: {', '.join(missing_files)}")
        print("Please ensure the vector database has been created.")
        print("Run: python scaffold_core/vector/transformVector.py")
        return False

    return True


def main():
    """Main startup function."""
    print("🚀 Scaffold AI Pilot UI - Startup Check")
    print("=" * 50)

    # Check Python version
    if not check_python_version():
        sys.exit(1)

    print("\n📦 Checking Dependencies...")
    if not check_dependencies():
        sys.exit(1)

    print("\n📁 Checking Data Files...")
    if not check_data_files():
        print(
            "\n💡 Tip: If you haven't set up the vector database yet, "
            "refer to the setup guide."
        )
        response = input("\nDo you want to continue anyway? (y/N): ")
        if response.lower() != 'y':
            sys.exit(1)

    print("\n🎉 All checks passed! Starting the UI...")
    print("=" * 50)

    # Set up centralized logging for the app
    setup_logging()

    # Start the Flask application
    try:
        from app import app
        print("📍 Access the UI at: http://localhost:5002")
        print("📊 Feedback dashboard at: http://localhost:5002/feedback")
        print("💡 Press Ctrl+C to stop the server\n")

        app.run(host='0.0.0.0', port=5002, debug=True)

    except KeyboardInterrupt:
        print("\n👋 Shutting down gracefully...")
    except Exception as e:
        print(f"\n❌ Error starting the application: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 