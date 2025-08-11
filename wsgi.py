#!/usr/bin/env python3
"""
WSGI entry point for Scaffold AI deployment
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

# Import the Flask app
from frontend.app_enhanced import app

# WSGI application
application = app

if __name__ == "__main__":
    # For local development
    port = int(os.environ.get("PORT", 5002))
    app.run(host="0.0.0.0", port=port, debug=False)
