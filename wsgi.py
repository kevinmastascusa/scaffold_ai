#!/usr/bin/env python3
"""
WSGI entry point for Scaffold AI deployment
"""

import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

# Import the Flask app then expose as `application` for WSGI servers
from frontend.app_enhanced import app as application  # noqa: E402

if __name__ == "__main__":
    # For local development
    port = int(os.environ.get("PORT", 5002))
    application.run(host="0.0.0.0", port=port, debug=False)
