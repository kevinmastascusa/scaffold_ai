#!/usr/bin/env python3
"""
Simple test version of the Scaffold AI Pilot UI
Uses mock responses to test the interface without loading the full LLM
"""

import os
import sys
import json
import datetime
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

# Add project root to Python path
project_root = Path(__file__).parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Initialize Flask app
# Point to the correct template and static folders
app = Flask(
    __name__,
    template_folder=str(project_root / 'templates'),
    static_folder=str(project_root / 'static')  # Assuming a static folder might exist
)
CORS(app)

# Configuration
app.config['SECRET_KEY'] = 'scaffold-ai-pilot-ui-test-key'
app.config['DEBUG'] = True

# Create directories for storing feedback and logs
FEEDBACK_DIR = project_root / "ui_feedback"
FEEDBACK_DIR.mkdir(exist_ok=True)

# Mock data for testing
# ... (rest of the file is unchanged) 