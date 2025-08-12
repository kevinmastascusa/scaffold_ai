#!/usr/bin/env python3
"""
Railway-optimized Flask app with lazy loading to avoid boot timeouts.
"""

import datetime
import json
import os
import sys
import uuid
from pathlib import Path
from werkzeug.utils import secure_filename

from flask import Flask, jsonify, render_template, request, session
from flask_cors import CORS

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# Initialize Flask app
app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# Configuration
app.config['SECRET_KEY'] = 'scaffold-ai-railway-ui-key'
app.config['DEBUG'] = False
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# File upload configuration
UPLOAD_FOLDER = Path("uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True)
ALLOWED_EXTENSIONS = {'pdf'}

# Create directories for storing feedback, logs, and conversations
FEEDBACK_DIR = Path("ui_feedback")
FEEDBACK_DIR.mkdir(exist_ok=True)

CONVERSATIONS_DIR = Path("conversations")
CONVERSATIONS_DIR.mkdir(exist_ok=True)

print("🚀 Railway app starting - models will load on first request")

# Lazy-loaded singletons
_query_system = None


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_query_system():
    global _query_system
    if _query_system is None:
        print("🔄 Loading AI models (first request)...")
        from scaffold_core.vector.enhanced_query_improved import (
            improved_enhanced_query_system,
        )
        improved_enhanced_query_system.initialize()
        _query_system = improved_enhanced_query_system
        print("✅ AI models loaded successfully")
    return _query_system


def get_conversation_history(session_id: str):
    conversation_file = CONVERSATIONS_DIR / f"{session_id}.json"
    if conversation_file.exists():
        with open(conversation_file, 'r') as f:
            return json.load(f)
    return []


def save_conversation_history(session_id: str, conversation):
    conversation_file = CONVERSATIONS_DIR / f"{session_id}.json"
    with open(conversation_file, 'w') as f:
        json.dump(conversation, f, indent=2, default=str)


@app.route('/')
def index():
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return render_template('index_enhanced.html')


@app.route('/api/health')
def health():
    return jsonify({"status": "ok"})


@app.route('/api/upload-syllabus', methods=['POST'])
def upload_syllabus():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Only PDF files are allowed'}), 400

        # Save file
        filename = secure_filename(file.filename or "")
        session_id = session.get('session_id', str(uuid.uuid4()))
        file_path = UPLOAD_FOLDER / f"{session_id}_{filename}"
        file.save(file_path)

        # Lazy import and process
        from scaffold_core.pdf_processor import process_syllabus_upload
        processing_result = process_syllabus_upload(str(file_path), session_id)

        if processing_result.get('processing_status') != 'success':
            return jsonify({'error': 'Failed to process PDF'}), 500

        # Store syllabus context for this session
        conversation = get_conversation_history(session_id)
        syllabus_message = {
            'id': str(uuid.uuid4()),
            'type': 'syllabus_context',
            'content': processing_result.get('text_content', '')[:500],
            'timestamp': datetime.datetime.now().isoformat(),
            'filename': filename,
        }
        conversation.append(syllabus_message)
        save_conversation_history(session_id, conversation)

        return jsonify({
            'success': True,
            'filename': filename,
            'message': 'Syllabus uploaded and analyzed successfully',
            'file_path': str(file_path),
            'analysis': processing_result.get('analysis', {}),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json(force=True)
        query = data.get('query', '').strip()
        if not query:
            return jsonify({'error': 'Empty query'}), 400

        session_id = session.get('session_id', str(uuid.uuid4()))

        # Ensure models are loaded
        query_system = get_query_system()

        from scaffold_core.vector.enhanced_query_improved import (
            query_enhanced_improved,
        )
        result = query_enhanced_improved(query, session_id)

        # Save conversation
        conversation = get_conversation_history(session_id)
        conversation.append({
            'id': str(uuid.uuid4()),
            'type': 'user',
            'content': query,
            'timestamp': datetime.datetime.now().isoformat(),
        })
        conversation.append({
            'id': str(uuid.uuid4()),
            'type': 'assistant',
            'content': result.get('answer', ''),
            'timestamp': datetime.datetime.now().isoformat(),
        })
        save_conversation_history(session_id, conversation)

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500



