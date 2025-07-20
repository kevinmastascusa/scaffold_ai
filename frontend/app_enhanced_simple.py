#!/usr/bin/env python3
"""
Scaffold AI Enhanced UI - Simplified Version
A Flask web application with PDF upload and conversational chat features.
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
sys.path.append(project_root)

# Initialize Flask app
app = Flask(__name__, template_folder='templates')
CORS(app)

# Configuration
app.config['SECRET_KEY'] = 'scaffold-ai-enhanced-ui-key'
app.config['DEBUG'] = True
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

# Global variables for lazy loading
enhanced_query_system = None
query_enhanced = None


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_conversation_history(session_id):
    """Get conversation history for a session."""
    conversation_file = CONVERSATIONS_DIR / f"{session_id}.json"
    if conversation_file.exists():
        with open(conversation_file, 'r') as f:
            return json.load(f)
    return []


def save_conversation_history(session_id, conversation):
    """Save conversation history for a session."""
    conversation_file = CONVERSATIONS_DIR / f"{session_id}.json"
    with open(conversation_file, 'w') as f:
        json.dump(conversation, f, indent=2, default=str)


def initialize_query_system():
    """Lazy initialization of the enhanced query system."""
    global enhanced_query_system, query_enhanced
    
    if enhanced_query_system is None:
        try:
            print("🔄 Initializing enhanced query system...")
            from scaffold_core.vector.enhanced_query import (
                query_enhanced, enhanced_query_system
            )
            enhanced_query_system.initialize()
            print("✅ Enhanced query system initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize enhanced query system: {e}")
            raise


@app.route('/')
def index():
    """Main page of the enhanced UI."""
    # Initialize session if not exists
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    
    return render_template('index_enhanced.html')


@app.route('/api/upload-syllabus', methods=['POST'])
def upload_syllabus():
    """API endpoint for uploading syllabus PDF."""
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        # Check if file was selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check if file type is allowed
        if not allowed_file(file.filename):
            return jsonify({'error': 'Only PDF files are allowed'}), 400
        
        # Secure the filename and save
        filename = secure_filename(file.filename)
        session_id = session.get('session_id', str(uuid.uuid4()))
        file_path = UPLOAD_FOLDER / f"{session_id}_{filename}"
        
        file.save(file_path)
        
        # Process the PDF and extract content
        try:
            from scaffold_core.pdf_processor import process_syllabus_upload
            processing_result = process_syllabus_upload(str(file_path), session_id)
            
            if processing_result['processing_status'] == 'success':
                return jsonify({
                    'success': True,
                    'filename': filename,
                    'message': 'Syllabus uploaded and analyzed successfully',
                    'file_path': str(file_path),
                    'analysis': processing_result['analysis'],
                    'suggestions': processing_result['sustainability_suggestions']
                })
            else:
                return jsonify({
                    'success': False,
                    'filename': filename,
                    'message': 'Syllabus uploaded but analysis failed',
                    'error': processing_result.get('error_message', 'Unknown error')
                })
                
        except Exception as e:
            return jsonify({
                'success': False,
                'filename': filename,
                'message': 'Syllabus uploaded but processing failed',
                'error': str(e)
            })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/chat', methods=['POST'])
def chat():
    """API endpoint for conversational chat."""
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        session_id = session.get('session_id', str(uuid.uuid4()))

        if not message:
            return jsonify({'error': 'Message is required'}), 400

        # Initialize query system if needed
        initialize_query_system()

        # Get conversation history
        conversation = get_conversation_history(session_id)
        
        # Add user message to conversation
        user_message = {
            'id': str(uuid.uuid4()),
            'type': 'user',
            'content': message,
            'timestamp': datetime.datetime.now().isoformat()
        }
        conversation.append(user_message)

        # Process the query using the enhanced system
        result = query_enhanced(message)
        
        # Create AI response
        ai_message = {
            'id': str(uuid.uuid4()),
            'type': 'assistant',
            'content': result['response'],
            'sources': [
                {
                    'source': candidate.get('source', {}),
                    'score': candidate.get(
                        'cross_score', candidate.get('score', 0)
                    ),
                    'text_preview': (
                        candidate.get('text', '')[:200] + '...'
                        if len(candidate.get('text', '')) > 200
                        else candidate.get('text', '')
                    )
                }
                for candidate in result['candidates'][:5]
            ],
            'timestamp': datetime.datetime.now().isoformat()
        }
        conversation.append(ai_message)

        # Save updated conversation
        save_conversation_history(session_id, conversation)

        return jsonify({
            'success': True,
            'response': ai_message,
            'conversation_id': session_id
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/conversation', methods=['GET'])
def get_conversation():
    """Get conversation history for current session."""
    try:
        session_id = session.get('session_id', str(uuid.uuid4()))
        conversation = get_conversation_history(session_id)
        
        return jsonify({
            'success': True,
            'conversation': conversation,
            'session_id': session_id
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/clear-conversation', methods=['POST'])
def clear_conversation():
    """Clear conversation history for current session."""
    try:
        session_id = session.get('session_id', str(uuid.uuid4()))
        conversation_file = CONVERSATIONS_DIR / f"{session_id}.json"
        
        if conversation_file.exists():
            conversation_file.unlink()
        
        return jsonify({
            'success': True,
            'message': 'Conversation cleared'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/query', methods=['POST'])
def api_query():
    """Legacy API endpoint for processing queries (maintained for compatibility)."""
    try:
        data = request.get_json()
        query = data.get('query', '').strip()

        if not query:
            return jsonify({'error': 'Query is required'}), 400

        # Initialize query system if needed
        initialize_query_system()

        # Process the query using the enhanced system
        result = query_enhanced(query)
        
        # Debug: Print the first candidate to see its structure
        if result['candidates']:
            print(f"DEBUG: First candidate structure: "
                  f"{result['candidates'][0]}")

        # Format the response for the UI
        response = {
            'query': query,
            'response': result['response'],
            'candidates_found': result['search_stats']['final_candidates'],
            'search_stats': result['search_stats'],
            'sources': [
                {
                    'source': candidate.get('source', {}),
                    'score': candidate.get(
                        'cross_score', candidate.get('score', 0)
                    ),
                    'text_preview': (
                        candidate.get('text', '')[:200] + '...'
                        if len(candidate.get('text', '')) > 200
                        else candidate.get('text', '')
                    )
                }
                for candidate in result['candidates'][:5]
            ],
            'timestamp': datetime.datetime.now().isoformat()
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/feedback', methods=['POST'])
def api_feedback():
    """API endpoint for collecting user feedback."""
    try:
        data = request.get_json()
        
        # Create feedback entry
        feedback = {
            'timestamp': datetime.datetime.now().isoformat(),
            'query': data.get('query', ''),
            'response': data.get('response', ''),
            'rating': data.get('rating', 0),
            'comments': data.get('comments', ''),
            'sources_count': data.get('sources_count', 0),
            'session_id': session.get('session_id', 'unknown')
        }
        
        # Save feedback to file
        feedback_file = FEEDBACK_DIR / f"feedback_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(feedback_file, 'w') as f:
            json.dump(feedback, f, indent=2)
        
        return jsonify({'success': True, 'message': 'Feedback saved'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/health')
def api_health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.datetime.now().isoformat(),
        'enhanced_query_system': 'initialized' if enhanced_query_system is not None else 'not_initialized'
    })


@app.route('/feedback')
def feedback_page():
    """Feedback dashboard page."""
    # Get all feedback files
    feedback_files = list(FEEDBACK_DIR.glob('feedback_*.json'))
    feedback_data = []
    
    for file in feedback_files[-10:]:  # Last 10 feedback entries
        try:
            with open(file, 'r') as f:
                feedback_data.append(json.load(f))
        except Exception as e:
            print(f"Error reading feedback file {file}: {e}")
    
    return render_template('feedback.html', feedback_data=feedback_data)


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Scaffold AI Enhanced UI - Simplified')
    parser.add_argument('--port', type=int, default=5003, help='Port to run the server on')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to run the server on')
    
    args = parser.parse_args()
    
    print(f"🚀 Starting Scaffold AI Enhanced UI (Simplified) on {args.host}:{args.port}")
    print(f"📍 Access the UI at: http://localhost:{args.port}")
    print(f"📊 Feedback dashboard at: http://localhost:{args.port}/feedback")
    print(f"💡 Press Ctrl+C to stop the server")
    print(f"⚠️  Enhanced query system will initialize on first request")
    
    app.run(host=args.host, port=args.port, debug=True) 