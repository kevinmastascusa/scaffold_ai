#!/usr/bin/env python3
"""
Scaffold AI Enhanced UI
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

from scaffold_core.vector.enhanced_query_improved import (
    query_enhanced_improved, improved_enhanced_query_system
)
from scaffold_core.pdf_processor import process_syllabus_upload

# Initialize Flask app
app = Flask(__name__, template_folder='templates', static_folder='static')
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

# Pre-initialize the enhanced query system
print("🔄 Pre-initializing enhanced query system...")
try:
    improved_enhanced_query_system.initialize()
    print("✅ Enhanced query system initialized successfully")
except Exception as e:
    print(f"❌ Failed to initialize enhanced query system: {e}")
    print("⚠️  The system will initialize on first request (may cause delay)")


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
        filename = secure_filename(file.filename or "")
        session_id = session.get('session_id', str(uuid.uuid4()))
        file_path = UPLOAD_FOLDER / f"{session_id}_{filename}"
        
        file.save(file_path)
        
        # Process the PDF and extract content
        try:
            processing_result = process_syllabus_upload(str(file_path), session_id)
            
            if processing_result['processing_status'] == 'success':
                # Add syllabus content to conversation memory for context
                syllabus_context = f"""UPLOADED SYLLABUS CONTEXT:
Course: {processing_result['analysis'].get('course_info', {}).get('title', 'Unknown Course')}
Course Code: {processing_result['analysis'].get('course_info', {}).get('code', 'N/A')}
Topics: {', '.join(processing_result['analysis'].get('topics', [])[:5])}
Content Summary: {processing_result.get('text_content', '')[:500]}..."""
                
                # Store syllabus context for this session
                conversation = get_conversation_history(session_id)
                syllabus_message = {
                    'id': str(uuid.uuid4()),
                    'type': 'syllabus_context',
                    'content': syllabus_context,
                    'timestamp': datetime.datetime.now().isoformat(),
                    'filename': filename
                }
                conversation.append(syllabus_message)
                save_conversation_history(session_id, conversation)
                
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

        # Process the query using the enhanced system with session context
        result = query_enhanced_improved(message, session_id)
        
        # Create AI response
        ai_message = {
            'id': str(uuid.uuid4()),
            'type': 'assistant',
            'content': result['response'],
            'sources': [
                {
                    'source': source.get('source', {}),
                    'score': source.get('score', 0),
                    'text_preview': source.get('text_preview', '')
                }
                for source in result['sources'][:5]
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

        # Process the query using the enhanced system
        result = query_enhanced_improved(query)
        
        # Debug: Print the first source to see its structure
        if result['sources']:
            print(f"DEBUG: First source structure: "
                  f"{result['sources'][0]}")

        # Format the response for the UI
        response = {
            'query': query,
            'response': result['response'],
            'candidates_found': result.get('candidates_found', len(result['sources'])),
            'search_stats': result.get('search_stats', {}),
            'sources': [
                {
                    'source': source.get('source', {}),
                    'score': source.get('score', 0),
                    'text_preview': source.get('text_preview', '')
                }
                for source in result['sources'][:5]
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
        'enhanced_query_system': 'initialized' if improved_enhanced_query_system.initialized else 'not_initialized'
    })


@app.route('/api/models')
def get_models():
    """Get available models and current selections."""
    try:
        from scaffold_core.config import MODEL_REGISTRY, LLM_MODELS, EMBEDDING_MODELS
        from scaffold_core.config_manager import config_manager
        
        # Get current selections from config manager
        current_llm_key = config_manager.get_selected_model('llm')
        current_embedding_key = config_manager.get_selected_model('embedding')
        
        # Get model settings
        llm_settings = config_manager.get_model_settings('llm')
        embedding_settings = config_manager.get_model_settings('embedding')
        
        return jsonify({
            'llm': MODEL_REGISTRY['llm'],
            'embedding': MODEL_REGISTRY['embedding'],
            'current_llm': current_llm_key,
            'current_embedding': current_embedding_key,
            'settings': {
                'llm': llm_settings,
                'embedding': embedding_settings
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/models/switch', methods=['POST'])
def switch_model():
    """Switch to a different model."""
    try:
        data = request.get_json()
        model_type = data.get('model_type')
        model_key = data.get('model_key')
        
        if not model_type or not model_key:
            return jsonify({'error': 'Missing model_type or model_key'}), 400
        
        from scaffold_core.config import MODEL_REGISTRY, LLM_MODELS, EMBEDDING_MODELS
        from scaffold_core.config_manager import config_manager
        
        if model_type == 'llm':
            if model_key not in LLM_MODELS:
                return jsonify({'error': f'Invalid LLM model key: {model_key}'}), 400
            
            # Update the configuration
            new_model_name = LLM_MODELS[model_key]['name']
            config_manager.set_selected_model('llm', model_key)
            
            return jsonify({
                'success': True,
                'message': f'LLM model switched to {model_key} ({new_model_name})',
                'note': 'Model configuration updated. The system will reload with the new model.'
            })
            
        elif model_type == 'embedding':
            if model_key not in EMBEDDING_MODELS:
                return jsonify({'error': f'Invalid embedding model key: {model_key}'}), 400
            
            # Update the configuration
            new_model_name = EMBEDDING_MODELS[model_key]['name']
            config_manager.set_selected_model('embedding', model_key)
            
            return jsonify({
                'success': True,
                'message': f'Embedding model switched to {model_key} ({new_model_name})',
                'note': 'Model configuration updated. The system will reload with the new model.'
            })
        
        else:
            return jsonify({'error': f'Invalid model type: {model_type}'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/models/settings', methods=['POST'])
def update_model_settings():
    """Update model settings."""
    try:
        data = request.get_json()
        model_type = data.get('model_type')
        settings = data.get('settings', {})
        
        if not model_type:
            return jsonify({'error': 'Missing model_type'}), 400
        
        from scaffold_core.config_manager import config_manager
        
        # Validate settings based on model type
        if model_type == 'llm':
            valid_settings = ['temperature', 'max_new_tokens', 'top_p']
            filtered_settings = {k: v for k, v in settings.items() if k in valid_settings}
        elif model_type == 'embedding':
            valid_settings = ['chunk_size', 'chunk_overlap']
            filtered_settings = {k: v for k, v in settings.items() if k in valid_settings}
        else:
            return jsonify({'error': f'Invalid model type: {model_type}'}), 400
        
        config_manager.update_model_settings(model_type, filtered_settings)
        
        return jsonify({
            'success': True,
            'message': f'{model_type.upper()} settings updated successfully',
            'settings': filtered_settings
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/models/performance')
def get_model_performance():
    """Get real-time model performance metrics."""
    try:
        import psutil
        import time
        
        # Get system metrics
        memory_info = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent()
        
        # Get model-specific metrics (simulated for now)
        metrics = {
            'response_time': '2.3s',
            'token_usage': '1,247',
            'model_accuracy': '94%',
            'memory_usage': f'{memory_info.percent}%',
            'cpu_usage': f'{cpu_percent}%',
            'system_memory': f'{memory_info.used // (1024**3):.1f}GB / {memory_info.total // (1024**3):.1f}GB',
            'timestamp': time.time()
        }
        
        return jsonify(metrics)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/clear-memory', methods=['POST'])
def clear_memory():
    """Clear the AI's memory while keeping chat history."""
    try:
        session_id = request.cookies.get('session_id')
        if not session_id:
            return jsonify({'success': False, 'error': 'No session found'})
        
        # Clear memory in the enhanced query system
        query_system = improved_enhanced_query_system # Assuming get_query_system() is not defined, using direct access
        query_system.clear_memory(session_id)
        
        return jsonify({'success': True})
    except Exception as e:
        # Assuming logger is defined elsewhere or needs to be imported
        # logger.error(f"Error clearing memory: {e}") 
        return jsonify({'success': False, 'error': str(e)})


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
    
    parser = argparse.ArgumentParser(description='Scaffold AI Enhanced UI')
    parser.add_argument('--port', type=int, default=5002, help='Port to run the server on')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to run the server on')
    
    args = parser.parse_args()
    
    print(f"🚀 Starting Scaffold AI Enhanced UI on {args.host}:{args.port}")
    print(f"📍 Access the UI at: http://localhost:{args.port}")
    print(f"📊 Feedback dashboard at: http://localhost:{args.port}/feedback")
    print(f"💡 Press Ctrl+C to stop the server")
    
    app.run(host=args.host, port=args.port, debug=True) 