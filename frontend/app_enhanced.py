#!/usr/bin/env python3
"""
Scaffold AI Enhanced UI
A Flask web application with PDF upload and conversational chat features.
"""
# flake8: noqa
# mypy: ignore-errors
# pylint: skip-file

import datetime
import json
import os
import sys
import uuid
from pathlib import Path

from flask import Flask, jsonify, render_template, request, session
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from scaffold_core.vector.enhanced_query_improved import (  # noqa: E402
    query_enhanced_improved, improved_enhanced_query_system
)
from scaffold_core.pdf_processor import process_syllabus_upload  # noqa: E402

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
        with open(conversation_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []


def save_conversation_history(session_id, conversation):
    """Save conversation history for a session."""
    conversation_file = CONVERSATIONS_DIR / f"{session_id}.json"
    with open(conversation_file, 'w', encoding='utf-8') as f:
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
            processing_result = process_syllabus_upload(
                str(file_path), session_id
            )

            if processing_result['processing_status'] == 'success':
                # Add syllabus content to conversation memory for context
                syllabus_context = (
                    "UPLOADED SYLLABUS CONTEXT:\n"
                    f"Course: {processing_result['analysis'].get('course_info', {}).get('title', 'Unknown Course')}\n"
                    f"Course Code: {processing_result['analysis'].get('course_info', {}).get('code', 'N/A')}\n"
                    f"Topics: {', '.join(processing_result['analysis'].get('topics', [])[:5])}\n"
                    f"Content Summary: {processing_result.get('text_content', '')[:500]}..."
                )

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
                    'suggestions': processing_result[
                        'sustainability_suggestions'
                    ]
                })
            else:
                return jsonify({
                    'success': False,
                    'filename': filename,
                    'message': 'Syllabus uploaded but analysis failed',
                    'error': processing_result.get(
                        'error_message', 'Unknown error'
                    )
                })

        except Exception as e:  # noqa: BLE001
            return jsonify({
                'success': False,
                'filename': filename,
                'message': 'Syllabus uploaded but processing failed',
                'error': str(e)
            })

    except Exception as e:  # noqa: BLE001
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

        # Optional: clear model memory per request or via env toggle
        auto_clear = str(os.getenv('SC_AUTO_CLEAR_MEMORY', '')).lower() in (
            '1', 'true', 'yes'
        )
        reset_memory = bool(data.get('reset_memory'))
        reset_conversation = bool(data.get('reset_conversation'))
        if auto_clear or reset_memory:
            try:
                improved_enhanced_query_system.clear_memory(session_id)
            except Exception:  # noqa: BLE001
                pass
            if reset_conversation:
                conversation_file = CONVERSATIONS_DIR / f"{session_id}.json"
                if conversation_file.exists():
                    conversation_file.unlink()

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

    except Exception as e:  # noqa: BLE001
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

    except Exception as e:  # noqa: BLE001
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

    except Exception as e:  # noqa: BLE001
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

        # Format the response for the UI
        response = {
            'query': query,
            'response': result['response'],
            'candidates_found': result.get(
                'candidates_found', len(result['sources'])
            ),
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

    except Exception as e:  # noqa: BLE001
        return jsonify({'error': str(e)}), 500


@app.route('/api/health')
def api_health():
    """Health check endpoint."""
    flags = {
        'SC_STRICT_ANSWERS': os.getenv('SC_STRICT_ANSWERS', ''),
        'SC_ENABLE_TRUNCATION_DETECTION': os.getenv('SC_ENABLE_TRUNCATION_DETECTION', ''),
        'SC_QUALITY_MODE': os.getenv('SC_QUALITY_MODE', ''),
        'SC_INCLUDE_CUDA': os.getenv('SC_INCLUDE_CUDA', ''),
        'SC_FORCE_CPU': os.getenv('SC_FORCE_CPU', ''),
        'SC_ENABLE_TOT': os.getenv('SC_ENABLE_TOT', ''),
        'SC_TOT_BREADTH': os.getenv('SC_TOT_BREADTH', ''),
        'SC_TOT_DEPTH': os.getenv('SC_TOT_DEPTH', ''),
        'SC_ENABLE_PROOFREAD': os.getenv('SC_ENABLE_PROOFREAD', ''),
    }
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.datetime.now().isoformat(),
        'enhanced_query_system': (
            'initialized' if improved_enhanced_query_system.initialized
            else 'not_initialized'
        ),
        'flags': flags,
    })


@app.route('/api/flags', methods=['GET'])
def get_flags():
    """Return current runtime flags so the UI can reflect state."""
    return jsonify({
        'SC_STRICT_ANSWERS': os.getenv('SC_STRICT_ANSWERS', ''),
        'SC_ENABLE_TRUNCATION_DETECTION': os.getenv('SC_ENABLE_TRUNCATION_DETECTION', ''),
        'SC_QUALITY_MODE': os.getenv('SC_QUALITY_MODE', ''),
        'SC_INCLUDE_CUDA': os.getenv('SC_INCLUDE_CUDA', ''),
        'SC_FORCE_CPU': os.getenv('SC_FORCE_CPU', ''),
        'SC_ENABLE_TOT': os.getenv('SC_ENABLE_TOT', ''),
        'SC_TOT_BREADTH': os.getenv('SC_TOT_BREADTH', ''),
        'SC_TOT_DEPTH': os.getenv('SC_TOT_DEPTH', ''),
        'SC_ENABLE_PROOFREAD': os.getenv('SC_ENABLE_PROOFREAD', ''),
        'success': True,
    })


@app.route('/api/flags', methods=['POST'])
def set_flags():
    """Update runtime flags from the UI. Some changes may require restart."""
    try:
        data = request.get_json() or {}

        def _as_env_bool(val):
            if isinstance(val, bool):
                return '1' if val else '0'
            s = str(val).lower()
            return '1' if s in ('1', 'true', 'yes', 'on') else '0'

        # Boolean flags
        if 'SC_STRICT_ANSWERS' in data:
            os.environ['SC_STRICT_ANSWERS'] = _as_env_bool(data['SC_STRICT_ANSWERS'])
        if 'SC_ENABLE_TRUNCATION_DETECTION' in data:
            os.environ['SC_ENABLE_TRUNCATION_DETECTION'] = _as_env_bool(data['SC_ENABLE_TRUNCATION_DETECTION'])
        if 'SC_QUALITY_MODE' in data:
            os.environ['SC_QUALITY_MODE'] = _as_env_bool(data['SC_QUALITY_MODE'])
        if 'SC_INCLUDE_CUDA' in data:
            os.environ['SC_INCLUDE_CUDA'] = _as_env_bool(data['SC_INCLUDE_CUDA'])
            # If GPU requested, clear any CPU-forcing env to prefer GPU
            if os.environ['SC_INCLUDE_CUDA'] == '1':
                os.environ['SC_FORCE_CPU'] = '0'
        if 'SC_FORCE_CPU' in data:
            os.environ['SC_FORCE_CPU'] = _as_env_bool(data['SC_FORCE_CPU'])
        if 'SC_ENABLE_PROOFREAD' in data:
            os.environ['SC_ENABLE_PROOFREAD'] = _as_env_bool(data['SC_ENABLE_PROOFREAD'])

        # ToT flags (numeric)
        if 'SC_ENABLE_TOT' in data:
            os.environ['SC_ENABLE_TOT'] = _as_env_bool(data['SC_ENABLE_TOT'])
        if 'SC_TOT_BREADTH' in data:
            os.environ['SC_TOT_BREADTH'] = str(data['SC_TOT_BREADTH'])
        if 'SC_TOT_DEPTH' in data:
            os.environ['SC_TOT_DEPTH'] = str(data['SC_TOT_DEPTH'])

        return jsonify({'success': True})
    except Exception as exc:  # noqa: BLE001
        return jsonify({'success': False, 'error': str(exc)}), 500


@app.route('/api/models')
def get_models():
    """Get available models and current selections."""
    try:
        from scaffold_core.config import (  # noqa: WPS433, E402
            MODEL_REGISTRY
        )
        from scaffold_core.config_manager import config_manager  # noqa: WPS433, E402

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
    except Exception as e:  # noqa: BLE001
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

        from scaffold_core.config import (  # noqa: WPS433, E402
            MODEL_REGISTRY, LLM_MODELS, EMBEDDING_MODELS
        )
        from scaffold_core.config import (  # noqa: WPS433, E402
            LLM_MODELS, EMBEDDING_MODELS
        )
        from scaffold_core.config_manager import config_manager  # noqa: WPS433, E402

        if model_type == 'llm':
            if model_key not in LLM_MODELS:
                return jsonify({
                    'error': f'Invalid LLM model key: {model_key}'
                }), 400

            # Update the configuration
            new_model_name = LLM_MODELS[model_key]['name']
            config_manager.set_selected_model('llm', model_key)

            return jsonify({
                'success': True,
                'message': f'LLM model switched to {model_key} ({new_model_name})',
                'note': ('Model configuration updated. The system will reload '
                        'with the new model.')
            })

        elif model_type == 'embedding':
            if model_key not in EMBEDDING_MODELS:
                return jsonify({
                    'error': f'Invalid embedding model key: {model_key}'
                }), 400

            # Update the configuration
            new_model_name = EMBEDDING_MODELS[model_key]['name']
            config_manager.set_selected_model('embedding', model_key)

            return jsonify({
                'success': True,
                'message': f'Embedding model switched to {model_key} ({new_model_name})',
                'note': ('Model configuration updated. The system will reload '
                        'with the new model.')
            })

        else:
            return jsonify({'error': f'Invalid model type: {model_type}'}), 400

    except Exception as e:  # noqa: BLE001
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

        from scaffold_core.config_manager import config_manager  # noqa: WPS433, E402

        # Validate settings based on model type
        if model_type == 'llm':
            valid_settings = ['temperature', 'max_new_tokens', 'top_p']
            filtered_settings = {
                k: v for k, v in settings.items() if k in valid_settings
            }
        elif model_type == 'embedding':
            valid_settings = ['chunk_size', 'chunk_overlap']
            filtered_settings = {
                k: v for k, v in settings.items() if k in valid_settings
            }
        else:
            return jsonify({'error': f'Invalid model type: {model_type}'}), 400

        config_manager.update_model_settings(model_type, filtered_settings)

        return jsonify({
            'success': True,
            'message': f'{model_type.upper()} settings updated successfully',
            'settings': filtered_settings
        })

    except Exception as e:  # noqa: BLE001
        return jsonify({'error': str(e)}), 500


@app.route('/api/models/performance')
def get_model_performance():
    """Get real-time model performance metrics."""
    try:
        import psutil  # noqa: WPS433
        import time  # noqa: WPS433

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
            'system_memory': (
                f'{memory_info.used // (1024**3):.1f}GB / '
                f'{memory_info.total // (1024**3):.1f}GB'
            ),
            'timestamp': time.time()
        }

        return jsonify(metrics)

    except Exception as e:  # noqa: BLE001
        return jsonify({'error': str(e)}), 500


@app.route('/api/clear-memory', methods=['POST'])
def clear_memory():
    """Clear the AI's memory while keeping chat history."""
    try:
        # Use Flask session (consistent with other endpoints)
        session_id = session.get('session_id')
        if not session_id:
            return jsonify({
                'success': False, 'error': 'No session found'
            }), 400

        # Clear memory in the enhanced query system
        improved_enhanced_query_system.clear_memory(session_id)

        return jsonify({'success': True})
    except Exception as e:  # noqa: BLE001
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/feedback')
def feedback_page():
    """Feedback dashboard page."""
    # Get all feedback files
    feedback_files = list(FEEDBACK_DIR.glob('feedback_*.json'))
    feedback_data = []

    for file in feedback_files[-10:]:  # Last 10 feedback entries
        try:
            with open(file, 'r', encoding='utf-8') as f:
                feedback_data.append(json.load(f))
        except Exception as e:  # noqa: BLE001
            print(f"Error reading feedback file {file}: {e}")

    return render_template('feedback.html', feedback_data=feedback_data)


@app.errorhandler(404)
def not_found(error):  # noqa: ARG001
    """Handle 404 errors."""
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def internal_error(error):  # noqa: ARG001
    """Handle 500 errors."""
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    import argparse  # noqa: WPS433

    parser = argparse.ArgumentParser(description='Scaffold AI Enhanced UI')
    parser.add_argument(
        '--port', type=int, default=5002, help='Port to run the server on'
    )
    parser.add_argument(
        '--host', type=str, default='0.0.0.0', help='Host to run the server on'
    )

    args = parser.parse_args()

    print(f"🚀 Starting Scaffold AI Enhanced UI on {args.host}:{args.port}")
    print(f"📍 Access the UI at: http://localhost:{args.port}")
    print(f"📊 Feedback dashboard at: http://localhost:{args.port}/feedback")
    print("💡 Press Ctrl+C to stop the server")

    app.run(host=args.host, port=args.port, debug=True)