#!/usr/bin/env python3
"""
Scaffold AI Enhanced UI - Simplified Version
A simplified Flask web application for the Scaffold AI interface with PDF upload and chat functionality.
This version avoids the enhanced query system to prevent crashes.
"""

import os
import sys
import json
import uuid
import datetime
from pathlib import Path
from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Initialize Flask app
app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# Configuration
app.secret_key = 'scaffold_ai_enhanced_ui_secret_key_2024'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Directories
UPLOAD_FOLDER = project_root / 'uploads'
CONVERSATIONS_DIR = project_root / 'conversations'
FEEDBACK_DIR = project_root / 'feedback'

# Create directories if they don't exist
UPLOAD_FOLDER.mkdir(exist_ok=True)
CONVERSATIONS_DIR.mkdir(exist_ok=True)
FEEDBACK_DIR.mkdir(exist_ok=True)

# Global variables for query system (disabled to prevent crashes)
enhanced_query_system = None
query_enhanced = None

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'pdf'

def get_conversation_history(session_id):
    """Get conversation history for a session."""
    conversation_file = CONVERSATIONS_DIR / f"{session_id}.json"
    if conversation_file.exists():
        try:
            with open(conversation_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading conversation file: {e}")
    return []

def save_conversation_history(session_id, conversation):
    """Save conversation history for a session."""
    conversation_file = CONVERSATIONS_DIR / f"{session_id}.json"
    try:
        with open(conversation_file, 'w') as f:
            json.dump(conversation, f, indent=2)
    except Exception as e:
        print(f"Error saving conversation file: {e}")

def get_simple_response(message):
    """Generate a simple response without the enhanced query system."""
    # Simple keyword-based responses for sustainability education
    message_lower = message.lower()
    
    if 'fluid mechanics' in message_lower or 'fluid' in message_lower:
        return {
            'content': "For Fluid Mechanics courses, you can integrate sustainability by:\n\n• **Energy Efficiency**: Discuss pump and turbine efficiency, renewable energy applications\n• **Environmental Impact**: Cover water treatment, pollution control, and sustainable water management\n• **Case Studies**: Include examples of sustainable fluid systems in buildings and infrastructure\n• **Green Technologies**: Explore wind turbines, hydroelectric power, and sustainable HVAC systems\n\nWould you like me to provide more specific examples for any of these areas?",
            'sources': []
        }
    elif 'thermodynamics' in message_lower:
        return {
            'content': "Thermodynamics offers excellent opportunities for sustainability integration:\n\n• **Energy Conservation**: First and Second Law applications to sustainable systems\n• **Renewable Energy**: Solar thermal, geothermal, and biomass energy cycles\n• **Efficiency Analysis**: Heat engines, refrigeration cycles, and combined heat and power\n• **Environmental Thermodynamics**: Climate change, atmospheric processes, and carbon cycles\n\nI can help you develop specific modules for any of these topics.",
            'sources': []
        }
    elif 'materials' in message_lower:
        return {
            'content': "Materials engineering is crucial for sustainability:\n\n• **Green Materials**: Biodegradable polymers, sustainable composites, and recycled materials\n• **Life Cycle Assessment**: Environmental impact analysis of material choices\n• **Energy Materials**: Solar cells, batteries, and energy storage materials\n• **Circular Economy**: Design for disassembly, recycling, and material recovery\n\nLet me know if you'd like specific examples for your course!",
            'sources': []
        }
    elif 'design' in message_lower:
        return {
            'content': "Engineering design is perfect for sustainability integration:\n\n• **Sustainable Design Principles**: Cradle-to-cradle, biomimicry, and green design\n• **Environmental Impact**: Life cycle assessment and environmental footprint analysis\n• **Social Responsibility**: Human-centered design and community impact\n• **Innovation**: Sustainable technology development and green innovation\n\nI can help you create design projects that incorporate these principles.",
            'sources': []
        }
    elif 'syllabus' in message_lower or 'upload' in message_lower:
        return {
            'content': "I can help you analyze your syllabus and provide personalized sustainability integration suggestions! Please upload your syllabus PDF using the upload button in the sidebar. Once uploaded, I'll analyze the content and provide specific recommendations for incorporating sustainability into your course.",
            'sources': []
        }
    else:
        return {
            'content': f"I understand you're asking about: '{message}'. I'm here to help you integrate sustainability into your engineering courses. You can:\n\n• Ask me about specific courses (like Fluid Mechanics, Thermodynamics, Materials, etc.)\n• Upload your syllabus for personalized suggestions\n• Get examples of sustainability integration for different engineering topics\n\nWhat specific course or topic would you like to discuss?",
            'sources': []
        }

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
        
        # Generate simple analysis and suggestions
        suggestions = [
            "Consider adding a sustainability module to your course",
            "Include case studies of sustainable engineering practices",
            "Incorporate life cycle assessment principles",
            "Add assignments focused on environmental impact analysis"
        ]
        
        return jsonify({
            'success': True,
            'filename': filename,
            'message': 'Syllabus uploaded successfully',
            'file_path': str(file_path),
            'suggestions': suggestions
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

        # Generate response using simple system
        response_data = get_simple_response(message)
        
        # Create AI response
        ai_message = {
            'id': str(uuid.uuid4()),
            'type': 'assistant',
            'content': response_data['content'],
            'sources': response_data['sources'],
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
        print(f"Error in chat API: {e}")
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
    """Legacy query endpoint - redirects to chat."""
    try:
        data = request.get_json()
        message = data.get('query', '').strip()
        
        if not message:
            return jsonify({'error': 'Query is required'}), 400
        
        # Use the same response system as chat
        response_data = get_simple_response(message)
        
        response = {
            'response': response_data['content'],
            'sources': response_data['sources'],
            'query': message
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
        'enhanced_query_system': 'disabled',
        'query_function_available': True,
        'mode': 'simple'
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
    print(f"⚠️  Running in simple mode - enhanced query system disabled")
    
    app.run(host=args.host, port=args.port, debug=True) 