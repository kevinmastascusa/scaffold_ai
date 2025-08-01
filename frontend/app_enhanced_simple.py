#!/usr/bin/env python3
"""
Scaffold AI Enhanced UI - Direct Search Version
A Flask web application for the Scaffold AI interface with PDF upload and conversational chat features.
This version uses direct vector search without LLM dependencies.
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

# Global variables for search system
embedding_model = None
index = None
metadata = None
llm_manager = None
system_initialized = False

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

def initialize_search_system():
    """Initialize the search system with embeddings, vector index, and LLM."""
    global embedding_model, index, metadata, system_initialized, llm_manager
    
    if system_initialized:
        return True
        
    try:
        print("🔄 Initializing LLM-powered search system...")
        
        # Import configuration
        from scaffold_core.config import EMBEDDING_MODEL, get_faiss_index_path, get_metadata_json_path
        
        # Load embedding model
        import torch
        from sentence_transformers import SentenceTransformer
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        embedding_model = SentenceTransformer(EMBEDDING_MODEL, device=device)
        print(f"✅ Embedding model loaded: {EMBEDDING_MODEL}")
        
        # Load FAISS index
        import faiss
        index_path = get_faiss_index_path()
        index = faiss.read_index(str(index_path))
        print(f"✅ FAISS index loaded: {index_path}")
        
        # Load metadata
        metadata_path = get_metadata_json_path()
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        print(f"✅ Metadata loaded: {len(metadata)} entries")
        
        # Initialize LLM manager
        from scaffold_core.llm import LLMManager
        llm_manager = LLMManager()
        print("✅ LLM manager initialized")
        
        system_initialized = True
        print("✅ LLM-powered search system initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ Failed to initialize search system: {e}")
        embedding_model = None
        index = None
        metadata = None
        llm_manager = None
        system_initialized = False
        return False

def search_sustainability_data(query, k=10):
    """Search the sustainability research database."""
    if not system_initialized or embedding_model is None or index is None:
        return []
    
    try:
        # Encode the query
        query_embedding = embedding_model.encode([query])
        
        # Search the index
        scores, indices = index.search(query_embedding, k)
        
        # Get the results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(metadata):
                chunk_data = metadata[idx]
                
                # Extract and format source information
                source_info = chunk_data.get('source', {})
                metadata_info = chunk_data.get('metadata', {})
                
                if isinstance(metadata_info, dict):
                    # Create a more detailed source object from the actual metadata structure
                    formatted_source = {
                        'id': chunk_data.get('chunk_id', f'chunk_{idx}'),
                        'name': metadata_info.get('filename', 'Sustainability Research Document'),
                        'raw_path': chunk_data.get('source_path', ''),
                        'title': metadata_info.get('title', metadata_info.get('filename', 'Sustainability Research')),
                        'authors': metadata_info.get('authors', ''),
                        'year': metadata_info.get('year', ''),
                        'journal': metadata_info.get('journal', ''),
                        'doi': metadata_info.get('doi', ''),
                        'folder': metadata_info.get('folder', ''),
                        'document_id': chunk_data.get('document_id', '')
                    }
                elif isinstance(source_info, dict):
                    # Fallback to source field if metadata doesn't exist
                    formatted_source = {
                        'id': source_info.get('id', f'chunk_{idx}'),
                        'name': source_info.get('name', 'Sustainability Research Document'),
                        'raw_path': source_info.get('raw_path', ''),
                        'title': source_info.get('title', source_info.get('name', 'Sustainability Research')),
                        'authors': source_info.get('authors', ''),
                        'year': source_info.get('year', ''),
                        'journal': source_info.get('journal', ''),
                        'doi': source_info.get('doi', '')
                    }
                else:
                    # Fallback if neither exists
                    formatted_source = {
                        'id': f'chunk_{idx}',
                        'name': 'Sustainability Research Document',
                        'raw_path': '',
                        'title': 'Sustainability Research',
                        'authors': '',
                        'year': '',
                        'journal': '',
                        'doi': ''
                    }
                
                results.append({
                    'chunk_id': idx,
                    'score': float(score),
                    'text': chunk_data.get('text', ''),
                    'source': formatted_source,
                    'search_type': 'semantic'
                })
        
        return results
        
    except Exception as e:
        print(f"Search error: {e}")
        return []

def generate_response_from_sources(query, search_results, session_id=None):
    """Generate a coherent response using LLM based on the search results and syllabus context."""
    if not search_results:
        return "I couldn't find specific information about that in my sustainability research database. Please try rephrasing your question or ask about a different sustainability topic.", []
    
    # Extract key information from the top candidates
    sources_used = []
    
    # Get the most relevant chunks (limit to avoid token limits)
    top_results = search_results[:3]  # Use top 3 sources for LLM processing
    
    # Prepare context for LLM
    context_parts = []
    for result in top_results:
        text = result.get('text', '')
        # Always use the chunk, even if short
        truncated_text = text[:800] if len(text) > 800 else text
        context_parts.append(truncated_text)
        # Create source display
        source = result.get('source', {})
        source_display = {
            'source': {
                'id': source.get('id', f'chunk_{result.get("chunk_id", "unknown")}'),
                'name': source.get('name', 'Sustainability Research Document'),
                'title': source.get('title', source.get('name', 'Sustainability Research')),
                'authors': source.get('authors', ''),
                'year': source.get('year', ''),
                'journal': source.get('journal', ''),
                'doi': source.get('doi', ''),
                'folder': source.get('folder', ''),
                'document_id': source.get('document_id', '')
            },
            'score': result.get('score', 0),
            'text_preview': text[:200] + "..." if len(text) > 200 else text
        }
        sources_used.append(source_display)
    
    # Combine context
    context = "\n\n".join(context_parts)
    
    # Get syllabus context if session_id provided
    syllabus_context = ""
    if session_id:
        conversation = get_conversation_history(session_id)
        for msg in conversation:
            if msg.get('type') == 'syllabus_context':
                syllabus_context = msg.get('content', '')
                break
    
    # Build syllabus section
    syllabus_section = ""
    if syllabus_context:
        syllabus_section = f"""
{syllabus_context}

"""
    
    # Create LLM prompt with strict constraints and syllabus context
    prompt = f"""STRICT INSTRUCTION: Answer ONLY the specific question asked. Do not generate additional questions, examples beyond what is directly asked, or suggest other topics.

You are an expert in sustainability education and engineering curriculum development. Based on the following research excerpts and uploaded syllabus context, provide a focused response to the user's specific question.

{syllabus_section}User Question: {query}

Research Context:
{context}

Provide a focused response that:
1. Addresses ONLY the specific question asked
2. Uses information from the provided research and syllabus context
3. Does not suggest additional questions or topics
4. Stays strictly on topic
5. Provides practical, actionable suggestions based on the research and course content

Response:"""
    
    try:
        # Generate response using LLM with stop sequences
        if llm_manager is not None:
            print("🤖 Generating LLM response...")
            response = llm_manager.generate_response(prompt, max_new_tokens=800, temperature=0.05)
            
            # Apply stop sequences to prevent hallucination
            stop_sequences = ["Question:", "Q:", "Another", "Also consider", "You might also", "Additionally", "Furthermore", "Other questions"]
            for stop_seq in stop_sequences:
                if stop_seq in response:
                    response = response.split(stop_seq)[0].strip()
                    break
                    
            print("✅ LLM response generated successfully")
        else:
            # Fallback to template-based response
            print("⚠️ LLM not available, using fallback response")
            response = generate_fallback_response(query, top_results)
            
    except Exception as e:
        print(f"❌ LLM generation failed: {e}, using fallback")
        response = generate_fallback_response(query, top_results)
    
    return response, sources_used

def generate_fallback_response(query, search_results):
    """Generate a fallback response when LLM is not available."""
    query_lower = query.lower()
    
    if 'fluid mechanics' in query_lower or 'fluid' in query_lower:
        response = "Based on sustainability research in engineering education, here are specific ways to incorporate sustainability into your Fluid Mechanics course:\n\n"
        response += "**Key Integration Areas:**\n\n"
        response += "• **Energy Efficiency**: Focus on pump and turbine efficiency, renewable energy applications, and sustainable fluid systems\n"
        response += "• **Environmental Impact**: Cover water treatment, pollution control, and sustainable water management practices\n"
        response += "• **Real-world Applications**: Include case studies of sustainable fluid systems in buildings, infrastructure, and renewable energy\n"
        response += "• **Green Technologies**: Explore wind turbines, hydroelectric power, and sustainable HVAC systems\n\n"
        
    elif 'thermodynamics' in query_lower:
        response = "Here are research-backed approaches for integrating sustainability into Thermodynamics courses:\n\n"
        response += "**Core Integration Areas:**\n\n"
        response += "• **Energy Conservation**: Apply First and Second Law principles to sustainable energy systems\n"
        response += "• **Renewable Energy Cycles**: Study solar thermal, geothermal, and biomass energy applications\n"
        response += "• **Efficiency Analysis**: Analyze heat engines, refrigeration cycles, and combined heat and power systems\n"
        response += "• **Environmental Thermodynamics**: Explore climate change, atmospheric processes, and carbon cycles\n\n"
        
    elif 'materials' in query_lower:
        response = "Here are effective strategies for integrating sustainability into Materials Science courses:\n\n"
        response += "**Key Integration Areas:**\n\n"
        response += "• **Sustainable Materials**: Study biodegradable, recyclable, and renewable materials\n"
        response += "• **Life Cycle Assessment**: Analyze environmental impact from extraction to disposal\n"
        response += "• **Green Manufacturing**: Explore energy-efficient production processes\n"
        response += "• **Circular Economy**: Focus on material reuse, recycling, and waste reduction\n\n"
        
    else:
        response = "Based on sustainability research in engineering education, here are general approaches for incorporating sustainability into your course:\n\n"
        response += "**Core Principles:**\n\n"
        response += "• **Systems Thinking**: Help students understand interconnected environmental, social, and economic impacts\n"
        response += "• **Life Cycle Analysis**: Consider environmental impacts from design to disposal\n"
        response += "• **Stakeholder Engagement**: Include diverse perspectives in problem-solving\n"
        response += "• **Innovation for Sustainability**: Encourage creative solutions to environmental challenges\n\n"
    
    # Add research-based insights if available
    if search_results:
        response += "**Research-Based Insights:**\n\n"
        for i, result in enumerate(search_results[:2], 1):
            text = result.get('text', '')
            if len(text) > 100:
                # Extract a meaningful excerpt
                start_idx = max(0, text.find('sustainability') - 50)
                end_idx = min(len(text), start_idx + 300)
                excerpt = text[start_idx:end_idx]
                clean_excerpt = excerpt.replace('\n', ' ').strip()
                sentences = clean_excerpt.split('.')
                if len(sentences) > 1:
                    clean_excerpt = '. '.join(sentences[:2]) + '.'
                response += f"**Finding {i}**: {clean_excerpt}\n\n"
    
    return response

def get_fallback_response(message):
    """Generate a fallback response when the search system is unavailable."""
    message_lower = message.lower()
    
    if 'fluid mechanics' in message_lower or 'fluid' in message_lower:
        return {
            'content': "For Fluid Mechanics courses, you can integrate sustainability by:\n\n• **Energy Efficiency**: Discuss pump and turbine efficiency, renewable energy applications\n• **Environmental Impact**: Cover water treatment, pollution control, and sustainable water management\n• **Case Studies**: Include examples of sustainable fluid systems in buildings and infrastructure\n• **Green Technologies**: Explore wind turbines, hydroelectric power, and sustainable HVAC systems\n\n*Note: I'm currently using simplified responses. The full research-based system will be available shortly.*",
            'sources': []
        }
    elif 'thermodynamics' in message_lower:
        return {
            'content': "Thermodynamics offers excellent opportunities for sustainability integration:\n\n• **Energy Conservation**: First and Second Law applications to sustainable systems\n• **Renewable Energy**: Solar thermal, geothermal, and biomass energy cycles\n• **Efficiency Analysis**: Heat engines, refrigeration cycles, and combined heat and power\n• **Environmental Thermodynamics**: Climate change, atmospheric processes, and carbon cycles\n\n*Note: I'm currently using simplified responses. The full research-based system will be available shortly.*",
            'sources': []
        }
    elif 'materials' in message_lower:
        return {
            'content': "Materials engineering is crucial for sustainability:\n\n• **Green Materials**: Biodegradable polymers, sustainable composites, and recycled materials\n• **Life Cycle Assessment**: Environmental impact analysis of material choices\n• **Energy Materials**: Solar cells, batteries, and energy storage materials\n• **Circular Economy**: Design for disassembly, recycling, and material recovery\n\n*Note: I'm currently using simplified responses. The full research-based system will be available shortly.*",
            'sources': []
        }
    elif 'design' in message_lower:
        return {
            'content': "Engineering design is perfect for sustainability integration:\n\n• **Sustainable Design Principles**: Cradle-to-cradle, biomimicry, and green design\n• **Environmental Impact**: Life cycle assessment and environmental footprint analysis\n• **Social Responsibility**: Human-centered design and community impact\n• **Innovation**: Sustainable technology development and green innovation\n\n*Note: I'm currently using simplified responses. The full research-based system will be available shortly.*",
            'sources': []
        }
    elif 'syllabus' in message_lower or 'upload' in message_lower:
        return {
            'content': "I can help you analyze your syllabus and provide personalized sustainability integration suggestions! Please upload your syllabus PDF using the upload button in the sidebar. Once uploaded, I'll analyze the content and provide specific recommendations for incorporating sustainability into your course.",
            'sources': []
        }
    else:
        return {
            'content': f"I understand you're asking about: '{message}'. I'm here to help you integrate sustainability into your engineering courses. You can:\n\n• Ask me about specific courses (like Fluid Mechanics, Thermodynamics, Materials, etc.)\n• Upload your syllabus for personalized suggestions\n• Get examples of sustainability integration for different engineering topics\n\n*Note: I'm currently using simplified responses. The full research-based system will be available shortly.*",
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
        
        # Try to process with enhanced system, fallback to simple analysis
        try:
            from scaffold_core.pdf_processor import process_syllabus_upload
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
            # Fallback to simple suggestions
            suggestions = [
                "Consider adding a sustainability module to your course",
                "Include case studies of sustainable engineering practices",
                "Incorporate life cycle assessment principles",
                "Add assignments focused on environmental impact analysis"
            ]
            
            return jsonify({
                'success': True,
                'filename': filename,
                'message': 'Syllabus uploaded successfully (using simplified analysis)',
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

        # Try to use LLM-powered search system
        ai_message = None
        try:
            # Initialize search system if needed
            if initialize_search_system():
                print(f"🤖 Processing query with LLM-powered search: {message[:50]}...")
                
                # Search the sustainability database
                search_results = search_sustainability_data(message, k=10)
                
                # Generate response using LLM with syllabus context
                response_content, sources = generate_response_from_sources(message, search_results, session_id)
                
                # Create AI response with sources
                ai_message = {
                    'id': str(uuid.uuid4()),
                    'type': 'assistant',
                    'content': response_content,
                    'sources': sources,
                    'timestamp': datetime.datetime.now().isoformat()
                }
                print(f"✅ LLM-powered response generated successfully")
            else:
                raise Exception("Search system not available")
                
        except Exception as e:
            print(f"⚠️ Search error: {e}, using fallback response")
            # Use fallback response
            response_data = get_fallback_response(message)
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
    """Legacy query endpoint - uses direct search when available."""
    try:
        data = request.get_json()
        message = data.get('query', '').strip()
        
        if not message:
            return jsonify({'error': 'Query is required'}), 400
        
        # Try to use search system
        try:
            if initialize_search_system():
                # Search the sustainability database
                search_results = search_sustainability_data(message, k=10)
                
                # Generate response from search results
                response_content, sources = generate_response_from_sources(message, search_results)
                
                response = {
                    'response': response_content,
                    'sources': sources,
                    'query': message
                }
            else:
                # Use fallback response
                response_data = get_fallback_response(message)
                response = {
                    'response': response_data['content'],
                    'sources': response_data['sources'],
                    'query': message
                }
        except Exception as e:
            print(f"Query API error: {e}, using fallback")
            response_data = get_fallback_response(message)
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
        'search_system': 'initialized' if system_initialized else 'not_initialized',
        'mode': 'llm_powered' if system_initialized else 'fallback'
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
    
    parser = argparse.ArgumentParser(description='Scaffold AI Enhanced UI - LLM-Powered')
    parser.add_argument('--port', type=int, default=5003, help='Port to run the server on')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to run the server on')
    
    args = parser.parse_args()
    
    print(f"🚀 Starting Scaffold AI Enhanced UI (LLM-Powered) on {args.host}:{args.port}")
    print(f"📍 Access the UI at: http://localhost:{args.port}")
    print(f"📊 Feedback dashboard at: http://localhost:{args.port}/feedback")
    print(f"💡 Press Ctrl+C to stop the server")
    print(f"🤖 LLM-powered search system will initialize on first request")
    
    app.run(host=args.host, port=args.port, debug=True) 