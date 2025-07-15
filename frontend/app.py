#!/usr/bin/env python3
"""
Scaffold AI Pilot UI
A Flask web application for testing and getting feedback on the enhanced query
system.
"""

import datetime
import json
import os
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from scaffold_core.vector.enhanced_query import (
    query_enhanced, enhanced_query_system
)

# Initialize Flask app
app = Flask(__name__, template_folder='templates')
CORS(app)

# Configuration
app.config['SECRET_KEY'] = 'scaffold-ai-pilot-ui-key'
app.config['DEBUG'] = True

# Create directories for storing feedback and logs
FEEDBACK_DIR = Path("ui_feedback")
FEEDBACK_DIR.mkdir(exist_ok=True)

# Pre-initialize the enhanced query system
print("🔄 Pre-initializing enhanced query system...")
try:
    enhanced_query_system.initialize()
    print("✅ Enhanced query system initialized successfully")
except Exception as e:
    print(f"❌ Failed to initialize enhanced query system: {e}")
    print("⚠️  The system will initialize on first request (may cause delay)")


@app.route('/')
def index():
    """Main page of the pilot UI."""
    return render_template('index.html')


@app.route('/api/query', methods=['POST'])
def api_query():
    """API endpoint for processing queries."""
    try:
        data = request.get_json()
        query = data.get('query', '').strip()

        if not query:
            return jsonify({'error': 'Query is required'}), 400

        # Process the query using the enhanced system
        result = query_enhanced(query)

        # Format the response for the UI
        response = {
            'query': query,
            'response': result['response'],
            'candidates_found': result['search_stats']['final_candidates'],
            'search_stats': result['search_stats'],
            'sources': [
                {
                    'source': candidate.get('source', 'Unknown'),
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

        # Validate required fields
        required_fields = ['query', 'response', 'rating']
        for field in required_fields:
            if field not in data:
                return jsonify(
                    {'error': f'Missing required field: {field}'}
                ), 400

        # Create feedback entry
        feedback = {
            'timestamp': datetime.datetime.now().isoformat(),
            'query': data['query'],
            'response': data['response'],
            'rating': data['rating'],
            'comments': data.get('comments', ''),
            'helpful_aspects': data.get('helpful_aspects', []),
            'improvement_suggestions': data.get(
                'improvement_suggestions', ''
            ),
            'user_id': data.get('user_id', 'anonymous')
        }

        # Save feedback to file
        feedback_file = FEEDBACK_DIR / (
            f"feedback_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            ".json"
        )
        with open(feedback_file, 'w', encoding='utf-8') as f:
            json.dump(feedback, f, ensure_ascii=False, indent=2)

        return jsonify({'message': 'Feedback saved successfully'})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/health')
def api_health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.datetime.now().isoformat(),
        'version': '1.0.0'
    })


@app.route('/feedback')
def feedback_page():
    """Feedback summary page for administrators."""
    try:
        feedback_files = list(FEEDBACK_DIR.glob("feedback_*.json"))
        feedback_data = []

        for file in sorted(feedback_files, reverse=True)[:50]:
            with open(file, 'r', encoding='utf-8') as f:
                feedback_data.append(json.load(f))

        return render_template('feedback.html', feedback_data=feedback_data)

    except Exception as e:
        return render_template('feedback.html', error=str(e))


@app.errorhandler(404)
def not_found(error):
    """Handler for 404 errors."""
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_error(error):
    """Handler for 500 errors."""
    return render_template('500.html'), 500


if __name__ == '__main__':
    print("🚀 Starting Scaffold AI Pilot UI...")
    print("📍 Access the UI at: http://localhost:5000")
    print(
        "🔧 API endpoints available at: "
        "/api/query, /api/feedback, /api/health"
    )
    print("📊 Feedback dashboard at: http://localhost:5000/feedback")
    print("\n💡 Press Ctrl+C to stop the server")

    app.run(host='0.0.0.0', port=5000, debug=True) 