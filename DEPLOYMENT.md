# Scaffold AI Deployment Guide

## Quick Start

### Option 1: Local Windows (Recommended for Professor)
1. Download the project folder
2. Double-click `run_app.bat`
3. Browser opens automatically at http://localhost:5002

### Option 2: Render (Free Cloud Hosting)
1. Push to GitHub
2. Connect to Render.com
3. Deploy using `render.yaml` configuration
4. Share the public URL

### Option 3: Docker
```bash
docker build -t scaffold-ai .
docker run -p 8000:8080 scaffold-ai
```

## Environment Variables

- `SC_LLM_KEY=distilgpt2` - Uses token-free model (default)
- `HUGGINGFACE_TOKEN=hf_xxx` - Required for advanced models
- `PORT=5002` - Server port

## File Structure for Deployment

Essential files to include:
```
scaffold_ai/
├── frontend/                 # Web interface
├── scaffold_core/           # Core AI logic
├── vector_outputs/          # Pre-built search index
│   ├── scaffold_index_1.faiss
│   └── scaffold_metadata_1.json
├── requirements.txt         # Python dependencies
├── wsgi.py                 # Production server entry
├── Dockerfile              # Container configuration
├── render.yaml             # Render deployment config
└── run_app.bat            # Windows launcher
```

## Features

- **Sustainability Education Assistant**: AI-powered research database
- **PDF Upload**: Analyze course syllabi for sustainability integration
- **Conversational Interface**: Chat-based interaction
- **Source Citations**: Links to original research papers
- **No Token Required**: Works with public models by default

## Technical Requirements

- Python 3.12+
- 4GB+ RAM (for vector search)
- 2GB+ disk space
- Internet connection (for model downloads)

## Troubleshooting

- **Port already in use**: Change `PORT` in `run_app.bat`
- **Model download fails**: Check internet connection
- **Memory issues**: Reduce batch sizes in vector processing
