# Local Setup Guide

This guide explains how to set up the local components required for the Scaffold AI pipeline, specifically the Ollama LLM endpoint.

## Prerequisites

- Python 3.8+ installed
- Git repository cloned
- Basic familiarity with command line tools

## Setting Up Ollama

### 1. Install Ollama

**Windows:**
```bash
# Download from https://ollama.ai/download
# Or use winget
winget install Ollama.Ollama
```

**macOS:**
```bash
# Download from https://ollama.ai/download
# Or use Homebrew
brew install ollama
```

**Linux:**
```bash
# Download from https://ollama.ai/download
# Or use curl
curl -fsSL https://ollama.ai/install.sh | sh
```

### 2. Start Ollama Service

```bash
# Start the Ollama service
ollama serve
```

The service will start on the default local endpoint. Keep this terminal window open while using the application.

### 3. Pull the Mistral Model

```bash
# Pull the Mistral model (this may take several minutes)
ollama pull mistral
```

### 4. Verify Installation

```bash
# Test that Ollama is working
ollama list
```

You should see the `mistral` model listed.

## Configuration

### Environment Variables (Recommended)

Create a `.env` file in your project root:

```bash
# .env file
OLLAMA_ENDPOINT=http://localhost:11434/v1/chat/completions
OLLAMA_MODEL=mistral
```

### Alternative: Direct Configuration

If you prefer to configure directly in code, you can modify the configuration in `scaffold_core/config.py`:

```python
# Only for local development - do not commit these changes
LLM_ENDPOINT = "http://localhost:11434/v1/chat/completions"
```

**⚠️ Security Note:** Never commit localhost endpoints to version control. Use environment variables instead.

## Testing the Setup

### 1. Test Ollama Connection

```bash
# Test the API endpoint
curl -X POST http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistral",
    "messages": [{"role": "user", "content": "Hello, world!"}]
  }'
```

### 2. Test with Python

```python
import requests
import json

url = "http://localhost:11434/v1/chat/completions"
data = {
    "model": "mistral",
    "messages": [{"role": "user", "content": "Hello, world!"}]
}

response = requests.post(url, json=data)
print(response.json())
```

## Troubleshooting

### Common Issues

1. **Ollama service not running**
   - Make sure `ollama serve` is running in a terminal
   - Check if the service is running: `ollama list`

2. **Model not found**
   - Pull the model: `ollama pull mistral`
   - Check available models: `ollama list`

3. **Connection refused**
   - Verify Ollama is running on the correct port
   - Check firewall settings
   - Ensure no other service is using the port

4. **Permission errors**
   - Run with appropriate permissions
   - Check file/directory permissions

### Port Configuration

By default, Ollama runs on port 11434. If you need to change this:

1. Set the `OLLAMA_HOST` environment variable:
   ```bash
   export OLLAMA_HOST=0.0.0.0:8080  # Example: different port
   ```

2. Update your endpoint configuration accordingly:
   ```bash
   OLLAMA_ENDPOINT=http://localhost:8080/v1/chat/completions
   ```

## Security Considerations

### Local Development Only

- The localhost endpoint should only be used for local development
- Never expose this endpoint to the internet
- Use environment variables for configuration
- Don't commit endpoint URLs to version control

### Production Deployment

For production use, consider:
- Using a cloud-based LLM service
- Implementing proper authentication
- Using secure API gateways
- Setting up monitoring and logging

## Next Steps

Once Ollama is set up and running:

1. Install Python dependencies: `pip install -r requirements.txt`
2. Run the text processing pipeline
3. Test the complete workflow with a sample query

## Support

If you encounter issues:

1. Check the [Ollama documentation](https://ollama.ai/docs)
2. Review the troubleshooting section above
3. Check the project's issue tracker
4. Ensure all prerequisites are met

---

**Remember:** Keep your localhost endpoint configuration secure and never expose it in public documentation or version control. 