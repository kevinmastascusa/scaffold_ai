# Scaffold AI CLI API Instructions

This document provides comprehensive instructions for using the Scaffold AI system via command-line interface (CLI).

## Table of Contents
1. [Quick Start](#quick-start)
2. [Available Commands](#available-commands)
3. [Query Generation](#query-generation)
4. [Model Benchmarking](#model-benchmarking)
5. [UI Management](#ui-management)
6. [Examples](#examples)
7. [Troubleshooting](#troubleshooting)

## Quick Start

### Prerequisites
- Python 3.12.10
- Virtual environment activated
- All dependencies installed (`pip install -r requirements.txt`)
- HuggingFace token set in environment variables

### Basic Usage
```bash
# Generate a query result and save to file
python generate_query_results.py --query "What is sustainability in engineering?"

# Generate JSON output for piping to other tools
python generate_query_results.py --json --query "What is sustainability in engineering?" > result.json
```

## Available Commands

### 1. Query Generation Script
**File**: `generate_query_results.py`

#### Options:
- `--query TEXT`: Specify a query to run
- `--json`: Output raw JSON instead of formatted text file
- No arguments: Interactive mode with menu

#### Examples:
```bash
# Single query with text output
python generate_query_results.py --query "How do engineering programs address sustainability?"

# JSON output for programmatic use
python generate_query_results.py --json --query "What are climate change competencies?" > output.json

# Interactive mode
python generate_query_results.py
```

### 2. Model Benchmarking Script
**File**: `scaffold_core/benchmark_models.py`

#### Options:
- `--type {embedding,cross_encoder,llm,all}`: Type of model to benchmark
- Default: `all` (benchmarks all model types)

#### Examples:
```bash
# Benchmark all models
python scaffold_core/benchmark_models.py

# Benchmark only embedding models
python scaffold_core/benchmark_models.py --type embedding

# Benchmark only LLM models
python scaffold_core/benchmark_models.py --type llm
```

### 3. UI Management Scripts
**Files**: 
- `frontend/start_ui.py` - Start the Flask web UI
- `frontend/app.py` - Direct Flask app execution

#### Examples:
```bash
# Start the web UI
python frontend/start_ui.py

# Direct Flask execution
python frontend/app.py
```

## Query Generation

### Interactive Mode
When running without arguments, the script provides an interactive menu:

```bash
python generate_query_results.py
```

**Output:**
```
Scaffold AI Query Results Generator
==================================================

Available example queries:
1. What is sustainability in engineering education?
2. How are climate change concepts integrated into engineering curricula?
3. What are the key competencies for environmental literacy in engineering?
4. How do engineering programs address sustainability challenges?
5. What methods are used to teach sustainability in civil engineering?

Options:
0. Run all example queries
C. Enter custom query

Enter your choice (0-5 or C):
```

### Programmatic Usage

#### Single Query with Text Output
```bash
python generate_query_results.py --query "What is sustainability in engineering?"
```

**Output**: Creates a formatted text file in `query_outputs/` directory

#### JSON Output for Automation
```bash
python generate_query_results.py --json --query "What is sustainability in engineering?" > result.json
```

**Output**: Raw JSON to stdout, suitable for:
- Piping to other tools
- Processing with scripts
- API integration

### Output Formats

#### Text File Format
```
================================================================================
SCAFFOLD AI QUERY RESULTS
================================================================================
Query: What is sustainability in engineering?
Timestamp: 2025-07-15T17:17:19

SEARCH STATISTICS
----------------------------------------
Initial candidates found: 49
After reranking: 49
After filtering: 49
Final candidates used: 3

AI RESPONSE
----------------------------------------
[Sustainability response text...]

SOURCES USED
----------------------------------------
Source 1:
  Score: 0.7228
  Name: sustainability incorporation in courses...
  ID: f33dafd0
  Path: C:\Users\dlaev\GitHub\scaffold_ai\data\...
  Preview: [Text preview...]
```

#### JSON Format
```json
{
  "query": "What is sustainability in engineering?",
  "response": "[AI generated response]",
  "candidates": [
    {
      "chunk_id": 3839,
      "score": 0.7228298017771702,
      "text": "[Source text]",
      "source": {
        "id": "f33dafd0",
        "name": "[Document name]",
        "raw_path": "[File path]"
      },
      "search_type": "keyword",
      "cross_score": 4.243017673492432,
      "contextual_score": 5
    }
  ],
  "search_stats": {
    "initial_candidates": 49,
    "reranked_candidates": 49,
    "filtered_candidates": 49,
    "final_candidates": 3
  }
}
```

## Model Benchmarking

### Benchmark All Models
```bash
python scaffold_core/benchmark_models.py
```

**Output:**
```
--- Starting EMBEDDING MODEL Benchmarks ---
all-MiniLM-L6-v2                                    |   2.34s |    156.2 MB | Fast, general-purpose embedding model
all-mpnet-base-v2                                   |   4.12s |    438.7 MB | High-quality, slower embedding model

--- Starting CROSS-ENCODER MODEL Benchmarks ---
cross-encoder/ms-marco-MiniLM-L-6-v2                |   1.89s |    156.2 MB | Fast reranking model for search

--- Starting LLM MODEL Benchmarks ---
TinyLlama/TinyLlama-1.1B-Chat-v1.0                  |   8.45s |   2048.1 MB | Fast, lightweight chat model
Output: Sustainability in engineering involves...
```

### Benchmark Specific Model Types
```bash
# Only embedding models
python scaffold_core/benchmark_models.py --type embedding

# Only LLM models  
python scaffold_core/benchmark_models.py --type llm

# Only cross-encoder models
python scaffold_core/benchmark_models.py --type cross_encoder
```

## UI Management

### Start Web UI
```bash
python frontend/start_ui.py
```

**Features:**
- Web interface at `http://localhost:5000`
- Feedback dashboard at `http://localhost:5000/feedback`
- REST API endpoints:
  - `GET /api/health` - Health check
  - `POST /api/query` - Submit queries
  - `GET /api/feedback` - View feedback

### API Endpoints (when UI is running)

#### Health Check
```bash
curl http://localhost:5000/api/health
```

#### Submit Query
```bash
curl -X POST http://localhost:5000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is sustainability in engineering?"}'
```

## Examples

### Example 1: Generate Research Report
```bash
# Generate a comprehensive report on sustainability education
python generate_query_results.py --query "How do universities integrate sustainability into engineering curricula?" > sustainability_report.txt
```

### Example 2: Batch Processing
```bash
# Create a script to process multiple queries
cat queries.txt | while read query; do
  python generate_query_results.py --json --query "$query" > "results/${query// /_}.json"
done
```

### Example 3: API Integration
```bash
# Start the UI
python frontend/start_ui.py &

# Wait for startup, then query
sleep 10
curl -X POST http://localhost:5000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are climate change competencies?"}' \
  | python -m json.tool > api_result.json
```

### Example 4: Performance Testing
```bash
# Benchmark models before deployment
python scaffold_core/benchmark_models.py > benchmark_results.txt

# Test query performance
time python generate_query_results.py --json --query "Test query" > /dev/null
```

## Troubleshooting

### Common Issues

#### 1. Model Loading Takes Time
**Symptom**: Script appears to hang during first run
**Solution**: This is normal for first-time model downloads. Wait 1-2 minutes.

#### 2. Memory Issues
**Symptom**: Out of memory errors
**Solution**: 
```bash
# Use CPU instead of GPU
export CUDA_VISIBLE_DEVICES=""
python generate_query_results.py --query "Your query"
```

#### 3. Token Limit Errors
**Symptom**: "Token limit exceeded" errors
**Solution**: The system automatically truncates long contexts. If issues persist, check `scaffold_core/config.py` for `TOP_K_FINAL` setting.

#### 4. Import Errors
**Symptom**: Module not found errors
**Solution**: Ensure virtual environment is activated and dependencies are installed:
```bash
# Activate virtual environment
source scaffold_env/bin/activate  # Linux/Mac
# or
scaffold_env\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

#### 5. HuggingFace Token Issues
**Symptom**: Authentication errors
**Solution**: Set your HuggingFace token:
```bash
export HUGGINGFACE_TOKEN="your_token_here"
```

### Debug Mode
For troubleshooting, run with debug output:
```bash
# Enable debug logging
export PYTHONPATH=.
python -u generate_query_results.py --query "Test query" 2>&1 | tee debug.log
```

### Performance Optimization
```bash
# Use smaller models for faster inference
# Edit scaffold_core/config.py to use:
# - TinyLlama instead of larger models
# - all-MiniLM-L6-v2 for embeddings
# - Reduce TOP_K_FINAL for shorter prompts
```

## Advanced Usage

### Custom Configuration
Edit `scaffold_core/config.py` to customize:
- Model selection
- Search parameters
- Token limits
- Device settings

### Integration with Other Tools
```bash
# Process JSON output with jq (if available)
python generate_query_results.py --json --query "Query" | jq '.response'

# Extract sources only
python generate_query_results.py --json --query "Query" | jq '.sources[].source.name'

# Count candidates
python generate_query_results.py --json --query "Query" | jq '.search_stats.final_candidates'
```

### Automation Scripts
Create batch processing scripts:
```bash
#!/bin/bash
# batch_process.sh
queries=(
  "What is sustainability in engineering?"
  "How do universities teach climate change?"
  "What are environmental competencies?"
)

for query in "${queries[@]}"; do
  echo "Processing: $query"
  python generate_query_results.py --query "$query"
done
```

## Support

For additional help:
1. Check the logs in the `logs/` directory
2. Review the `PROGRESS.md` file for recent updates
3. Examine the `documentation/` folder for detailed guides
4. Check the `scaffold_core/config.py` file for configuration options 