# Scaffold AI CLI Quick Reference

## 🚀 Quick Start
```bash
# Single query with text output
python generate_query_results.py --query "Your question here"

# JSON output for automation
python generate_query_results.py --json --query "Your question here" > result.json

# Interactive mode
python generate_query_results.py
```

## 📋 Common Commands

### Query Generation
```bash
# Basic query
python generate_query_results.py --query "What is sustainability in engineering?"

# Save to specific file
python generate_query_results.py --query "Your query" > my_results.txt

# JSON output for processing
python generate_query_results.py --json --query "Your query" > data.json
```

### Model Benchmarking
```bash
# All models
python scaffold_core/benchmark_models.py

# Specific model types
python scaffold_core/benchmark_models.py --type embedding
python scaffold_core/benchmark_models.py --type llm
python scaffold_core/benchmark_models.py --type cross_encoder
```

### Web UI
```bash
# Start Flask web interface
python frontend/start_ui.py

# Access at: http://localhost:5000
```

## 🔧 API Endpoints (when UI running)

```bash
# Health check
curl http://localhost:5000/api/health

# Submit query
curl -X POST http://localhost:5000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Your question"}'
```

## 📁 Output Locations

- **Text files**: `query_outputs/` directory
- **Logs**: `logs/` directory  
- **Model cache**: `models/` directory
- **Vector data**: `vector_outputs/` directory

## ⚡ Performance Tips

```bash
# Use CPU only (if GPU issues)
export CUDA_VISIBLE_DEVICES=""

# Debug mode
python -u generate_query_results.py --query "Test" 2>&1 | tee debug.log

# Batch processing
for query in "query1" "query2" "query3"; do
  python generate_query_results.py --json --query "$query" > "results/${query}.json"
done
```

## 🛠️ Troubleshooting

```bash
# Check Python version
python --version

# Verify virtual environment
which python  # Should point to scaffold_env

# Check dependencies
pip list | grep -E "(transformers|torch|sentence-transformers)"

# Set HuggingFace token
export HUGGINGFACE_TOKEN="your_token_here"
```

## 📊 Example Queries

```bash
# Sustainability education
python generate_query_results.py --query "How do universities integrate sustainability into engineering curricula?"

# Climate change competencies  
python generate_query_results.py --query "What are the key competencies for climate resilience in engineering?"

# Environmental literacy
python generate_query_results.py --query "What methods are used to teach environmental literacy in civil engineering?"
```

## 🔄 Automation Examples

```bash
# Process multiple queries from file
while read query; do
  python generate_query_results.py --json --query "$query" > "results/${query// /_}.json"
done < queries.txt

# Extract just the response from JSON
python generate_query_results.py --json --query "Your query" | python -c "import sys, json; print(json.load(sys.stdin)['response'])"

# Get source count
python generate_query_results.py --json --query "Your query" | python -c "import sys, json; print(len(json.load(sys.stdin)['candidates']))"
``` 