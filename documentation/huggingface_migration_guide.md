# Hugging Face Integration Guide: Migrating from Ollama

**Date:** June 29, 2025  
**Status:** Production Ready  
**Compatibility:** Python 3.11+, Windows/Linux/macOS

## 📋 Overview

This guide explains how to use Hugging Face Transformers instead of Ollama for LLM functionality in the Scaffold AI project. The migration provides better model compatibility, easier deployment, and more reliable performance across different environments.

## 🔄 Migration Summary

| Aspect | Ollama | Hugging Face |
|--------|--------|--------------|
| **Installation** | External binary + server | Python package |
| **Model Access** | Local download + serve | Direct API access |
| **Configuration** | Server endpoints | Model identifiers |
| **Dependencies** | System-level | Python environment |
| **Compatibility** | Platform-specific | Cross-platform |
| **Performance** | Optimized serving | Flexible inference |

## 🚀 Quick Start

### 1. Install Dependencies

The required packages are already included in `requirements.txt`:

```bash
# Activate your virtual environment first
pip install transformers torch sentencepiece accelerate
```

### 2. Configure Hugging Face Token (Required for Gated Models)

For gated models like Mistral-7B-Instruct-v0.2, you'll need a Hugging Face token with proper permissions:

#### Step 1: Create a Hugging Face Account
1. Visit https://huggingface.co/join
2. Sign up with your email address
3. Verify your email address

#### Step 2: Create an Access Token
1. **Visit the tokens page:** https://huggingface.co/settings/tokens
2. **Click "New token"**
3. **Configure your token:**
   - **Name:** Give it a descriptive name (e.g., "Scaffold AI Project")
   - **Type:** Select "Fine-grained" (required for gated repositories)
   - **Permissions:** Check the boxes for:
     - ✅ **Read access to contents of all public gated repos you can access**
     - ✅ **Read access to contents of all repos under your namespace**
   - **Repositories:** Use the dropdown to select specific gated repositories:
     - Search for and select `mistralai/Mistral-7B-Instruct-v0.2`
     - Or select "All repositories" if you want broader access
4. **Click "Generate token"**
5. **Copy the token immediately** - you won't be able to see it again!

**Important:** You must create the token AFTER you've been granted access to the gated repository. If you create the token before getting access, it won't work.

#### Step 3: Request Access to Gated Repositories
For models like `mistralai/Mistral-7B-Instruct-v0.2`:

1. **Visit the model page:** https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2
2. **Look for the "Request access" button** (appears if the model is gated)
3. **Click "Request access"**
4. **Accept the license terms** (usually just requires clicking "Accept" on the license agreement)
5. **Wait for approval** (usually automatic or takes a few minutes to a few hours)
6. **Check your email** for approval notification (if required)

**Note:** Most gated repositories only require accepting the license terms, not filling out a detailed form. The process is usually straightforward.

#### Step 4: Set Environment Variable

```bash
# Windows (PowerShell)
$env:HUGGINGFACE_TOKEN = "hf_your_actual_token_here"
[Environment]::SetEnvironmentVariable("HUGGINGFACE_TOKEN", "hf_your_actual_token_here", "User")

# Linux/macOS
export HUGGINGFACE_TOKEN="hf_your_actual_token_here"
echo 'export HUGGINGFACE_TOKEN="hf_your_actual_token_here"' >> ~/.bashrc
```

#### Step 5: Alternative - Create a `.env` file

```bash
# In your project root directory
echo "HUGGINGFACE_TOKEN=hf_your_actual_token_here" > .env
```

#### Step 6: Verify Token Access

```bash
# Test token access (optional)
python -c "
from huggingface_hub import whoami
print('Token is valid:', whoami())
"
```

### 3. Update Configuration

The project is already configured to use Hugging Face. Current settings in `scaffold_core/config.py`:

```python
# LLM Configuration (Hugging Face)
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
LLM_MAX_LENGTH = 512
LLM_TEMPERATURE = 0.7
LLM_DO_SAMPLE = True
```

## 🔧 Detailed Configuration

### Model Selection

The project supports various Hugging Face models. Here are tested options:

#### ✅ Recommended Models

1. **Mistral-7B-Instruct-v0.2** (Current default)
   ```python
   LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
   ```
   - **Pros:** Official model, excellent performance, good tokenization
   - **Cons:** Requires access approval, 7B parameters (larger)
   - **Status:** ✅ Fully tested and working

2. **OpenHermes-2.5-Mistral-7B**
   ```python
   LLM_MODEL = "teknium/OpenHermes-2.5-Mistral-7B"
   ```
   - **Pros:** No gating, Mistral-based, good performance
   - **Cons:** Community model, 7B parameters
   - **Status:** ✅ Tested and working

3. **TinyLlama-1.1B-Chat-v1.0**
   ```python
   LLM_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
   ```
   - **Pros:** Small size (1.1B), fast inference, no gating
   - **Cons:** Lower performance than larger models
   - **Status:** ✅ Tested and working

#### ⚠️ Models with Issues

1. **Mistral-7B-Instruct-v0.3**
   ```python
   LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
   ```
   - **Issue:** Tokenizer compatibility problems with `PyPreTokenizerTypeWrapper`
   - **Status:** ❌ Not recommended

### Advanced Configuration Options

```python
# In scaffold_core/config.py

# Basic LLM settings
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
LLM_MAX_LENGTH = 512          # Maximum tokens to generate
LLM_TEMPERATURE = 0.7         # Creativity (0.0-1.0)
LLM_DO_SAMPLE = True          # Enable sampling
LLM_TOP_P = 0.9              # Nucleus sampling parameter
LLM_TOP_K = 50               # Top-k sampling parameter

# Performance settings
LLM_DEVICE = "auto"           # "auto", "cpu", "cuda"
LLM_TORCH_DTYPE = "auto"      # "auto", "float16", "float32"
LLM_USE_FAST_TOKENIZER = False # Fallback for compatibility
```

## 🛠️ Implementation Details

### LLM Manager Class

The `scaffold_core/llm.py` module provides the `LLMManager` class:

```python
from scaffold_core.llm import LLMManager

# Initialize
llm_manager = LLMManager()

# Generate text
response = llm_manager.generate_response("What is sustainability?")
print(response)
```

### Key Features

1. **Automatic Model Loading:** Downloads and caches models automatically
2. **Tokenizer Compatibility:** Handles both fast and slow tokenizers
3. **Chat Format Support:** Proper formatting for instruction-tuned models
4. **Error Handling:** Graceful fallbacks and informative error messages
5. **Memory Management:** Efficient model loading and cleanup

### Chat Format Examples

Different models use different chat formats:

#### Mistral Format
```python
# Input: "What is sustainability?"
# Formatted: "[INST] What is sustainability? [/INST]"
```

#### TinyLlama Format
```python
# Input: "What is sustainability?"
# Formatted: "<|user|>\nWhat is sustainability?\n<|assistant|>\n"
```

## 🧪 Testing and Validation

### Run System Tests

```bash
# Run comprehensive tests
python scaffold_core/scripts/run_tests.py

# Generate detailed report
python scaffold_core/scripts/generate_test_report.py
```

### Test Results

The system has been thoroughly tested with 100% success rate:

- **Model Loading:** ✅ All components load successfully
- **Embedding Generation:** ✅ Proper tensor shapes and normalization
- **Similarity Scoring:** ✅ Meaningful relevance scores
- **Cross-Encoder:** ✅ Accurate relevance ranking
- **LLM Integration:** ✅ Proper text generation

## 🔍 Troubleshooting

### Common Issues and Solutions

#### 1. Token Access Issues

**Problem:** `HTTPError: 401 Client Error: Unauthorized`

**Solutions:**
1. **Verify token is set correctly:**
   ```bash
   # Check if token is set
   echo $HUGGINGFACE_TOKEN  # Linux/macOS
   echo $env:HUGGINGFACE_TOKEN  # Windows PowerShell
   
   # Token should start with "hf_" and be about 37 characters long
   ```

2. **Create a new token:**
   - Visit https://huggingface.co/settings/tokens
   - Delete old token if needed
   - Create new token with "Read" permissions
   - Make sure to copy it immediately

3. **Test token validity:**
   ```bash
   python -c "from huggingface_hub import whoami; print(whoami())"
   ```

**Problem:** `OSError: You are trying to access a gated repo`

**Solutions:**
1. **Make sure you're logged in with your token**
2. **Request access to the specific model first**
3. **Wait for approval before trying to download**

#### 2. Model Access Denied

**Problem:** `GatedRepoError: Access to model mistralai/Mistral-7B-Instruct-v0.2 is restricted`

**Step-by-step solution:**
1. **Ensure you have a Hugging Face account and token** (see Step 2 above)
2. **Visit the model page:** https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2
3. **Look for "Request access" button:**
   - If you see it, click it
   - If you don't see it, you may already have access or need to log in
4. **Accept the license terms:**
   - Usually just requires clicking "Accept" on the license agreement
   - No detailed form to fill out for most models
   - Some models may ask for basic information like intended use
5. **Wait for approval:** Usually automatic or takes 15 minutes to 2 hours
6. **Check your email** for approval notification (if required)
7. **Try running the code again** after approval

**Important:** Most gated repositories only require accepting the license terms, not filling out detailed forms. The process is typically straightforward.

**Alternative - Use non-gated models:**
```python
# These models don't require special access:
LLM_MODEL = "teknium/OpenHermes-2.5-Mistral-7B"  # No gating required
# or
LLM_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Smaller, faster
```

#### 3. Token Configuration Issues

**Problem:** Token exists but still getting access denied errors

**Common causes and solutions:**
1. **Token created before getting repository access:**
   - Delete your old token
   - Request access to the gated repository first
   - Create a new token AFTER getting access

2. **Wrong token type:**
   - Make sure you selected "Fine-grained" not just "Read"
   - "Read" tokens don't work for gated repositories

3. **Missing permissions:**
   - Check "Read access to contents of all public gated repos you can access"
   - Make sure you selected the specific repository in the dropdown

4. **Repository not selected:**
   - In the token creation, use the dropdown to select `mistralai/Mistral-7B-Instruct-v0.2`
   - Or select "All repositories" for broader access

**Solution - Create a new token with correct settings:**
1. Go to https://huggingface.co/settings/tokens
2. Delete your old token
3. Create new "Fine-grained" token
4. Select the gated repository from dropdown
5. Check the required permissions

#### 4. Repository Access Verification

**Problem:** Not sure if you have access to a gated repository

**Solution - Check access status:**
```python
# Run this to check if you have access
python -c "
from huggingface_hub import repo_info
try:
    info = repo_info('mistralai/Mistral-7B-Instruct-v0.2')
    print('✅ Access granted! Model info:', info.modelId)
except Exception as e:
    print('❌ Access denied:', str(e))
"
```

#### 5. Tokenizer Compatibility Issues

**Problem:** `PyPreTokenizerTypeWrapper` errors

**Solution:** The system automatically falls back to slow tokenizers:
```python
# This is handled automatically in LLMManager
tokenizer = AutoTokenizer.from_pretrained(
    model_name, 
    use_fast=False  # Fallback for compatibility
)
```

#### 6. Memory Issues

**Problem:** `CUDA out of memory` or system slowdown

**Solutions:**
1. **Use smaller model:**
   ```python
   LLM_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
   ```

2. **Reduce max length:**
   ```python
   LLM_MAX_LENGTH = 256  # Reduce from 512
   ```

3. **Use CPU inference:**
   ```python
   LLM_DEVICE = "cpu"
   ```

#### 7. SentencePiece Dependency

**Problem:** `No module named 'sentencepiece'`

**Solution:**
```bash
pip install sentencepiece
```

### Performance Optimization

#### For CPU-Only Systems
```python
# Optimize for CPU inference
LLM_DEVICE = "cpu"
LLM_TORCH_DTYPE = "float32"
LLM_MAX_LENGTH = 256
```

#### For GPU Systems
```python
# Optimize for GPU inference
LLM_DEVICE = "cuda"
LLM_TORCH_DTYPE = "float16"
LLM_MAX_LENGTH = 512
```

## 📊 Comparison: Ollama vs Hugging Face

### Advantages of Hugging Face

✅ **Easier Setup:** No external server required  
✅ **Better Integration:** Direct Python API  
✅ **Model Flexibility:** Easy model switching  
✅ **Cross-Platform:** Works on all operating systems  
✅ **Version Control:** Specific model versions  
✅ **Debugging:** Better error messages and logging  
✅ **Testing:** Easier to unit test and validate  

### When to Use Ollama

- **Production Serving:** High-throughput inference server
- **Resource Optimization:** Optimized for specific hardware
- **Model Management:** Centralized model serving
- **API Consistency:** Standardized REST API

### When to Use Hugging Face

- **Development:** Rapid prototyping and testing
- **Research:** Model experimentation and comparison
- **Integration:** Embedded in Python applications
- **Flexibility:** Custom inference pipelines

## 🔄 Migration Steps from Ollama

If you're migrating from an existing Ollama setup:

### 1. Remove Ollama Dependencies

```python
# OLD: Ollama configuration
# OLLAMA_BASE_URL = "http://localhost:11434"
# OLLAMA_MODEL = "llama2"

# NEW: Hugging Face configuration
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
```

### 2. Update Import Statements

```python
# OLD: Ollama client
# from ollama import Client

# NEW: Hugging Face LLM manager
from scaffold_core.llm import LLMManager
```

### 3. Update Code Usage

```python
# OLD: Ollama usage
# client = Client(host='http://localhost:11434')
# response = client.chat(model='llama2', messages=[...])

# NEW: Hugging Face usage
llm_manager = LLMManager()
response = llm_manager.generate_response("Your query here")
```

### 4. Test the Migration

```bash
# Run tests to verify everything works
python scaffold_core/scripts/run_tests.py
```

## 📚 Additional Resources

- **Hugging Face Documentation:** https://huggingface.co/docs/transformers
- **Model Hub:** https://huggingface.co/models
- **Tokenizers Guide:** https://huggingface.co/docs/tokenizers
- **Transformers Installation:** https://huggingface.co/docs/transformers/installation

## 🔮 Future Enhancements

Potential improvements for the Hugging Face integration:

1. **Model Caching:** Implement local model caching strategies
2. **Batch Processing:** Support for batch inference
3. **Streaming:** Real-time response streaming
4. **Model Quantization:** 4-bit and 8-bit quantization support
5. **Multi-GPU:** Support for model parallelism
6. **Custom Models:** Fine-tuning and custom model support

---

**Need Help?** Check the test reports in `documentation/query_system_test_report.md` or run the diagnostic tests with `python scaffold_core/scripts/run_tests.py`. 