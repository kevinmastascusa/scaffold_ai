# 🌱 Scaffold AI: Curriculum Recommendation Tool for Sustainability and Climate Resilience

**Collaborators:** Kevin Mastascusa, Joseph Di Stefano

**Date:** 6/26/2025 | **Last Updated:** 6/29/2025

## 🌍 Project Overview

This project involves developing a specialized large language model (LLM)-based tool to assist educators in integrating sustainability and climate resilience topics into academic programs. The tool leverages state-of-the-art AI techniques to recommend high-quality, literature-backed educational materials, case studies, and project ideas.

## 🚀 Getting Started

### Prerequisites

- Python 3.11 or higher (recommended: Python 3.11 for best compatibility)
- Git
- 16GB+ RAM recommended for optimal performance
- NVIDIA GPU recommended but not required
- Windows: Microsoft Visual C++ Build Tools (for some package installations)
- Linux: python3-dev and build-essential packages

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/kevinmastascusa/scaffold_ai.git
   cd scaffold_ai
   ```

2. **Create and activate virtual environment:**
   ```bash
   # Windows (PowerShell)
   Remove-Item -Path scaffold_env -Recurse -Force -ErrorAction SilentlyContinue
   python -m venv scaffold_env
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
   .\scaffold_env\Scripts\activate

   # macOS/Linux
   rm -rf scaffold_env
   python -m venv scaffold_env
   source scaffold_env/bin/activate
   ```

   If you get a permission error, try:
   - Running PowerShell as administrator (Windows)
   - Using `sudo` for the commands (Linux/macOS)
   - Checking if the directory is being used by another process

3. **Install dependencies:**
   ```bash
   # Upgrade pip first
   python -m pip install --upgrade pip
   # Install requirements (this may take 5-10 minutes)
   pip install -r requirements.txt
   ```

   Note: 
   - Some packages are quite large (torch, transformers, etc.) and may take a while to download
   - If you encounter issues:
     - Windows: Install Visual C++ Build Tools
     - Linux: Run `sudo apt-get install python3-dev build-essential`

4. **Run the setup script:**
   ```bash
   python setup.py
   ```

   This will:
   - Create necessary directories if they don't exist (data/, outputs/, vector_outputs/, math_outputs/)
   - Validate the workspace structure
   - Check for existing PDF files
   - The script will work even if directories already exist

5. **Configure Hugging Face (optional):**
   - For some models (like Llama 2), get your token from https://huggingface.co/settings/tokens
   - Create a `.env` file:
     ```bash
     HUGGINGFACE_TOKEN=your_token_here
     ```

6. **Add your PDF documents:**
   ```bash
   # Create data directory if it doesn't exist (skip if it exists)
   mkdir -p data
   ```
   Place your PDF files in the `data/` directory. The system will automatically process all PDF files found in this directory and its subdirectories.

### Troubleshooting

If you encounter issues during installation:

1. **Virtual Environment Issues:**
   - Delete the existing `scaffold_env` directory and try creating it again
   - Ensure you have write permissions in the current directory
   - Try creating the virtual environment in a different location

2. **Package Installation Errors:**
   - Make sure you're using Python 3.11 (some packages might not work with 3.12+)
   - Install required system dependencies (Visual C++ Build Tools on Windows, build-essential on Linux)
   - If a package fails to install, try installing it separately with `pip install package-name`

3. **Setup Script Errors:**
   - Ensure all directories are writable
   - Check if any files are locked by other processes
   - Make sure you're in the correct directory when running the script

### Quick Start

1. **Process your documents:**
   ```bash
   # Extract and chunk PDF documents
   python scaffold_core/scripts/chunk/ChunkTest.py
   
   # Create vector embeddings
   python scaffold_core/vector/main.py
   ```

2. **Run analysis scripts:**
   ```bash
   # Generate Unicode analysis report
   python scaffold_core/scripts/tests/generate_unicode_report.py
   
   # Compare different extraction methods
   python scaffold_core/scripts/tests/compare_extractions.py
   
   # Programmatically analyze and fix combined words
   python scaffold_core/scripts/utils/postprocess_combined_words.py
   python scaffold_core/scripts/generate_combined_words_report.py
   ```
   - See `outputs/combined_words_analysis_report.txt` for a detailed summary of remaining combined words (now mostly legitimate technical/academic terms).

3. **Test the query system (Added 6/29/2025):**
   ```bash
   # Run comprehensive system tests
   python scaffold_core/scripts/run_tests.py
   
   # Generate detailed test report
   python scaffold_core/scripts/generate_test_report.py
   ```
   - See `documentation/query_system_test_report.md` for comprehensive system analysis and test results.

### Project Structure

```
scaffold_ai/
├── data/                    # Place your PDF documents here
├── outputs/                 # Chunked text extracts
├── vector_outputs/          # Vector embeddings and indexes
├── math_outputs/            # Math-aware processing results
├── scaffold_core/           # Core processing modules
│   ├── config.py           # Central configuration
│   ├── llm.py            # Hugging Face LLM integration
│   ├── scripts/            # Processing scripts
│   └── vector/             # Vector processing
├── setup.py                # Setup script
└── requirements.txt        # Python dependencies
```

### Configuration

All paths and settings are centrally managed in `scaffold_core/config.py`. The configuration automatically adapts to your workspace location, making the project portable for anyone who clones the repository.

### LLM Integration

The project uses Hugging Face's Transformers library with the following features:
- Mistral-7B-Instruct model by default
- Automatic GPU acceleration when available
- Mixed precision for better memory efficiency
- Easy model switching and parameter tuning

For detailed setup and configuration options, see the [Local Setup Guide](documentation/local_setup_guide.md).

## 🎯 Goals and Objectives

The primary goal is to create a user-friendly, accurate, and literature-grounded AI tool capable of:

* 📚 Suggesting relevant and up-to-date curriculum content.
* 🔍 Ensuring transparency by referencing scholarly sources for every recommendation.
* 🧩 Facilitating easy integration into existing courses, supporting targeted learning outcomes.

## 🛠️ Combined Words Issue: Solved

- **Combined words (e.g., "environmentalsustainability") are now programmatically detected and fixed.**
- **Automated post-processing and reporting scripts:**
  - `scaffold_core/scripts/utils/postprocess_combined_words.py` (fixes combined words)
  - `scaffold_core/scripts/generate_combined_words_report.py` (generates detailed report)
- **Final analysis:**
  - No camelCase or PascalCase issues remain.
  - Remaining combined words are legitimate technical/academic terms.
  - See `outputs/combined_words_analysis_report.txt` for details.

## 🛠️ Proposed System Architecture

The system will include three key components:

* **Retrieval-Augmented Generation (RAG) Framework**
* **Vector Embeddings:** Pre-process and embed key sustainability and resilience literature into a vector database (e.g., FAISS, Pinecone).
* **Document Retrieval:** Efficiently search and retrieve relevant sections from scholarly sources based on embedded user queries.

## 🤖 Large Language Model (LLM)

**Current Implementation (6/29/2025):**
* **Mistral-7B-Instruct-v0.2** - Successfully integrated and tested
  - Official Mistral AI model with 7B parameters
  - Hugging Face Transformers integration with proper tokenization
  - CPU-based inference with mixed precision support
  - Comprehensive testing shows 100% success rate

**Previously Considered Models:**
* **Llama 3 (Meta):** meta-llama/Llama-3.1-8B-Instruct, Llama-3.2-1B
* **Other Mistral Variants:** Mistral-7B-v0.1, Mistral-7B-Instruct-v0.3
* **Phi-3 Mini (Microsoft):** Phi-3.5-mini-instruct, Phi-3-mini-4k-instruct

## 🔗 Citation Tracking and Transparency

* 🔗 Direct linking between generated content and original sources.
* 🖥️ Interactive UI to show how each recommendation is grounded in literature.

## 🔄 Technical Workflow

1. 📥 **Corpus Collection:** Curate scholarly papers, reports, and policy documents.
2. 🗃️ **Data Preprocessing:** Clean, segment, and prepare documents.
3. 🤖 **Embedding and Storage:** Embed corpus data and store in a vector database.
4. ⚙️ **Inference Engine:** Retrieve and use embeddings to augment LLM output.
5. 📝 **Citation Layer:** Annotate outputs with clear citation links.

## 📅 Project Timeline Overview

The project follows a structured timeline with week-by-week development phases. Key phases include:

* 🏗️ Setting up the preprocessing pipeline and repository structure
* 🤖 Embedding the curated document corpus and validating retrieval quality
* 🧪 Integrating the LLM and developing the initial prototype
* 🎨 Building and refining the user interface
* 🧾 Implementing citation tracking and performing usability testing
* 🧑‍🏫 Engaging stakeholders for feedback and refining the final product

Optional enhancements may include a real-time feedback loop in the UI and tag-based filtering of recommendations.

## 📈 Evaluation Overview

The system will be evaluated based on its ability to:

* 🤖 Retrieve relevant and accurate curriculum materials
* 🔍 Generate transparent, literature-backed recommendations
* ⚡ Provide a responsive and accessible user experience
* 👥 Satisfy stakeholders through iterative testing and feedback

Evaluation will include both qualitative feedback from faculty and technical performance benchmarks such as system responsiveness, citation traceability, and usability outcomes.

## ✅ Expected Outcomes

* 🛠️ A functioning prototype generating cited curriculum recommendations.
* 🖥️ Intuitive UI ready for pilot use.
* 📄 Comprehensive documentation for future development.

## 🧾 TODO Section

### 🔥 High Priority (Current Sprint)

1. **Semantic Search and Retrieval** ✅ **COMPLETED (6/29/2025)**
   * ✅ Created query interface for the completed FAISS index
   * ✅ Implemented semantic search functionality using vector embeddings
   * ✅ Added comprehensive query performance testing and optimization
   * ✅ Successfully integrated Mistral-7B-Instruct-v0.2 LLM
   * ✅ All system components tested with 100% success rate

2. **Citation Layer Implementation** 🆕 **NEW PRIORITY**
   * Implement automatic citation extraction and source linking
   * Add citation formatting (APA, MLA, Chicago) and validation
   * Display citations in LLM responses with proper attribution

3. **Advanced Query Testing** 🆕 **NEW PRIORITY**
   * Create comprehensive sustainability query test suite
   * Add query result quality assessment metrics
   * Develop A/B testing framework for retrieval strategies

4. **Build User Interface for Knowledge Base**
   * Develop web interface for querying the vectorized knowledge base
   * Create intuitive search interface with result display
   * Add query result visualization and export functionality
   * Integrate citation display and source linking in UI

### 🔧 Medium Priority

3. **Enhance PDF Extraction and Chunking**
   * Integrate full Unicode analysis into page-based chunks
   * Complete math-aware chunking improvements in `ChunkTest_Math.py`
   * Re-integrate math-aware chunking with vector pipeline

4. **Advanced Search Features**
   * Implement filters and faceted search capabilities
   * Add relevance scoring and ranking improvements
   * Create advanced citation tracking and source linking
   * Implement citation-based result ranking and credibility scoring

5. **Testing and Validation** ✅ **LARGELY COMPLETED (6/29/2025)**
   * ✅ Implemented comprehensive automated testing system
   * ✅ Added full test suite for vector operations and LLM integration
   * ✅ Created performance benchmarks and validation metrics
   * ✅ Generated detailed test reports with system specifications
   * 🔄 **Remaining:** Configuration system validation tests

### 📈 Future Enhancements

6. **Citation and Source Management**
   * Citation network analysis and bibliography generation
   * Citation impact scoring and credibility metrics
   * Citation export to reference management tools

7. **Advanced Query Analytics**
   * Query intent classification and auto-completion
   * Query result personalization and difficulty assessment
   * Multi-modal query support (text + images + documents)

8. **System Optimization**
   * Add support for incremental updates to the vector database
   * Implement progress tracking for long-running processes
   * Add configuration profiles for different deployment scenarios

9. **Documentation and Examples**
   * Add vector query examples in the README
   * Create user guides for different use cases
   * Develop API documentation for programmatic access

10. **Optional Enhancements**
    * Real-time feedback loop in the UI
    * Tag-based filtering of recommendations
    * Advanced analytics and usage reporting

## 📅 Week 1 Tasks

1. **Define Preprocessing Methodology** 🔄 **LARGELY COMPLETE**
   * ✅ Established detailed document preprocessing methodology, including chunking size, format, and metadata extraction
   * ✅ Implemented page-based chunking (one complete page per chunk)
   * ✅ Created comprehensive Unicode and text analysis pipeline
   * ✅ Processed 273 PDF documents successfully with 4,859 chunks generated
   * 🔄 **Still in progress:** Math-aware chunking improvements and full Unicode integration
   * 📝 **Note:** Methodology is largely defined but refinement may be needed based on downstream LLM integration and user feedback

2. **GitHub Repository Setup** ✅ **COMPLETED**
   * ✅ Set up GitHub repository with appropriate structure, branches, and initial documentation
   * ✅ Created centralized configuration system for portable deployment
   * ✅ Implemented automated setup process with `python setup.py`

3. **Embedding Techniques and Vector Database** ✅ **COMPLETED**
   * ✅ Selected sentence-transformers for embedding generation
   * ✅ Finalized FAISS as vector database choice
   * ✅ Successfully generated embeddings for all 4,859 text chunks
   * ✅ Created FAISS index for efficient similarity search and retrieval
   * ✅ Resolved all dependency conflicts and compatibility issues
   * ✅ **Added 6/29/2025:** Implemented full query system with LLM integration

4. **Open-Source License Compliance** 🔄 **PENDING**
   * ✅ Confirmed that current libraries used (sentence-transformers, FAISS, torch) meet open-source license requirements
   * 🔄 **Pending:** Final LLM model selection and license verification for downstream model integration

5. **README.md Documentation** ✅ **COMPLETED**
   * ✅ Created and incrementally updated comprehensive README.md with full setup instructions, usage examples, and project context
   * ✅ Added project structure overview and configuration explanation

## 📝 Model Version/Hash Logging
- Log all model names, descriptions, and hashes for reproducibility:
  ```bash
  python -m scaffold_core.model_logging
  ```
- See `outputs/model_version_log.json` for the log.

## ⚡ Model Benchmarking
- Benchmark all models for latency, memory, and output:
  ```bash
  python -m scaffold_core.benchmark_models
  ```
