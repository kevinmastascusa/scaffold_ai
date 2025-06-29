# 🌱 Scaffold AI: Curriculum Recommendation Tool for Sustainability and Climate Resilience

**Collaborators:** Kevin Mastascusa, Joseph Di Stefano

**Date:** 6/26/2025

## 🌍 Project Overview

This project involves developing a specialized large language model (LLM)-based tool to assist educators in integrating sustainability and climate resilience topics into academic programs. The tool leverages state-of-the-art AI techniques to recommend high-quality, literature-backed educational materials, case studies, and project ideas.

## 🚀 Getting Started

### Prerequisites

- Python 3.11 or higher (recommended: Python 3.11 for best compatibility)
- Git

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/scaffold_ai.git
   cd scaffold_core
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the setup script:**
   ```bash
   python setup.py
   ```

4. **Set up local LLM endpoint (required for full functionality):**
   - See [Local Setup Guide](documentation/local_setup_guide.md) for detailed instructions
   - Install and configure Ollama for local Mistral model
   - Configure environment variables for secure endpoint access

5. **Add your PDF documents:**
   Place your PDF files in the `data/` directory. The system will automatically process all PDF files found in this directory and its subdirectories.

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
   python scaffold_core/scripts/postprocess_combined_words.py
   python scaffold_core/scripts/generate_combined_words_report.py
   ```
   - See `outputs/combined_words_analysis_report.txt` for a detailed summary of remaining combined words (now mostly legitimate technical/academic terms).

### Project Structure

```
scaffold_ai/
├── data/                    # Place your PDF documents here
├── outputs/                 # Chunked text extracts
├── vector_outputs/          # Vector embeddings and indexes
├── math_outputs/            # Math-aware processing results
├── scaffold_core/           # Core processing modules
│   ├── config.py           # Central configuration
│   ├── scripts/            # Processing scripts
│   └── vector/             # Vector processing
├── setup.py                # Setup script
└── requirements.txt        # Python dependencies
```

### Configuration

All paths and settings are centrally managed in `scaffold_core/config.py`. The configuration automatically adapts to your workspace location, making the project portable for anyone who clones the repository.

## 🎯 Goals and Objectives

The primary goal is to create a user-friendly, accurate, and literature-grounded AI tool capable of:

* 📚 Suggesting relevant and up-to-date curriculum content.
* 🔍 Ensuring transparency by referencing scholarly sources for every recommendation.
* 🧩 Facilitating easy integration into existing courses, supporting targeted learning outcomes.

## 🛠️ Combined Words Issue: Solved

- **Combined words (e.g., "environmentalsustainability") are now programmatically detected and fixed.**
- **Automated post-processing and reporting scripts:**
  - `scaffold_core/scripts/postprocess_combined_words.py` (fixes combined words)
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

Open-source models under consideration:

* **Llama 3 (Meta):** meta-llama/Llama-3.1-8B-Instruct, Llama-3.2-1B
* **Mistral-7B (Mistral AI):** Mistral-7B-v0.1, Mistral-7B-Instruct-v0.2, v0.3
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

1. **Implement Semantic Search and Retrieval**
   * Create query interface for the completed FAISS index
   * Implement semantic search functionality using vector embeddings
   * Add query performance testing and optimization

2. **Build User Interface for Knowledge Base**
   * Develop web interface for querying the vectorized knowledge base
   * Create intuitive search interface with result display
   * Add query result visualization and export functionality

### 🔧 Medium Priority

3. **Enhance PDF Extraction and Chunking**
   * Integrate full Unicode analysis into page-based chunks
   * Complete math-aware chunking improvements in `ChunkTest_Math.py`
   * Re-integrate math-aware chunking with vector pipeline

4. **Advanced Search Features**
   * Implement filters and faceted search capabilities
   * Add relevance scoring and ranking improvements
   * Create citation tracking and source linking

5. **Testing and Validation**
   * Implement automated testing for the configuration system
   * Add comprehensive test suite for vector operations
   * Create performance benchmarks and validation metrics

### 📈 Future Enhancements

6. **System Optimization**
   * Add support for incremental updates to the vector database
   * Implement progress tracking for long-running processes
   * Add configuration profiles for different deployment scenarios

7. **Documentation and Examples**
   * Add vector query examples in the README
   * Create user guides for different use cases
   * Develop API documentation for programmatic access

8. **Optional Enhancements**
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

4. **Open-Source License Compliance** 🔄 **PENDING**
   * ✅ Confirmed that current libraries used (sentence-transformers, FAISS, torch) meet open-source license requirements
   * 🔄 **Pending:** Final LLM model selection and license verification for downstream model integration

5. **README.md Documentation** ✅ **COMPLETED**
   * ✅ Created and incrementally updated comprehensive README.md with full setup instructions, usage examples, and project context
   * ✅ Added project structure overview and configuration explanation
