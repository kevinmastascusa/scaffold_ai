# 🌱 Scaffold AI: Curriculum Recommendation Tool for Sustainability and Climate Resilience

**Collaborators:** Kevin Mastascusa, Joseph Di Stefano

**Date:** 6/23/2025

## 🌍 Project Overview

This project involves developing a specialized large language model (LLM)-based tool to assist educators in integrating sustainability and climate resilience topics into academic programs. The tool leverages state-of-the-art AI techniques to recommend high-quality, literature-backed educational materials, case studies, and project ideas.

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Git

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/scaffold_ai.git
   cd scaffold_ai
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the setup script:**
   ```bash
   python setup.py
   ```

4. **Add your PDF documents:**
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

### Pending Tasks

1. **Enhance PDF Extraction and Chunking**
   * Robustly handle Unicode and mathematical formulas.
   * Ensure Unicode and math symbols are preserved, detected, and reported in outputs.
   * Combined words issue: **solved** (see above).

2. **Integrate Unicode Cleaning Utility**
   * Add Unicode cleaning utility into the math-aware chunking pipeline.

3. **Resolve Environment and Import Issues**
   * Ensure smooth execution by addressing compatibility issues.

4. **Optional Enhancements**
   * Further improve chunking or Unicode/math reporting.
   * Rebuild or reinstall PyTorch and Sentence Transformers if compatibility issues persist.

5. **Documentation Updates**
   * Add detailed instructions for running scripts and interpreting outputs.
   * Include examples of expected outputs for clarity.

## 📅 Week 1 Tasks

1. **Define Preprocessing Methodology**
   * Establish detailed document preprocessing methodology, including chunking size, format, and metadata extraction.

2. **GitHub Repository Setup**
   * Set up GitHub repository with appropriate structure, branches, and initial documentation. ✅

3. **Embedding Techniques and Vector Database**
   * Select embedding techniques and finalize vector database choice (FAISS or Pinecone).

4. **Open-Source License Compliance**
   * Confirm that all libraries and models used (e.g., LLaMA, FAISS, MLflow) meet open-source license requirements for academic/public use.

5. **README.md Documentation**
   * Create and incrementally update a single README.md starting in Week 1, finalizing it in Week 11 with full setup instructions, usage examples, and project context. ✅
