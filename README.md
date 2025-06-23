# 🌱 Scaffold AI: Curriculum Recommendation Tool for Sustainability and Climate Resilience

**Collaborators:** Kevin Mastascusa, Joseph Di Stephano

**Date:** 6/23/2025

## 🌍 Project Overview

This project involves developing a specialized large language model (LLM)-based tool to assist educators in integrating sustainability and climate resilience topics into academic programs. The tool leverages state-of-the-art AI techniques to recommend high-quality, literature-backed educational materials, case studies, and project ideas.

## 🎯 Goals and Objectives

The primary goal is to create a user-friendly, accurate, and literature-grounded AI tool capable of:

* 📚 Suggesting relevant and up-to-date curriculum content.
* 🔍 Ensuring transparency by referencing scholarly sources for every recommendation.
* 🧩 Facilitating easy integration into existing courses, supporting targeted learning outcomes.

## 🛠️ Proposed System Architecture

The system will include three key components:

* **Retrieval-Augmented Generation (RAG) Framework**
* **Vector Embeddings:** Pre-process and embed key sustainability and resilience literature into a vector database (e.g., FAISS, Pinecone).
* **Document Retrieval:** Efficiently search and retrieve relevant sections from scholarly sources based on embedded user queries.

## 🧠 Large Language Model (LLM)

Open-source models under consideration:

* **Llama 3 (Meta):** meta-llama/Llama-3.1-8B-Instruct, Llama-3.2-1B
* **Mistral-7B (Mistral AI):** Mistral-7B-v0.1, Mistral-7B-Instruct-v0.2, v0.3
* **Phi-3 Mini (Microsoft):** Phi-3.5-mini-instruct, Phi-3-mini-4k-instruct

## 🔗 Citation Tracking and Transparency

* 🔗 Direct linking between generated content and original sources.
* 🖥️ Interactive UI to show how each recommendation is grounded in literature.

## 🔄 Technical Workflow

1. 📥 **Corpus Collection:** Curate scholarly papers, reports, and policy documents.
2. 🧹 **Data Preprocessing:** Clean, segment, and prepare documents.
3. 🧠 **Embedding and Storage:** Embed corpus data and store in a vector database.
4. ⚙️ **Inference Engine:** Retrieve and use embeddings to augment LLM output.
5. 📝 **Citation Layer:** Annotate outputs with clear citation links.

## 📅 Project Timeline Overview

The project follows a structured timeline with week-by-week development phases. Key phases include:

* 🏗️ Setting up the preprocessing pipeline and repository structure
* 🧠 Embedding the curated document corpus and validating retrieval quality
* 🔧 Integrating the LLM and developing the initial prototype
* 🖼️ Building and refining the user interface
* 🧾 Implementing citation tracking and performing usability testing
* 🧑‍🏫 Engaging stakeholders for feedback and refining the final product

Optional enhancements may include a real-time feedback loop in the UI and tag-based filtering of recommendations.

## 📈 Evaluation Overview

The system will be evaluated based on its ability to:

* 🧠 Retrieve relevant and accurate curriculum materials
* 🔍 Generate transparent, literature-backed recommendations
* ⚡ Provide a responsive and accessible user experience
* 👥 Satisfy stakeholders through iterative testing and feedback

Evaluation will include both qualitative feedback from faculty and technical performance benchmarks such as system responsiveness, citation traceability, and usability outcomes.

## ✅ Expected Outcomes

* 🛠️ A functioning prototype generating cited curriculum recommendations.
* 🖥️ Intuitive UI ready for pilot use.
* 📄 Comprehensive documentation for future development.

## 🧾 Conclusion

This tool aims to enhance the integration of sustainability topics in education through transparency, traceability, and collaboration. Stakeholder engagement will guide the development of a practical and impactful final product.
