# Changelog

## v0.1.0-prebeta (2025-09-18)

- Academic report (LaTeX) with TikZ workflow/settings figure (`report/scaffold_ai_status_report.pdf`).
- MiKTeX-friendly Windows build script (`build_report.bat`) with latexmk/pdflatex fallback.
- RAG pipeline stabilized: FAISS retrieval, cross-encoder rerank, TinyLlama generation.
- Centralized settings in `scaffold_core/config.py` captured in figure/table.
- Documentation updates and Pre-Beta status in `README.md`.

Known limitations:
- Compact LLM limits depth on complex synthesis.
- Citation granularity limited to available metadata.
