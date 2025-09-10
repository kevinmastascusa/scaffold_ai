### ScaffoldAI — Technical Guide (Current Setup)

This document describes the present system architecture, retrieval/inference flow, packaging, and key operational notes.

#### 1) High‑level architecture
The system comprises three cooperative layers:
- Presentation: the Enhanced Flask UI in `frontend/app_enhanced.py` serves upload, chat, and feedback routes.
- Processing: `scaffold_core/` modules handle PDF parsing, chunking/embedding, similarity search, LLM invocation, and citation mapping.
- Storage: FAISS indices and JSON metadata live in `vector_outputs/` alongside run artifacts.

Notes
- UI serves upload, chat, and feedback pages; endpoints are hosted by Flask.
- Vector store is FAISS with separate JSON metadata for provenance and display.
- Models are pulled via Hugging Face; a token in `.env.local` enables gated models.

#### 2) Retrieval and inference flow (RAG)
End‑to‑end steps:
1) User submits a query; the server constructs a prompt template.
2) The vector index is queried to retrieve a small set of relevant chunks with metadata.
3) Retrieved passages and the user query are combined into a constrained context.
4) The LLM (`scaffold_core/llm.py`) generates a concise answer grounded in that context.
5) Citations are mapped back to source spans so the UI can display evidence.

Key details
- Chunking/embedding occurs offline; FAISS + metadata are distributed in `vector_outputs/`.
- Retrieval is similarity search over embeddings; top‑k is tuned for precision and coverage.
- Generation is constrained by retrieved context; output includes citation hooks.

#### 3) Packaging and distribution (Windows folder build)
Packaging outline:
- Build uses PyInstaller to produce a folder app at `dist/ScaffoldAI-EnhancedUI/` containing the `.exe` and all required libraries.
- The build script copies `.env.local` (if present) next to the `.exe` so model access is ready.
- Distribute the entire folder; teachers run the `.exe` from inside that folder with `vector_outputs/` present.

Operational notes
- Build script: `build_exe.bat` (copies `.env.local` if present into the `dist` folder).
- Teacher runs the `.exe` from inside the output folder; keep `vector_outputs/` beside it.

#### 4) Configuration
- `.env.local` (optional): `HUGGINGFACE_TOKEN=...`
- `model_config.json`: model choice and UI‑exposed parameters.
- `requirements.txt`: Python dependencies for developer installs.

#### 5) Troubleshooting (concise)
- Missing Python DLL on teacher machines: install Microsoft VC++ Redistributable (x64) 2015–2022.
- Models blocked or 401/403: add a valid Hugging Face token to `.env.local`.
- No results or empty answers: ensure `vector_outputs/` contains the expected FAISS and metadata files.


