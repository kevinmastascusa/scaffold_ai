### Scaffold AI — Research Report and Handoff Guide

This single document serves both as a research summary and a comprehensive handoff for future contributors.

## Executive summary
- **Purpose**: AI-assisted curriculum recommendations for sustainability/climate resilience.
- **Interface**: Modern Flask web app with chat and syllabus PDF upload.
- **Core techniques**: RAG over a curated corpus (FAISS), LLM inference via Hugging Face, robust citation layer.
- **Where to run**: Windows EXE for teachers; Python workflow for developers. Teacher quick start: see `TEACHER_ZIP_RUN.md`.
- **Status (updated 2025-08-26)**:
  - Quick tests: 15/15 success, avg 1.48s per prompt (see `quick_llm_test_summary_20250803_001740.txt`).
  - Existing model suite: 15/15 success, avg 2.44s (`existing_model_test_summary_20250803_002413.txt`).
  - ONNX benchmark: first-query latency improved (~30.35s → ~16.98s) on TinyLlama; overall average comparable (see `benchmark_results.json`).

## System overview
- **Frontend/UI**: `frontend/app_enhanced.py`, templates in `frontend/templates/`, static assets in `frontend/static/`.
- **Core processing**: `scaffold_core/` modules for PDF processing, vector search, LLM, and citation handling.
- **Indexes and outputs**: `vector_outputs/` for FAISS + metadata, `outputs/` for extracts, `conversations/` for chat logs.

## Architecture diagram (high‑level)
```mermaid
graph TD
  User["Teacher Browser"] -->|HTTP| Flask["Flask Enhanced UI\n(frontend/app_enhanced.py)"]
  Flask -->|Upload PDF| PDF["PDF Processor\n(scaffold_core/pdf_processor.py)"]
  Flask -->|Query| Query["Enhanced Query\n(scaffold_core/vector/enhanced_query_improved.py)"]
  Query --> RAG["Retrieval + Reranking"]
  RAG --> FAISS[("FAISS Index\nvector_outputs/scaffold_index_1.faiss")]
  RAG --> Meta[("Metadata\nvector_outputs/scaffold_metadata_1.json")]
  Query --> LLM["LLM Inference\n(scaffold_core/llm.py)"]
  LLM --> HF["Hugging Face Models"]
  Query --> Cite["Citation Handler\n(scaffold_core/citation_handler.py)"]
  Cite --> Flask
  PDF --> Outputs[("outputs/*.json")]
  Flask --> Feedback["Feedback UI (/feedback)"]
  subgraph Storage
    FAISS
    Meta
    Outputs
    Conversations[("conversations/")]
  end
```

If the diagram above does not render, refer to this image:

![High-level Architecture](diagrams/complete_pipeline_architecture.png)

## End‑to‑end data flow
```mermaid
flowchart LR
  A[Teacher] --> B[Browser UI]
  B --> C[/Upload Syllabus/]
  C --> D[PDF Processor: extract text, sections, objectives]
  D --> E[Chunk + Embed (if not already built)]
  E --> F[(FAISS Index)]
  B --> G[/Ask Question/]
  G --> H[Retrieve top‑k candidates from FAISS]
  H --> I[Cross‑encoder reranking (optional)]
  I --> J[LLM with prompt + retrieved context]
  J --> K[Citation assembly]
  K --> L[Response + citations to UI]
```

If the diagram above does not render, refer to this image:

![End-to-end Data Flow](diagrams/data_flow_diagram.png)

## Reproducible setup
### Option A: Windows EXE (no install)
- Download from releases, unzip, and run `dist\ScaffoldAI-EnhancedUI\ScaffoldAI-EnhancedUI.exe`.
- Optional: in the same folder, create `.env.local` with `HUGGINGFACE_TOKEN=...`.
- Ensure `vector_outputs/scaffold_index_1.faiss` and `vector_outputs/scaffold_metadata_1.json` are present.

### Option B: Developer setup (Python 3.12+)
```bat
python -m venv scaffold_env_312
scaffold_env_312\Scripts\activate
pip install -r requirements.txt
set HUGGINGFACE_TOKEN=hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
python frontend\start_enhanced_ui.py --port 5002
```
- Open `http://localhost:5002`.

## Data preparation pipeline
1. Place PDFs in `data/` (and subfolders).
2. Extract + chunk:
   ```bat
   python scaffold_core\scripts\chunk\ChunkTest.py
   ```
3. Build embeddings + FAISS index:
   ```bat
   python scaffold_core\vector\main.py
   ```
4. Outputs:
   - Extracts: `outputs/`
   - Index + metadata: `vector_outputs/`

## Model and configuration
- Central config: `scaffold_core/config.py`
- LLM logic: `scaffold_core/llm.py` (uses Hugging Face; set `HUGGINGFACE_TOKEN`).
- Query/RAG: `scaffold_core/vector/enhanced_query_improved.py`
- Model selection and parameters: see `model_config.json` and UI controls (if enabled).

## Running the Enhanced UI
- Direct app: `frontend/app_enhanced.py` (served via `frontend/start_enhanced_ui.py`).
- Endpoints: `/api/chat`, `/api/upload-syllabus`, `/api/feedback`, `/api/health`.
- Templates: `frontend/templates/index_enhanced.html`.

## Evaluation and benchmarks
- Quick tests: `quick_llama_eval.py`, `simple_llama_test.py`, `pipeline_llama_test.py`.
- Comprehensive: `full_test_suite.py`, `scaffold_core/scripts/run_tests.py`.
- Results snapshots: `existing_model_test_results_*.json`, `quick_llm_test_summary_*.txt`, `Full Test Results*.txt`.
- GPU/ONNX logs: `benchmark_gpu_log.txt`, `benchmark_onnx_log.txt`.

### Latest results snapshot
- Quick LLM test (2025‑08‑03 00:17): 15/15 passed; average response time 1.48s; average response length 135 chars. File: `quick_llm_test_summary_20250803_001740.txt`.
- Existing model test (2025‑08‑03 00:23): 15/15 passed; average response time 2.44s; average response length 226 chars. File: `existing_model_test_summary_20250803_002413.txt`.
- TinyLlama benchmark (2025‑08‑04):
  - Baseline first‑query: 30.35s → ONNX first‑query: 16.98s (≈44% faster cold start).
  - Baseline avg across 3 prompts: 23.91s; ONNX avg: 24.46s (comparable overall). File: `benchmark_results.json`.


## Packaging and releases
- Build EXE (Windows):
  ```bat
  scaffold_env_312\Scripts\activate
  build_exe.bat
  ```
- Output: `dist/ScaffoldAI-EnhancedUI/ScaffoldAI-EnhancedUI.exe`
- Provide checksums (e.g., `*.exe.sha256.txt`) and a zipped folder including `vector_outputs`.

## Troubleshooting
- Defender warning: “More info” → “Run anyway”.
- Port busy: start with `--port 5003` and open `http://localhost:5003`.
- Missing data files: ensure FAISS index + metadata in `vector_outputs/`.
- Token/model access: set `HUGGINGFACE_TOKEN` in `.env.local` or environment.
- Missing modules in packaged EXE: add `--collect-all <module>` to `build_exe.bat`.

## Repo orientation (handoff)
- Key dirs:
  - `frontend/`: Flask app + UI
  - `scaffold_core/`: processing, LLM, vector, citations
  - `outputs/`, `vector_outputs/`, `conversations/`: run artifacts
  - `documentation/`: additional docs and reports
- High‑value files:
  - `frontend/start_enhanced_ui.py`: startup checks and run
  - `scaffold_core/pdf_processor.py`: syllabus/PDF handling
  - `scaffold_core/vector/enhanced_query_improved.py`: retrieval + LLM glue
  - `scaffold_core/citation_handler.py`: citation assembly
  - `scaffold_core/llm.py`: model loading and invocation

## Maintenance & future work
- Data: refresh corpus and re‑embed when adding PDFs.
- Models: evaluate larger/smarter models as resources allow.
- UI: multi‑file upload, analytics, export, and collaboration features.
- Infra: optional GPU acceleration, ONNX optimization, and caching.

## Next steps (detailed implementation plan)

### 1) Host the web app on "tux" (Linux server)

- **Prereqs (once)**
  - System packages:
    ```bash
    sudo apt update && sudo apt install -y python3-venv python3-pip build-essential nginx
    ```
  - Optional TLS (if using a domain): `sudo apt install -y certbot python3-certbot-nginx`

- **App install (as deploy user)**
  ```bash
  cd /opt
  sudo mkdir -p /opt/scaffold_ai && sudo chown "$USER" /opt/scaffold_ai
  git clone https://github.com/kevinmastascusa/scaffold_ai.git /opt/scaffold_ai
  cd /opt/scaffold_ai
  python3 -m venv venv
  source venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt gunicorn
  ```

- **Environment** (secrets not in git)
  ```bash
  cat > /opt/scaffold_ai/.env << 'EOF'
  FLASK_ENV=production
  SC_HOST=127.0.0.1
  SC_PORT=8000
  HUGGINGFACE_TOKEN=hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
  # Optional: increase workers/threads based on CPU cores
  GUNICORN_WORKERS=3
  GUNICORN_THREADS=2
  EOF
  ```

- **Gunicorn systemd unit**
  ```ini
  # /etc/systemd/system/scaffoldai.service
  [Unit]
  Description=Scaffold AI (Flask + Gunicorn)
  After=network.target

  [Service]
  Type=simple
  User=%i
  WorkingDirectory=/opt/scaffold_ai
  EnvironmentFile=/opt/scaffold_ai/.env
  ExecStart=/opt/scaffold_ai/venv/bin/gunicorn \
    --workers ${GUNICORN_WORKERS:-3} --threads ${GUNICORN_THREADS:-2} \
    --timeout 180 --bind ${SC_HOST:-127.0.0.1}:${SC_PORT:-8000} \
    frontend.app_enhanced:app
  Restart=always
  RestartSec=5

  [Install]
  WantedBy=multi-user.target
  ```
  Enable + start:
  ```bash
  sudo systemctl daemon-reload
  sudo systemctl enable scaffoldai.service
  sudo systemctl start scaffoldai.service
  sudo systemctl status scaffoldai.service --no-pager
  ```

- **Nginx reverse proxy**
  ```nginx
  # /etc/nginx/sites-available/scaffoldai
  server {
    listen 80;
    server_name your.domain.example; # or _ for IP access

    client_max_body_size 32M; # allow PDF uploads

    location / {
      proxy_pass http://127.0.0.1:8000;
      proxy_set_header Host $host;
      proxy_set_header X-Real-IP $remote_addr;
      proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
      proxy_set_header X-Forwarded-Proto $scheme;
      proxy_read_timeout 300;
    }
  }
  ```
  Activate and reload:
  ```bash
  sudo ln -s /etc/nginx/sites-available/scaffoldai /etc/nginx/sites-enabled/scaffoldai
  sudo nginx -t && sudo systemctl reload nginx
  ```
  TLS (optional): `sudo certbot --nginx -d your.domain.example`

- **Data/index deployment**
  - Place `vector_outputs/scaffold_index_1.faiss` and `vector_outputs/scaffold_metadata_1.json` under `/opt/scaffold_ai/vector_outputs/`.
  - For updates, rsync only changed files. Reload Gunicorn if needed.

- **Alternative quick hosting**
  - Cloudflare Tunnel (see `CLOUDFLARE_SETUP.md`) to expose `http://127.0.0.1:8000` without opening ports.

### 2) Implement robust context management

- **Goals**: persistent conversations, controllable context window, summarization to keep prompts small, privacy.

- **Storage**
  - Start with SQLite (zero‑ops), upgrade to Postgres when multi‑user is needed.
  - Optional Redis for ephemeral context cache.

- **Schema (suggested)**
  ```sql
  -- conversations(id PK, user_id, title, created_at, updated_at)
  -- messages(id PK, conversation_id FK, role ENUM('user','assistant','system'), content TEXT, tokens INT, created_at)
  -- memory_summaries(id PK, conversation_id FK, summary TEXT, last_message_id INT, created_at)
  -- citations(id PK, message_id FK, doi TEXT, source_id TEXT, score REAL, span_start INT, span_end INT)
  ```

- **API additions**
  - `POST /api/session/start` → returns `conversation_id`
  - `POST /api/session/end` → archive
  - `GET /api/session/:id` → messages + summary
  - Existing `/api/chat` accepts `conversation_id` and returns `message_id`, updated `summary` when rolled

- **Frontend**
  - Persist `conversationId` in `localStorage`.
  - Add simple session selector and “New chat”.

- **Context selection algorithm**
  1. Load `summary` if exists; start prompt with it.
  2. From latest `N` messages, include until token budget reached.
  3. If budget exceeded, run summarization on the oldest half and store/update `memory_summaries`.
  4. Optionally retrieve top‑k session snippets via mini‑embedding for relevance to current query.

- **Summarization trigger**
  - Trigger when total tokens in window > threshold (e.g., 2k).
  - Use a small instruct model for speed; store version/hash for reproducibility.

### 3) Improve citations and claim traceability

- **Retrieval hygiene**
  - Deduplicate by DOI/URL; prefer canonical DOI.
  - Track page ranges/paragraph IDs.

- **Inline references**
  - Use bracketed indices like `[1]`, `[2]` in the generated text, each mapping to a Source list with title, authors, year, DOI/URL.
  - Attach citation spans (start/end char offsets) to messages for UI highlighting.

- **Metadata enrichment**
  - Resolve DOIs via Crossref with caching and rate limiting.
  - Normalize titles/authors; store in `vector_outputs` metadata when rebuilding.

- **Scoring & filtering**
  - Combine retriever score + reranker score; show confidence in UI.
  - Drop low‑confidence citations or flag them as tentative.

- **UI enhancements**
  - Clickable citations scroll to the Sources section; hover to preview.
  - Toggle “Show citation spans” for auditors.

- **Evaluation**
  - Manual spot checks with `CITATION_DEBUG_REPORT.md`.
  - Metrics: citation precision@k, missing‑citation rate, hallucination flag rate.

### 4) Monitoring, security, and performance

- **Monitoring**: request IDs, structured JSON logs, time‑to‑first‑token, retriever latency, LLM latency; redact PII.
- **Security**: CORS restrict to your domain, basic auth or SSO in front of Nginx, rate limiting (Nginx `limit_req`).
- **Performance**: preload models, tune Gunicorn workers/threads, enable HTTP keep‑alive, cache frequent queries, consider quantized models.

## Ethics, privacy, and licensing
- Avoid uploading sensitive PDFs.
- Cite sources transparently.

## Contacts
- See repo README for collaborators and contact info.


