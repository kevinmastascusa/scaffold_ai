### Scaffold AI — Teacher Start Guide

This guide helps you run Scaffold AI quickly with minimal setup.

## What you need
- A Windows computer
- A web browser (Chrome, Edge, or Firefox)

## Quick start (Windows — no install)
1. Download the app from the project’s releases page: [GitHub Releases](https://github.com/kevinmastascusa/scaffold_ai/releases).
2. Unzip the downloaded folder (for example, `ScaffoldAI-EnhancedUI-vX.Y.Z.zip`).
3. Open the unzipped folder, then open `dist\ScaffoldAI-EnhancedUI\`.
4. Double‑click `ScaffoldAI-EnhancedUI.exe`.
5. In your browser, go to `http://localhost:5002`.

If double‑clicking doesn’t work, open Command Prompt and run:
```bat
"dist\ScaffoldAI-EnhancedUI\ScaffoldAI-EnhancedUI.exe" --host 0.0.0.0 --port 5002
```

## Optional: Hugging Face token (improves model access)
Some AI models require a Hugging Face token. You can run without it, but access may be limited.
1. Get a token from your Hugging Face account → Settings → Access Tokens.
2. In the same folder as the `.exe`, create a file named `.env.local` (or `.env`) containing:
```dotenv
HUGGINGFACE_TOKEN=hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```
3. Save the file and start the app again.

## Required data files
Make sure these files exist next to the app inside a `vector_outputs` folder:
- `vector_outputs\scaffold_index_1.faiss`
- `vector_outputs\scaffold_metadata_1.json`

If missing, copy the entire `vector_outputs` folder from the project package or ask your project lead for the correct files.

## Using the app
- Open `http://localhost:5002`.
- You can:
  - Ask curriculum questions (e.g., “How can I integrate sustainability into my Fluid Mechanics course?”).
  - Upload a course syllabus PDF and request targeted suggestions.
  - Review sources/citations for each response.

## Troubleshooting
- Windows Defender warning: click “More info” → “Run anyway”.
- Port already in use: start with another port, e.g.:
  ```bat
  "dist\ScaffoldAI-EnhancedUI\ScaffoldAI-EnhancedUI.exe" --port 5003
  ```
  then open `http://localhost:5003`.
- Missing data files: ensure the two files listed under “Required data files” are present.
- Token/model access errors: add your `HUGGINGFACE_TOKEN` to `.env.local` and restart the app.

## Fallback: Run with Python (if .exe is blocked)
1. Install Python 3.12 or newer.
2. In the project root, run:
   ```bat
   python -m venv scaffold_env_312
   scaffold_env_312\Scripts\activate
   pip install -r requirements.txt
   set HUGGINGFACE_TOKEN=hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
   python frontend\start_enhanced_ui.py --port 5002
   ```
3. Open `http://localhost:5002`.

## Share with colleagues
Zip the entire `dist\ScaffoldAI-EnhancedUI\` folder (including `vector_outputs`) and share. They only need to unzip and run `ScaffoldAI-EnhancedUI.exe`.


