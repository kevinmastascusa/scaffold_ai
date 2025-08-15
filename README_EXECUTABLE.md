# Scaffold AI Enhanced UI — Windows EXE Quick Start (for Teachers)

Run Scaffold AI with no installation. This guide covers downloading from GitHub, adding an optional Hugging Face token, required data files, and launching the app.

## 1) Download from GitHub (no install)

1. Open the project releases page: [GitHub Releases](https://github.com/kevinmastascusa/scaffold_ai/releases)
2. Download the latest release asset:
   - Prefer a zipped app folder (e.g., `ScaffoldAI-EnhancedUI-vX.Y.Z.zip`). Unzip it anywhere (e.g., `Downloads` or `Desktop`).
   - If only a single executable is provided (e.g., `ScaffoldAI-EnhancedUI-vX.Y.Z.exe`), place it in its own folder.
3. Optional: Verify the checksum using the matching `*.exe.sha256.txt` file.

After unzipping, you should have a folder like:

```
dist\ScaffoldAI-EnhancedUI\
  ScaffoldAI-EnhancedUI.exe
  (other files and folders)
```

## 2) Optional: Enable Hugging Face models (token)

Some models require a Hugging Face token. You can run without a token, but availability may be limited.

1. Get your token: open Hugging Face → Settings → Access Tokens.
2. In the app folder (same folder as the `.exe`), create a file named `.env.local` (or `.env`) with:

```dotenv
HUGGINGFACE_TOKEN=hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

The launcher loads this automatically if `python-dotenv` is bundled.

## 3) Required data files

Make sure these files exist next to the app inside a `vector_outputs` folder:

- `vector_outputs/scaffold_index_1.faiss`
- `vector_outputs/scaffold_metadata_1.json`

If they are missing, copy the entire `vector_outputs` folder from the project root (or from a provided package) into the same folder as the `.exe`.

## 4) Run the app

### Easiest (double-click)
1. Open the folder `dist\ScaffoldAI-EnhancedUI\` (or the folder where you unzipped the release).
2. Double‑click `ScaffoldAI-EnhancedUI.exe`.
3. Open your browser to `http://localhost:5002`.

### Command Prompt (if double‑click doesn’t work)
```bat
"dist\ScaffoldAI-EnhancedUI\ScaffoldAI-EnhancedUI.exe" --host 0.0.0.0 --port 5002
```
Then open `http://localhost:5002`.

To use a different port (e.g., if 5002 is busy):
```bat
"dist\ScaffoldAI-EnhancedUI\ScaffoldAI-EnhancedUI.exe" --port 5003
```
Open `http://localhost:5003`.

## 5) Build the executable yourself (optional)

If you prefer to build locally from source:

1. Create/activate the project environment (Windows):
   ```bat
   scaffold_env_312\Scripts\activate
   ```
2. Build the executable:
   ```bat
   build_exe.bat
   ```
3. The build output will be at:
   - `dist/ScaffoldAI-EnhancedUI/ScaffoldAI-EnhancedUI.exe`

You can then run it as shown above.

## Troubleshooting

- Windows Defender warning: click “More info” → “Run anyway”.
- Missing data files: ensure `vector_outputs/scaffold_index_1.faiss` and `vector_outputs/scaffold_metadata_1.json` are next to the app.
- Token/model access errors: add your Hugging Face token to `.env.local` in the app folder and restart.
- Port in use: start with a different port (e.g., `--port 5003`) and open `http://localhost:5003`.
- Packaging misses a module: when building, add `--collect-all <module>` in `build_exe.bat` (e.g., for `torch`).
