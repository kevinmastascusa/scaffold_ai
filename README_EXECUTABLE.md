# Scaffold AI Enhanced UI Executable

This guide shows how to run the existing Flask Enhanced UI as a Windows executable without editing project files.

## 1) Provide your Hugging Face token

Create a `.env.local` (or `.env`) file in the project root with:

```dotenv
HUGGINGFACE_TOKEN=hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

The launcher `run_enhanced_ui.py` will load it automatically if `python-dotenv` is installed.

## 2) Build the executable

On Windows:

```bat
scaffold_env_312\Scripts\activate
build_exe.bat
```

This produces `dist/ScaffoldAI-EnhancedUI/ScaffoldAI-EnhancedUI.exe`.

## 3) Run the app

```bat
dist\ScaffoldAI-EnhancedUI\ScaffoldAI-EnhancedUI.exe --host 0.0.0.0 --port 5002
```

Then open `http://localhost:5002`.

## Notes

- Requires `vector_outputs/scaffold_index_1.faiss` and `vector_outputs/scaffold_metadata_1.json`.
- No changes to the existing UI code were made; the launcher only loads env and delegates to `frontend/start_enhanced_ui.py`.
- If packaging errors mention missing modules (e.g., torch), add `--collect-all <module>` to `build_exe.bat` as needed.
