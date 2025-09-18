@echo off
setlocal

REM Build Windows EXE for the Enhanced Flask UI without editing existing files
REM Requires an activated Python 3.12 venv with pyinstaller installed

cd /d %~dp0

if not exist scaffold_env_312\Scripts\python.exe (
  echo [!] Expected venv at scaffold_env_312 not found. Activate your env manually.
  goto :AFTER_ENV
)

call scaffold_env_312\Scripts\activate

:AFTER_ENV

python -m pip install --upgrade pip >nul 2>&1
python -m pip install pyinstaller python-dotenv --quiet

REM Ensure vector assets are present; PyInstaller needs explicit data includes
if not exist vector_outputs\scaffold_index_1.faiss (
  echo [!] Missing vector_outputs\scaffold_index_1.faiss
)
if not exist vector_outputs\scaffold_metadata_1.json (
  echo [!] Missing vector_outputs\scaffold_metadata_1.json
)

pyinstaller --noconfirm --clean --onedir --name ScaffoldAI-EnhancedUI ^
  --add-data "frontend;frontend" ^
  --add-data "scaffold_core;scaffold_core" ^
  --add-data "vector_outputs\scaffold_index_1.faiss;vector_outputs" ^
  --add-data "vector_outputs\scaffold_metadata_1.json;vector_outputs" ^
  --collect-all sentence_transformers ^
  --collect-all transformers ^
  --collect-all faiss ^
  --collect-all psutil ^
  --collect-all flask ^
  --collect-all flask_cors ^
  --collect-all werkzeug ^
  --collect-all jinja2 ^
  --collect-all optimum ^
  --collect-all onnxruntime ^
  --collect-all fitz ^
  --collect-all pymupdf ^
  --collect-all PyPDF2 ^
  --collect-all pypdf ^
  --collect-all pdfminer ^
  --collect-all pdfplumber ^
  run_enhanced_ui.py

echo.
echo Build complete. Run the app with:
echo   dist\ScaffoldAI-EnhancedUI\ScaffoldAI-EnhancedUI.exe --host 0.0.0.0 --port 5002
echo.

endlocal


