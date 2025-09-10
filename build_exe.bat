@echo off
setlocal

REM Build Windows EXE for the Enhanced Flask UI without editing existing files
REM Prefer the currently activated Python venv; otherwise try common venv folders

cd /d %~dp0

REM If a venv is already active, use it. Otherwise, try to activate a known one.
if defined VIRTUAL_ENV (
  echo Using existing venv: %VIRTUAL_ENV%
) else (
  if exist scaffold_env_312_py31210\Scripts\activate (
    call scaffold_env_312_py31210\Scripts\activate
  ) else (
    if exist scaffold_env_312\Scripts\activate (
      call scaffold_env_312\Scripts\activate
    ) else (
      echo [!] No virtual environment active and none found. Proceeding with system Python.
    )
  )
)

python -m pip install --upgrade pip >nul 2>&1
python -m pip install pyinstaller python-dotenv --quiet

REM Dependency setup (CPU default; enable GPU by setting SC_INCLUDE_CUDA=1)
set "PYI_EXCLUDES=--exclude-module torch.cuda --exclude-module torch.backends.cudnn --exclude-module torch.backends.cuda --exclude-module torch.backends.cublas"
if "%SC_INCLUDE_CUDA%"=="1" (
  echo [i] GPU build requested via SC_INCLUDE_CUDA=1
  set "PYI_EXCLUDES="
  echo [i] Installing CUDA-enabled libraries (torch/cu121 and onnxruntime-gpu)
  python -m pip uninstall -y onnxruntime >nul 2>&1
  python -m pip install --upgrade onnxruntime-gpu --quiet
  python -m pip uninstall -y torch torchvision torchaudio >nul 2>&1
  python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio --quiet
) else (
  echo [i] CPU-only build (default). To try GPU, set SC_INCLUDE_CUDA=1
  python -m pip uninstall -y onnxruntime-gpu >nul 2>&1
  python -m pip install --upgrade onnxruntime --quiet
  python -m pip uninstall -y torch torchvision torchaudio >nul 2>&1
  python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio --quiet
)

REM Ensure vector assets are present; PyInstaller needs explicit data includes
if not exist vector_outputs\scaffold_index_1.faiss (
  echo [!] Missing vector_outputs\scaffold_index_1.faiss
)
if not exist vector_outputs\scaffold_metadata_1.json (
  echo [!] Missing vector_outputs\scaffold_metadata_1.json
)

REM Warm up ONNX export/tokenizer cache into outputs\onnx_models for offline EXE
echo [i] Warming up ONNX export/tokenizer (may take several minutes on first run)
python -c "import os; os.environ.setdefault('SC_FORCE_CPU','1'); from scaffold_core.llm import get_llm; get_llm(); print('ONNX/tokenizer warmup complete')" 2>nul

pyinstaller --noconfirm --clean --onedir --name ScaffoldAI-EnhancedUI ^
  --hidden-import encodings ^
  --hidden-import codecs ^
  --hidden-import scipy.special._cdflib ^
  %PYI_EXCLUDES% ^
  --add-data "frontend;frontend" ^
  --add-data "scaffold_core;scaffold_core" ^
  --add-data "model_config.json;." ^
  --add-data "outputs\onnx_models;outputs\onnx_models" ^
  --add-data "vector_outputs\scaffold_index_1.faiss;vector_outputs" ^
  --add-data "vector_outputs\scaffold_metadata_1.json;vector_outputs" ^
  --collect-all torch ^
  --collect-all torchvision ^
  --collect-all torchaudio ^
  --collect-all sentence_transformers ^
  --collect-all transformers ^
  --collect-all faiss ^
  --collect-all psutil ^
  --collect-all flask --collect-all flask_cors ^
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

REM Copy optional local env file into the built app folder for distribution
if exist .env (
  if exist dist\ScaffoldAI-EnhancedUI (
    copy /Y .env dist\ScaffoldAI-EnhancedUI\.env >nul
  )
)
if exist .env.local (
  if exist dist\ScaffoldAI-EnhancedUI (
    copy /Y .env.local dist\ScaffoldAI-EnhancedUI\.env.local >nul
  )
)

echo.
echo Build complete. Run the app with:
echo   dist\ScaffoldAI-EnhancedUI\ScaffoldAI-EnhancedUI.exe --host 0.0.0.0 --port 5002
echo.

endlocal


