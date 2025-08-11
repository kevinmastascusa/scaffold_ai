@echo off
setlocal

echo Starting Scaffold AI Streamlit app...
echo.

REM Activate venv if present
if exist scaffold_env_312\Scripts\activate.bat (
  call scaffold_env_312\Scripts\activate.bat
)

REM Optional: set a token-free default LLM
if "%SC_LLM_KEY%"=="" set SC_LLM_KEY=distilgpt2

REM Launch
streamlit run streamlit_app.py --server.port 5003 --server.address 127.0.0.1

endlocal

