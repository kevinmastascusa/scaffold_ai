@echo off
setlocal

echo Starting Scaffold AI Streamlit app...
echo.

REM Activate venv if present
if exist scaffold_env_312\Scripts\activate.bat (
  call scaffold_env_312\Scripts\activate.bat
)

REM Optional: leave SC_LLM_KEY empty to use model_config.json; set it here only if you need a one-off override
REM if "%SC_LLM_KEY%"=="" set SC_LLM_KEY=distilgpt2

REM Launch
streamlit run streamlit_app.py --server.port 5003 --server.address 127.0.0.1

endlocal

