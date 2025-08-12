@echo off
echo Starting Scaffold AI Enhanced UI...
echo.

REM Set environment variables
set SC_LLM_KEY=tinyllama
set PYTHONPATH=%cd%

REM Activate virtual environment and start the server
call scaffold_env_312\Scripts\activate.bat
python -m frontend.start_enhanced_ui --host 127.0.0.1 --port 5002

REM Keep the window open if there's an error
if errorlevel 1 (
    echo.
    echo An error occurred. Press any key to exit...
    pause
)
