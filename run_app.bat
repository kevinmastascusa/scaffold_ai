@echo off
echo Starting Scaffold AI...
echo.

REM Set environment variables
set PORT=5002
set SC_LLM_KEY=distilgpt2
set PYTHONUNBUFFERED=1

REM Open browser
start "" http://localhost:%PORT%

REM Start the application
echo Starting server on http://localhost:%PORT%
echo Press Ctrl+C to stop the server
echo.
scaffold_env_312\Scripts\python -m frontend.start_enhanced_ui --host 127.0.0.1 --port %PORT%

pause
