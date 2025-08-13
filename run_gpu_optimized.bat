@echo off
echo 🚀 Starting Scaffold AI with GPU Optimization...
echo.

REM Set environment variables for GPU optimization
set SC_LLM_KEY=tinyllama-onnx
set PYTHONPATH=%cd%

REM Activate virtual environment and start the server
call scaffold_env_312\Scripts\activate.bat

REM Check GPU availability and show info
echo Checking GPU availability...
python -c "import torch; print('PyTorch CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

echo.
echo Starting optimized server...
python -m frontend.start_enhanced_ui --host 127.0.0.1 --port 5002

REM Keep the window open if there's an error
if errorlevel 1 (
    echo.
    echo An error occurred. Press any key to exit...
    pause
)
