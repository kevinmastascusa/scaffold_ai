@echo off
echo ========================================
echo ScaffoldAI Cloudflare Tunnel Setup
echo ========================================
echo.
echo This will start your current Python server and expose it via Cloudflare Tunnel
echo.
echo Step 1: Starting ScaffoldAI Enhanced UI (GPU-optimized)...
echo.

REM Start the Python app in a new window
start "ScaffoldAI App" cmd /c run_gpu_optimized.bat

REM Wait for app to start
echo Waiting for app to start (12 seconds)...
timeout /t 12 /nobreak > nul

REM Verify port 5002 is listening
netstat -an | findstr ":5002" > nul
if %errorlevel% neq 0 (
    echo ERROR: ScaffoldAI app is not running on port 5002
    echo Please ensure the server started successfully.
    echo.
    pause
    exit /b 1
)

echo ✅ ScaffoldAI app is running on http://localhost:5002
echo.
echo Step 2: Starting Cloudflare Tunnel...
echo.
echo Installing cloudflared (if not already installed)...
winget install Cloudflare.Cloudflared --accept-source-agreements --accept-package-agreements

echo.
echo Starting Cloudflare Tunnel...
echo This will create a public URL for your app.
echo Keep this window open to maintain the tunnel.
echo.
echo ========================================
echo PUBLIC URL WILL APPEAR BELOW
echo ========================================
echo.

REM Start Cloudflare Tunnel
cloudflared tunnel --url http://localhost:5002

echo.
echo Tunnel stopped. Press any key to exit...
pause
