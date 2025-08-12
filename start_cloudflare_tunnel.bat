@echo off
echo ========================================
echo ScaffoldAI Cloudflare Tunnel Setup
echo ========================================
echo.
echo This will start your ScaffoldAI app and expose it via Cloudflare Tunnel
echo.
echo Step 1: Starting ScaffoldAI Enhanced UI...
echo.

REM Start the ScaffoldAI app in the background
start /B "ScaffoldAI App" "ScaffoldAI-EnhancedUI.exe"

REM Wait for app to start
echo Waiting for app to start (10 seconds)...
timeout /t 10 /nobreak > nul

REM Check if app is running
netstat -an | findstr ":5002" > nul
if %errorlevel% neq 0 (
    echo ERROR: ScaffoldAI app is not running on port 5002
    echo Please make sure you have:
    echo 1. Downloaded ScaffoldAI-EnhancedUI.exe
    echo 2. Created .env file with your Hugging Face token
    echo 3. Placed both files in the same folder
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
