@echo off
setlocal

cd /d %~dp0

REM Ensure venv is active for consistent tools
if defined VIRTUAL_ENV (
  echo Using existing venv: %VIRTUAL_ENV%
) else if exist scaffold_env_312_py31210\Scripts\activate (
  call scaffold_env_312_py31210\Scripts\activate
) else if exist scaffold_env_312\Scripts\activate (
  call scaffold_env_312\Scripts\activate
) else (
  echo [!] No virtual environment active and none found. Proceeding with system Python.
)

REM Start rebuild detached, suppressing interactive prompts; log to build_exe.log
start "" /b cmd /c "call .\build_exe.bat" ^> build_exe.log 2^>^&1
echo Build started in background. Tail log with: type build_exe.log | more

endlocal


