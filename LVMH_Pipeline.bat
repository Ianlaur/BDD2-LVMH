@echo off
REM LVMH Voice-to-Tag Pipeline - Windows Double-Click Launcher
REM Double-click this file to run the full pipeline and open dashboard

cd /d "%~dp0"

echo.
echo ════════════════════════════════════════════════════════════
echo    LVMH Client Intelligence Pipeline
echo ════════════════════════════════════════════════════════════
echo.

REM Check/create virtual environment
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
    call .venv\Scripts\activate.bat
    pip install --upgrade pip -q
    pip install -r requirements.txt -q
    echo [OK] Environment ready
) else (
    call .venv\Scripts\activate.bat
    echo [OK] Virtual environment activated
)

REM Run the pipeline
echo.
echo Running pipeline...
echo ------------------------------------------------------------
python -m server.run_all
echo ------------------------------------------------------------
echo [OK] Pipeline complete

REM Kill any existing server on port 9000
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :9000 ^| findstr LISTENING') do (
    taskkill /F /PID %%a >nul 2>&1
)

REM Start local server in background
echo.
echo Starting local server on port 9000...
start /b python -m http.server 9000 >nul 2>&1
timeout /t 1 >nul
echo [OK] Server running

REM Open dashboard
echo.
echo Opening dashboard...
start http://localhost:9000/client/app/dashboard.html
echo [OK] Dashboard opened in browser

echo.
echo ════════════════════════════════════════════════════════════
echo Pipeline complete! Dashboard is now running.
echo.
echo Dashboard URL: http://localhost:9000/client/app/dashboard.html
echo.
echo Press any key to stop the server and exit.
echo ════════════════════════════════════════════════════════════

pause >nul

REM Kill server on exit
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :9000 ^| findstr LISTENING') do (
    taskkill /F /PID %%a >nul 2>&1
)
