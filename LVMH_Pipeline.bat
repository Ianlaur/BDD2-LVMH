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

REM Check if npm is available and dashboard exists
where npm >nul 2>&1
if %ERRORLEVEL%==0 (
    if exist "dashboard" (
        echo.
        echo Starting React dashboard...
        
        REM Install dependencies if needed
        if not exist "dashboard\node_modules" (
            echo Installing dashboard dependencies...
            cd dashboard
            call npm install -q
            cd ..
        )
        
        REM Start the React dev server
        cd dashboard
        start /b npm run dev
        cd ..
        
        timeout /t 3 >nul
        echo [OK] Dashboard running
        
        REM Open dashboard
        start http://localhost:5173
        echo [OK] Dashboard opened in browser
        
        echo.
        echo ════════════════════════════════════════════════════════════
        echo Pipeline complete! Dashboard is running.
        echo.
        echo Dashboard URL: http://localhost:5173
        echo.
        echo Press any key to stop and exit.
        echo ════════════════════════════════════════════════════════════
        
        pause >nul
        taskkill /f /im node.exe >nul 2>&1
    )
) else (
    REM Fallback to simple HTTP server
    for /f "tokens=5" %%a in ('netstat -aon ^| findstr :9000 ^| findstr LISTENING') do (
        taskkill /F /PID %%a >nul 2>&1
    )
    
    echo.
    echo Starting local server on port 9000...
    start /b python -m http.server 9000 >nul 2>&1
    timeout /t 1 >nul
    echo [OK] Server running
    
    start http://localhost:9000/dashboard/
    echo [OK] Dashboard opened in browser
    
    echo.
    echo ════════════════════════════════════════════════════════════
    echo Pipeline complete!
    echo.
    echo Dashboard URL: http://localhost:9000/dashboard/
    echo.
    echo Press any key to stop the server and exit.
    echo ════════════════════════════════════════════════════════════
    
    pause >nul
    
    for /f "tokens=5" %%a in ('netstat -aon ^| findstr :9000 ^| findstr LISTENING') do (
        taskkill /F /PID %%a >nul 2>&1
    )
)
