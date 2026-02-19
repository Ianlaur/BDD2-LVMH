@echo off
REM LVMH Client Intelligence Pipeline - Windows Setup and Run Script
REM Run this in Command Prompt or PowerShell

echo ============================================================
echo LVMH Client Intelligence Pipeline - Windows Setup
echo ============================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.9+ from https://python.org
    pause
    exit /b 1
)

REM Create virtual environment if it doesn't exist
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)

REM Activate virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate.bat

REM Upgrade pip and install requirements
echo Installing requirements...
pip install --upgrade pip
pip install -r requirements.txt

REM Download NLTK data
echo Downloading NLTK data...
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('punkt_tab')"

REM Download embedding model
echo Downloading embedding model (this may take a few minutes)...
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', cache_folder='models/sentence_transformers')"

echo.
echo ============================================================
echo Setup complete! Running pipeline...
echo ============================================================
echo.

REM Run the pipeline
python -m src.run_all

echo.
echo ============================================================
echo Pipeline finished!
echo ============================================================
echo.
echo To view the 3D visualization, open:
echo   demo\embedding_space_3d.html
echo.
pause
