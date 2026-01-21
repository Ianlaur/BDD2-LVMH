@echo off
REM LVMH Voice-to-Tag Pipeline - Windows Launcher
REM Double-click this file to run

cd /d "%~dp0"

echo ============================================================
echo LVMH Voice-to-Tag Pipeline
echo ============================================================
echo.
echo Select an option:
echo   1) Full setup + run (first time)
echo   2) Run pipeline only (already set up)
echo   3) View 3D visualization
echo   4) Clean outputs and re-run
echo   5) Exit
echo.
set /p choice="Enter choice [1-5]: "

if "%choice%"=="1" goto fullsetup
if "%choice%"=="2" goto runonly
if "%choice%"=="3" goto viewviz
if "%choice%"=="4" goto cleanrun
if "%choice%"=="5" goto end
echo Invalid choice
goto end

:fullsetup
echo.
echo Running full setup...

if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)

call .venv\Scripts\activate.bat

echo Installing requirements...
pip install --upgrade pip
pip install -r requirements.txt

echo Downloading NLTK data...
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('punkt_tab')"

echo Downloading embedding model...
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', cache_folder='models/sentence_transformers')"

echo.
echo Running pipeline...
python -m src.run_all

echo.
set /p openviz="Open 3D visualization? [y/n]: "
if "%openviz%"=="y" start demo\embedding_space_3d.html
goto end

:runonly
echo.
call .venv\Scripts\activate.bat
echo Running pipeline...
python -m src.run_all

echo.
set /p openviz="Open 3D visualization? [y/n]: "
if "%openviz%"=="y" start demo\embedding_space_3d.html
goto end

:viewviz
echo Opening 3D visualization...
start demo\embedding_space_3d.html
goto end

:cleanrun
echo Cleaning outputs...
if exist data\processed rd /s /q data\processed
if exist data\outputs rd /s /q data\outputs
if exist taxonomy rd /s /q taxonomy
if exist demo rd /s /q demo
mkdir data\processed data\outputs taxonomy demo

call .venv\Scripts\activate.bat
echo Running pipeline...
python -m src.run_all

echo.
set /p openviz="Open 3D visualization? [y/n]: "
if "%openviz%"=="y" start demo\embedding_space_3d.html
goto end

:end
echo.
echo Done!
pause
