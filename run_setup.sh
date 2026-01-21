#!/bin/bash
# LVMH Voice-to-Tag Pipeline - Setup and Run Script
# Works on macOS, Linux, and Windows (Git Bash/WSL)

set -e  # Exit on error

echo "============================================================"
echo "LVMH Voice-to-Tag Pipeline - Setup"
echo "============================================================"
echo

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo "ERROR: Python is not installed or not in PATH"
        echo "Please install Python 3.9+"
        exit 1
    fi
    PYTHON=python
else
    PYTHON=python3
fi

echo "Using Python: $($PYTHON --version)"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    $PYTHON -m venv .venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f ".venv/Scripts/activate" ]; then
    source .venv/Scripts/activate
else
    echo "ERROR: Could not find activation script"
    exit 1
fi

# Upgrade pip and install requirements
echo "Installing requirements..."
pip install --upgrade pip
pip install -r requirements.txt

# Download NLTK data
echo "Downloading NLTK data..."
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('punkt_tab')"

# Download embedding model
echo "Downloading embedding model (this may take a few minutes)..."
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', cache_folder='models/sentence_transformers')"

echo
echo "============================================================"
echo "Setup complete! Running pipeline..."
echo "============================================================"
echo

# Run the pipeline
python -m src.run_all

echo
echo "============================================================"
echo "Pipeline finished!"
echo "============================================================"
echo
echo "To view the 3D visualization, open:"
echo "  demo/embedding_space_3d.html"
