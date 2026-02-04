#!/bin/bash
# Setup script for LVMH Server Mac
# This installs all dependencies and downloads the ML model

set -e  # Exit on error

echo "=================================================="
echo "LVMH Server Setup"
echo "=================================================="
echo ""

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "Working directory: $SCRIPT_DIR"
echo ""

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

echo ""
echo "Activating virtual environment..."
source .venv/bin/activate

echo ""
echo "Upgrading pip..."
pip install --upgrade pip

echo ""
echo "Installing Python dependencies..."
pip install -r requirements.txt

echo ""
echo "Creating necessary directories..."
mkdir -p data/input
mkdir -p data/processed
mkdir -p data/outputs
mkdir -p taxonomy
mkdir -p models/sentence_transformers

echo ""
echo "Downloading sentence transformer model..."
echo "This may take a few minutes (400MB download)..."
python3 << 'PYTHON_SCRIPT'
import os
from sentence_transformers import SentenceTransformer

# Set cache directory
cache_dir = os.path.join(os.getcwd(), "models", "sentence_transformers")
os.makedirs(cache_dir, exist_ok=True)
os.environ['SENTENCE_TRANSFORMERS_HOME'] = cache_dir

# Download the model
model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
print(f"Downloading {model_name}...")
model = SentenceTransformer(model_name)
print(f"✓ Model downloaded and cached in {cache_dir}")
PYTHON_SCRIPT

echo ""
echo "Testing NLTK downloads..."
python3 << 'PYTHON_SCRIPT'
import nltk
import os

# Create nltk_data directory in project
nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.insert(0, nltk_data_dir)

# Download required NLTK data
try:
    nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)
    nltk.download('stopwords', download_dir=nltk_data_dir, quiet=True)
    nltk.download('wordnet', download_dir=nltk_data_dir, quiet=True)
    print("✓ NLTK data downloaded")
except Exception as e:
    print(f"Warning: NLTK download issue (may work anyway): {e}")
PYTHON_SCRIPT

echo ""
echo "=================================================="
echo "Setup Complete!"
echo "=================================================="
echo ""
echo "To start the server, run:"
echo "  ./start-server.sh"
echo ""
echo "Or manually:"
echo "  source .venv/bin/activate"
echo "  python server/api_server.py"
echo ""
