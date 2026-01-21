#!/bin/bash
# LVMH Voice-to-Tag Pipeline - macOS Double-Click Launcher
# Double-click this file in Finder to run the pipeline

# Change to the script's directory
cd "$(dirname "$0")"

echo "============================================================"
echo "LVMH Voice-to-Tag Pipeline"
echo "============================================================"
echo
echo "Select an option:"
echo "  1) Full setup + run (first time)"
echo "  2) Run pipeline only (already set up)"
echo "  3) View 3D visualization"
echo "  4) Clean outputs and re-run"
echo "  5) Exit"
echo
read -p "Enter choice [1-5]: " choice

case $choice in
    1)
        echo
        echo "Running full setup..."
        
        # Create venv if needed
        if [ ! -d ".venv" ]; then
            echo "Creating virtual environment..."
            python3 -m venv .venv
        fi
        
        source .venv/bin/activate
        
        echo "Installing requirements..."
        pip install --upgrade pip
        pip install -r requirements.txt
        
        echo "Downloading NLTK data..."
        python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('punkt_tab')"
        
        echo "Downloading embedding model..."
        python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', cache_folder='models/sentence_transformers')"
        
        echo
        echo "Running pipeline..."
        python -m src.run_all
        
        echo
        read -p "Open 3D visualization? [y/n]: " open_viz
        if [ "$open_viz" = "y" ]; then
            open demo/embedding_space_3d.html
        fi
        ;;
    2)
        echo
        source .venv/bin/activate
        echo "Running pipeline..."
        python -m src.run_all
        
        echo
        read -p "Open 3D visualization? [y/n]: " open_viz
        if [ "$open_viz" = "y" ]; then
            open demo/embedding_space_3d.html
        fi
        ;;
    3)
        echo "Opening 3D visualization..."
        open demo/embedding_space_3d.html
        ;;
    4)
        echo "Cleaning outputs..."
        rm -rf data/processed/* data/outputs/* taxonomy/* demo/*
        
        source .venv/bin/activate
        echo "Running pipeline..."
        python -m src.run_all
        
        echo
        read -p "Open 3D visualization? [y/n]: " open_viz
        if [ "$open_viz" = "y" ]; then
            open demo/embedding_space_3d.html
        fi
        ;;
    5)
        echo "Goodbye!"
        exit 0
        ;;
    *)
        echo "Invalid choice"
        ;;
esac

echo
echo "Done! Press any key to close..."
read -n 1
