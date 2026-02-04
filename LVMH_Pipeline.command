#!/bin/bash
# LVMH Voice-to-Tag Pipeline - macOS Double-Click Launcher
# Double-click this file in Finder to run the full pipeline and open dashboard

# Change to the script's directory
cd "$(dirname "$0")"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
GOLD='\033[0;33m'
NC='\033[0m'

<<<<<<< Updated upstream
echo
echo -e "${GOLD}════════════════════════════════════════════════════════════${NC}"
echo -e "${GOLD}   LVMH Client Intelligence Pipeline${NC}"
echo -e "${GOLD}════════════════════════════════════════════════════════════${NC}"
echo

# Check/create virtual environment
if [ ! -d ".venv" ]; then
    echo -e "${BLUE}Creating virtual environment...${NC}"
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip -q
    pip install -r requirements.txt -q
    echo -e "${GREEN}✓ Environment ready${NC}"
else
    source .venv/bin/activate
    echo -e "${GREEN}✓ Virtual environment activated${NC}"
fi
=======
echo "============================================================"
echo "LVMH Voice-to-Tag Pipeline"
echo "============================================================"
echo ""
echo "Select an option:"
echo "  1) Full setup + run (first time)"
echo "  2) Run pipeline only (already set up)"
echo "  3) View dashboard"
echo "  4) Clean outputs and re-run"
echo "  5) Exit"
echo ""
printf "Enter choice [1-5]: "
read choice

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
        python -m server.run_all
        
        echo ""
        printf "Open dashboard? [y/n]: "
        read open_viz
        if [ "$open_viz" = "y" ]; then
            open client/app/dashboard.html
        fi
        ;;
    2)
        echo ""
        source .venv/bin/activate
        echo "Running pipeline..."
        python -m server.run_all
        
        echo ""
        printf "Open dashboard? [y/n]: "
        read open_viz
        if [ "$open_viz" = "y" ]; then
            open client/app/dashboard.html
        fi
        ;;
    3)
        echo "Opening dashboard..."
        open client/app/dashboard.html
        ;;
    4)
        echo "Cleaning outputs..."
        rm -rf data/processed/* data/outputs/* taxonomy/* client/app/*
        
        source .venv/bin/activate
        echo "Running pipeline..."
        python -m server.run_all
        
        echo ""
        printf "Open dashboard? [y/n]: "
        read open_viz
        if [ "$open_viz" = "y" ]; then
            open client/app/dashboard.html
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
>>>>>>> Stashed changes

# Run the pipeline
echo
echo -e "${BLUE}Running pipeline...${NC}"
echo "------------------------------------------------------------"
python -m server.run_all
echo "------------------------------------------------------------"
echo -e "${GREEN}✓ Pipeline complete${NC}"

# Check if npm is installed and dashboard has dependencies
if command -v npm &> /dev/null && [ -d "dashboard" ]; then
    echo
    echo -e "${BLUE}Starting React dashboard...${NC}"
    
    # Install dependencies if needed
    if [ ! -d "dashboard/node_modules" ]; then
        echo "Installing dashboard dependencies..."
        cd dashboard && npm install -q && cd ..
    fi
    
    # Start the React dev server
    cd dashboard
    npm run dev &
    DASHBOARD_PID=$!
    cd ..
    
    sleep 3
    echo -e "${GREEN}✓ Dashboard running${NC}"
    
    # Open dashboard
    open "http://localhost:5173"
    echo -e "${GREEN}✓ Dashboard opened in browser${NC}"
    
    echo
    echo -e "${GOLD}════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}Pipeline complete! Dashboard is running.${NC}"
    echo
    echo "Dashboard URL: http://localhost:5173"
    echo
    echo "Press Ctrl+C to stop, or close this window."
    echo -e "${GOLD}════════════════════════════════════════════════════════════${NC}"
    
    wait $DASHBOARD_PID
else
    # Fallback to simple HTTP server if no npm
    lsof -ti:9000 | xargs kill -9 2>/dev/null
    
    echo
    echo -e "${BLUE}Starting local server on port 9000...${NC}"
    python3 -m http.server 9000 &>/dev/null &
    SERVER_PID=$!
    sleep 1
    echo -e "${GREEN}✓ Server running (PID: $SERVER_PID)${NC}"
    
    open "http://localhost:9000/dashboard/"
    echo -e "${GREEN}✓ Dashboard opened in browser${NC}"
    
    echo
    echo -e "${GOLD}════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}Pipeline complete!${NC}"
    echo
    echo "Dashboard URL: http://localhost:9000/dashboard/"
    echo
    echo "Press Ctrl+C to stop the server, or close this window."
    echo -e "${GOLD}════════════════════════════════════════════════════════════${NC}"
    
    wait $SERVER_PID
fi
