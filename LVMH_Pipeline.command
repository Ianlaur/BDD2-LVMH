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

# Run the pipeline
echo
echo -e "${BLUE}Running pipeline...${NC}"
echo "------------------------------------------------------------"
python -m server.run_all
echo "------------------------------------------------------------"
echo -e "${GREEN}✓ Pipeline complete${NC}"

# Kill any existing server on port 9000
lsof -ti:9000 | xargs kill -9 2>/dev/null

# Start local server in background
echo
echo -e "${BLUE}Starting local server on port 9000...${NC}"
python3 -m http.server 9000 &>/dev/null &
SERVER_PID=$!
sleep 1
echo -e "${GREEN}✓ Server running (PID: $SERVER_PID)${NC}"

# Open dashboard
echo
echo -e "${BLUE}Opening dashboard...${NC}"
open "http://localhost:9000/client/app/dashboard.html"
echo -e "${GREEN}✓ Dashboard opened in browser${NC}"

echo
echo -e "${GOLD}════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Pipeline complete! Dashboard is now running.${NC}"
echo
echo "Dashboard URL: http://localhost:9000/client/app/dashboard.html"
echo
echo "Press Ctrl+C to stop the server, or close this window."
echo -e "${GOLD}════════════════════════════════════════════════════════════${NC}"

# Keep server running until user closes
wait $SERVER_PID
