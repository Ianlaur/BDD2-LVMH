#!/bin/bash
# Quick server startup script
# Usage: ./start-server.sh

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# Kill any existing server on port 8000
lsof -ti:8000 | xargs kill -9 2>/dev/null
sleep 1

# Activate virtual environment and start server (with auto-reload for dev)
source .venv/bin/activate
python -m server.api_server --host 0.0.0.0 --port 8000 --reload
