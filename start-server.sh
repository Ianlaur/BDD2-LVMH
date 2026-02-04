#!/bin/bash
# Quick server startup script
# Usage: ./start-server.sh

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# Activate virtual environment and start server
source .venv/bin/activate
python -m server.api_server --host 0.0.0.0 --port 8000
