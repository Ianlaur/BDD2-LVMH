#!/bin/bash
# Quick server startup script
# Usage: ./start-server.sh

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# Kill any existing server on port 8000
lsof -ti:8000 | xargs kill -9 2>/dev/null
sleep 1

# Check for .env file (contains DATABASE_URL)
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        echo "⚠️  No .env found — copying .env.example → .env"
        cp .env.example .env
    else
        echo "⚠️  No .env file — server will run in file-only mode (no database)"
    fi
fi

# Activate virtual environment and install deps if needed
source .venv/bin/activate

# Quick check for critical packages
if ! python -c "import asyncpg, psycopg2, fastapi" 2>/dev/null; then
    echo "📦 Installing missing dependencies..."
    pip install -r requirements.txt
fi

python -m server.api_server --host 0.0.0.0 --port 8000 --reload
