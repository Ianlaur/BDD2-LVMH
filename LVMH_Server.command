#!/bin/bash
# LVMH API Server Launcher for macOS
# Double-click this file to start the server

# Get the directory where this script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

echo "========================================="
echo "LVMH Voice-to-Tag API Server"
echo "========================================="
echo ""

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found!"
    echo ""
    echo "Please run setup first:"
    echo "  make venv"
    echo "  make setup-models-local"
    echo ""
    read -p "Press Enter to exit..."
    exit 1
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  No .env file found!"
    if [ -f ".env.example" ]; then
        echo "📋 Copying .env.example → .env"
        cp .env.example .env
        echo "✅ .env created from template"
    else
        echo "❌ No .env or .env.example found!"
        echo "   The server needs DATABASE_URL to connect to Neon PostgreSQL."
        echo "   Create a .env file with:"
        echo "     DATABASE_URL=postgresql://user:pass@host/dbname?sslmode=require"
        echo ""
        read -p "Press Enter to exit..."
        exit 1
    fi
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Check if required packages are installed
MISSING=0
for pkg in fastapi uvicorn asyncpg psycopg2 dotenv; do
    if ! python -c "import $pkg" 2>/dev/null; then
        MISSING=1
        break
    fi
done

if [ $MISSING -eq 1 ]; then
    echo "📦 Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install dependencies."
        echo "   Try running: pip install -r requirements.txt"
        read -p "Press Enter to exit..."
        exit 1
    fi
    echo "✅ Dependencies installed"
fi

# Get local IP address
LOCAL_IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo "localhost")

echo ""
echo "========================================="
echo "🚀 Starting LVMH API Server"
echo "========================================="
echo ""
echo "Server will be accessible at:"
echo "  📱 Local:   http://localhost:8000"
echo "  🌐 Network: http://$LOCAL_IP:8000"
echo ""
echo "Dashboard should connect to:"
echo "  VITE_API_URL=http://$LOCAL_IP:8000"
echo ""
echo "API Endpoints:"
echo "  GET  /                      - API info"
echo "  GET  /api/data              - Dashboard data"
echo "  GET  /api/pipeline/status   - Pipeline status"
echo "  POST /api/pipeline/run      - Run pipeline"
echo "  GET  /api/lexicon           - Vocabulary"
echo "  GET  /api/knowledge-graph   - Knowledge graph"
echo ""
echo "Press Ctrl+C to stop the server"
echo "========================================="
echo ""

# Run the API server
python -m server.api_server --host 0.0.0.0 --port 8000

# Keep terminal open on exit
echo ""
echo "Server stopped."
read -p "Press Enter to close..."
