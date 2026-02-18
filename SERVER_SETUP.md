# Server Setup Guide - Running on Separate Mac

This guide explains how to set up and run the LVMH pipeline server on a dedicated Mac.

## 🖥️ Server Mac Setup

### 1. Initial Setup (One-Time)

```bash
# 1. Clone or copy the project to your server Mac
cd ~/Desktop
# (copy BDD2-LVMH folder here)

# 2. Navigate to project directory
cd BDD2-LVMH

# 3. Create virtual environment
make venv
# or manually:
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 4. Download ML models (required for embeddings)
make setup-models-local
# This downloads the sentence transformer model (~100MB)

# 5. Prepare data directory
mkdir -p data/input data/processed data/outputs taxonomy

# 6. Copy your CSV data file to data/input/
cp /path/to/your/LVMH_data.csv data/input/
```

### 2. Starting the Server

**Option A: Double-click launcher (easiest)**
```bash
# Just double-click this file in Finder:
LVMH_Server.command
```

**Option B: Terminal**
```bash
cd ~/Desktop/BDD2-LVMH
source .venv/bin/activate
python -m server.api_server --host 0.0.0.0 --port 8000
```

The server will start and show you:
```
🚀 Starting LVMH API Server
Server will be accessible at:
  📱 Local:   http://localhost:8000
  🌐 Network: http://192.168.1.XXX:8000
```

**Important:** Note the Network IP address - you'll need it for the dashboard Mac!

### 3. Finding Your Server's IP Address

If you need to find the IP manually:

```bash
# macOS
ipconfig getifaddr en0
# or
ifconfig | grep "inet " | grep -v 127.0.0.1
```

Common formats:
- `192.168.1.XXX` (home network)
- `10.0.0.XXX` (some routers)
- `172.16.X.XXX` (corporate networks)

### 4. Running the Pipeline

The pipeline processes your CSV data and generates insights.

**Option 1: Via API (from dashboard or curl)**
```bash
# From any machine on the network:
curl -X POST http://192.168.1.XXX:8000/api/pipeline/run

# Check status:
curl http://192.168.1.XXX:8000/api/pipeline/status
```

**Option 2: Directly on server**
```bash
cd ~/Desktop/BDD2-LVMH
source .venv/bin/activate
python -m server.run_all
```

### 5. Verify Server is Working

Open in browser on server Mac:
```
http://localhost:8000
```

You should see:
```json
{
  "service": "LVMH Voice-to-Tag API",
  "status": "running",
  "version": "1.0.0"
}
```

## 📱 Dashboard Mac Setup

### 1. Configure Dashboard to Connect to Server

```bash
cd ~/Desktop/BDD2-LVMH/dashboard

# Edit .env file (or create it)
echo "VITE_API_URL=http://192.168.1.XXX:8000" > .env
# Replace 192.168.1.XXX with your server's IP!
```

### 2. Start Dashboard

```bash
# Install dependencies (first time only)
npm install

# Start development server
npm run dev
```

Open browser to: `http://localhost:5173`

The dashboard will now fetch data from your server Mac!

## 🔧 Configuration

### Environment Variables (Server)

Create a `.env` file on the server Mac:

```bash
# Optional: Disable anonymization
ENABLE_ANONYMIZATION=true

# Optional: Aggressive name detection
ANONYMIZATION_AGGRESSIVE=false

# Optional: Custom number of clusters
CLUSTERS_K=8
```

### Firewall Settings

Make sure port 8000 is open on the server Mac:

**macOS System Preferences:**
1. System Preferences → Security & Privacy → Firewall
2. Click "Firewall Options"
3. Ensure Python is allowed to accept incoming connections

## 📊 Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                     TYPICAL WORKFLOW                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  SERVER MAC:                                                │
│  1. Start server: ./LVMH_Server.command                     │
│  2. Copy CSV to: data/input/                                │
│  3. Run pipeline: python -m server.run_all                  │
│     (or trigger via API)                                    │
│                                                              │
│  DASHBOARD MAC:                                             │
│  1. Configure: dashboard/.env → VITE_API_URL=http://...     │
│  2. Start dashboard: npm run dev                            │
│  3. Open: http://localhost:5173                             │
│  4. View results!                                           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## 🔍 Testing the Connection

### From Dashboard Mac:

```bash
# Test if you can reach the server
curl http://192.168.1.XXX:8000/

# Get dashboard data
curl http://192.168.1.XXX:8000/api/data

# Check pipeline status
curl http://192.168.1.XXX:8000/api/pipeline/status
```

### Common Issues:

**❌ "Connection refused"**
- Server not running → Start LVMH_Server.command
- Wrong IP address → Check `ipconfig getifaddr en0`
- Firewall blocking → Allow Python in firewall settings

**❌ "404 Not Found on /api/data"**
- Pipeline hasn't run yet → Run `python -m server.run_all` first
- No data.json generated → Check data/outputs/ folder

**❌ Dashboard shows "Using local data"**
- Can't reach server → Check IP and port
- Falls back to local data.json if present

## 📁 What Needs to Be on Server Mac

**Required:**
- ✅ All server code (`server/` directory)
- ✅ Configuration (`server/shared/config.py`)
- ✅ Requirements (`requirements.txt`)
- ✅ Virtual environment (`.venv/`)
- ✅ ML models (`models/sentence_transformers/`)
- ✅ Input data (`data/input/*.csv`)
- ✅ Taxonomy files (`taxonomy/` - optional, auto-generated)

**NOT needed:**
- ❌ Dashboard code (`dashboard/` - only on dashboard Mac)
- ❌ Node.js / npm (only on dashboard Mac)
- ❌ Old `src/` directory (deprecated)

## 🚀 Production Tips

### Keep Server Running (Background)

```bash
# Use nohup to keep server running after closing terminal
nohup python -m server.api_server --host 0.0.0.0 --port 8000 > server.log 2>&1 &

# Check if running
ps aux | grep api_server

# Stop server
pkill -f api_server
```

### Auto-start on Boot (Optional)

Create a LaunchAgent to start server automatically when Mac boots:

```bash
# Create plist file
nano ~/Library/LaunchAgents/com.lvmh.apiserver.plist
```

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.lvmh.apiserver</string>
    <key>ProgramArguments</key>
    <array>
        <string>/Users/YOURUSERNAME/Desktop/BDD2-LVMH/.venv/bin/python</string>
        <string>-m</string>
        <string>server.api_server</string>
        <string>--host</string>
        <string>0.0.0.0</string>
        <string>--port</string>
        <string>8000</string>
    </array>
    <key>WorkingDirectory</key>
    <string>/Users/YOURUSERNAME/Desktop/BDD2-LVMH</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/Users/YOURUSERNAME/Desktop/BDD2-LVMH/server.log</string>
    <key>StandardErrorPath</key>
    <string>/Users/YOURUSERNAME/Desktop/BDD2-LVMH/server.error.log</string>
</dict>
</plist>
```

```bash
# Load the service
launchctl load ~/Library/LaunchAgents/com.lvmh.apiserver.plist

# Start the service
launchctl start com.lvmh.apiserver

# Check status
launchctl list | grep lvmh
```

## 📞 Support

For issues:
1. Check server logs: `tail -f server.log`
2. Check dashboard console in browser
3. Verify network connectivity: `ping 192.168.1.XXX`
4. Ensure firewall allows connections
5. Restart both server and dashboard

---

**Ready to go!** 🚀

Start the server on your dedicated Mac, configure the dashboard Mac to point to it, and enjoy separated, scalable architecture!
