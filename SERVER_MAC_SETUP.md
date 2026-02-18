# LVMH Server Mac Setup Instructions

## Prerequisites
- macOS with Python 3.10 or higher
- At least 2GB free disk space for models
- Network connection for downloads

## Step 1: Transfer Project to Server Mac

Copy the entire BDD2-LVMH folder to your server Mac, or clone it from git:

```bash
# If using git
cd ~/Desktop
git clone <your-repo-url> BDD2-LVMH

# Or use AirDrop/USB to copy the folder
```

## Step 2: Run Setup Script

On the **server Mac**, open Terminal and run:

```bash
cd ~/Desktop/BDD2-LVMH  # or wherever you put the project
./setup_server.sh
```

This will:
- Create a Python virtual environment
- Install all dependencies (pandas, scikit-learn, sentence-transformers, etc.)
- Download the multilingual NLP model (~400MB)
- Set up NLTK data
- Create necessary directories

**Note:** This may take 5-10 minutes depending on internet speed.

## Step 3: Start the Server

After setup completes, start the API server:

```bash
./start-server.sh
```

Or manually:

```bash
source .venv/bin/activate
python server/api_server.py
```

You should see:
```
INFO:     Started server process [xxxxx]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

## Step 4: Get Server IP Address

Find your server Mac's IP address:

```bash
ifconfig | grep "inet " | grep -v 127.0.0.1
```

Look for the line with `inet 10.x.x.x` or `inet 192.168.x.x`

## Step 5: Configure Firewall (Important!)

The dashboard Mac needs to connect to port 8000 on the server.

### Option A: Allow Python through firewall
1. Open System Settings → Network → Firewall
2. Click "Options"
3. Add Python to allowed apps

### Option B: Temporarily disable firewall (testing only)
```bash
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --setglobalstate off
```

To re-enable:
```bash
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --setglobalstate on
```

## Step 6: Verify Server is Working

From any terminal on the server Mac:

```bash
curl http://localhost:8000/api/data
```

You should see JSON data about segments and clients.

## Step 7: Connect Dashboard

On your **dashboard Mac**, update `.env` file in the dashboard folder:

```bash
VITE_API_URL=http://YOUR_SERVER_IP:8000
```

Replace `YOUR_SERVER_IP` with the IP from Step 4.

Restart the dashboard dev server.

## Troubleshooting

### "Module not found" errors
Make sure virtual environment is activated:
```bash
source .venv/bin/activate
```

### Can't reach server from dashboard Mac
1. Verify both Macs are on the same WiFi network
2. Check firewall settings (Step 5)
3. Test with ping: `ping YOUR_SERVER_IP`

### Model download fails
If the automatic model download fails, manually download:
```bash
source .venv/bin/activate
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"
```

### Server crashes when running pipeline
Ensure you have at least 4GB RAM available. The ML models require memory.

## Starting Server on System Boot (Optional)

To make the server start automatically:

1. Create a LaunchAgent plist file
2. Or add to your `.zshrc`:
```bash
alias lvmh-start="cd ~/Desktop/BDD2-LVMH && ./start-server.sh"
```

## Stopping the Server

Press `Ctrl+C` in the terminal where the server is running.

## Data Files Location

- Input data: `data/input/`
- Processed data: `data/processed/`
- Output files: `data/outputs/`
- Taxonomy: `taxonomy/`

## Running the Pipeline Manually

To process new data:

```bash
source .venv/bin/activate
python -m server.run_all
```

Or trigger via API from dashboard (Upload CSV feature).
