# Deploying LVMH Server to Render

## Prerequisites
- GitHub account
- Render account (free tier works)
- Your code pushed to a GitHub repository

## Step 1: Prepare Your Repository

Make sure these files are in your repository:
- `requirements.txt` - Python dependencies
- `render.yaml` - Render configuration (already created)
- `server/` directory with all your code

## Step 2: Push to GitHub

```bash
cd /Users/ian/Desktop/BDD2-LVMH

# Initialize git if not already done
git init
git add .
git commit -m "Prepare for Render deployment"

# Add your GitHub repository
git remote add origin https://github.com/YOUR_USERNAME/BDD2-LVMH.git
git push -u origin main
```

## Step 3: Create Render Service

1. Go to https://render.com and sign in
2. Click "New +" → "Blueprint"
3. Connect your GitHub repository
4. Render will detect the `render.yaml` file
5. Click "Apply" to create the service

## Step 4: Configure Environment (Optional)

In Render dashboard, you can add environment variables:
- `CLUSTERS_K` - Number of segments (default: auto)
- `ENABLE_ANONYMIZATION` - Set to "false" to disable RGPD (default: true)

## Step 5: Wait for Build

The first build takes ~10-15 minutes because it:
- Installs all Python packages
- Downloads the 400MB multilingual model
- Runs initial setup

Watch the logs in Render dashboard.

## Step 6: Get Your Public URL

Once deployed, Render gives you a URL like:
```
https://lvmh-api-server.onrender.com
```

## Step 7: Update Dashboard

On your dashboard Mac, update `.env`:

```bash
VITE_API_URL=https://lvmh-api-server.onrender.com
```

No port number needed - Render handles that!

## Step 8: Test

Visit your Render URL:
```
https://lvmh-api-server.onrender.com
```

You should see:
```json
{
  "service": "LVMH Client Intelligence API",
  "status": "running",
  "version": "1.0.0"
}
```

## Important Notes

### Free Tier Limitations
- Spins down after 15 minutes of inactivity
- First request after spin-down takes 30-60 seconds
- 750 hours/month free compute

### Model Persistence
The `render.yaml` configures a 2GB persistent disk for the ML model cache, so it won't re-download on every deploy.

### Data Persistence
Uploaded CSVs and generated outputs are stored on the disk. They persist across deploys but **not** if you delete the service.

### CORS
The API allows all origins (`allow_origins=["*"]`). For production, update `server/api_server.py` line 37 to specify your dashboard URL:

```python
allow_origins=["https://your-dashboard.vercel.app"],
```

## Troubleshooting

### Build fails with memory error
The free tier has 512MB RAM. If the model download fails:
1. Upgrade to Starter plan ($7/month, 512MB → 2GB RAM)
2. Or deploy just the API without running the pipeline on Render

### Service keeps spinning down
Upgrade to paid plan for always-on service, or use a service like UptimeRobot to ping it every 10 minutes.

### Can't connect from dashboard
Check CORS settings and make sure the URL in `.env` matches exactly (no trailing slash).

## Alternative: Deploy Dashboard to Vercel

You can also deploy the dashboard:

```bash
cd dashboard
npm install -g vercel
vercel
```

Then both server and dashboard are cloud-hosted!

## Local Development

To test locally before deploying:
```bash
cd /Users/ian/Desktop/BDD2-LVMH
source .venv/bin/activate
python -m server.api_server
```

The server reads `PORT` from environment, so it works both locally and on Render.
