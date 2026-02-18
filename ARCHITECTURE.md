# Project Architecture

## Overview
This project follows a clean client/server architecture:
- **server/** - All backend processing (data pipeline, NLP, visualizations)
- **client/app/** - Frontend dashboard and user interface

## Directory Structure

```
BDD2-LVMH/
├── server/                  # Backend processing
│   ├── __init__.py
│   ├── run_all.py          # Main pipeline orchestrator
│   ├── ingest/             # Data ingestion
│   ├── extract/            # Concept extraction
│   ├── embeddings/         # Vector embeddings & UMAP
│   ├── lexicon/            # Lexicon building
│   ├── profiling/          # Client segmentation
│   ├── actions/            # Action recommendations
│   └── shared/             # Shared utilities & config
│       ├── config.py       # Global configuration
│       ├── utils.py        # Helper functions
│       ├── knowledge_graph.py
│       └── generate_dashboard.py
│
├── client/                  # Frontend
│   └── app/                # Dashboard application
│       ├── dashboard.html  # Main unified dashboard
│       ├── kg_obsidian.html
│       ├── embedding_space_3d.html
│       └── cytoscape.min.js
│
├── data/                    # Data files
│   ├── input/              # Raw input data
│   ├── processed/          # Processed intermediates
│   └── outputs/            # Final outputs
│
├── taxonomy/                # Taxonomy data
├── activations/             # Activation files
├── models/                  # ML models cache
│
├── LVMH_Pipeline.command   # macOS launcher
├── main.py                  # Simple entry point
└── requirements.txt         # Python dependencies
```

## Running the Pipeline

### Option 1: Using the launcher (macOS)
Double-click `LVMH_Pipeline.command` in Finder

### Option 2: Command line
```bash
# Activate virtual environment
source .venv/bin/activate

# Run the pipeline
python -m server.run_all
```

### Option 3: Open dashboard directly
```bash
open client/app/dashboard.html
```

## Key Changes from Original Structure

1. **Centralized backend**: All processing code moved from `src/` to `server/`
2. **Clean frontend separation**: Dashboard files in `client/app/`
3. **Updated imports**: All `from src.` changed to `from server.`
4. **Output paths**: Visualizations now generate to `client/app/`
5. **Config updates**: `DEMO_DIR` points to `client/app/`

## Import Structure

All backend modules use the `server.` namespace:
```python
from server.shared.config import DATA_DIR, DEMO_DIR
from server.shared.utils import log_stage
from server.ingest.run_ingest import run_ingest
from server.embeddings.build_vectors import build_vectors
```

## Configuration

Global settings in `server/shared/config.py`:
- `BASE_DIR`: Project root
- `DATA_DIR`: Data directory
- `DEMO_DIR`: Points to `client/app/` (frontend)
- `TAXONOMY_DIR`: Taxonomy files
- Model paths, seeds, clustering parameters
