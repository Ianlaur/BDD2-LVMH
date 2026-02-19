# Project Architecture — Production v1.0

## Overview
This project follows a modern full-stack architecture:
- **server/** - Backend processing (hybrid NLP pipeline, ML, API)
- **dashboard/** - React/TypeScript frontend (Vite)
- **Database** - Neon PostgreSQL (cloud-hosted)
- **LLM** - Ollama with Qwen2.5:3b for semantic enhancement

## Technology Stack

### Backend
- **Python 3.14+** - Core language
- **FastAPI** - REST API server (port 8000)
- **PostgreSQL** - Neon cloud database
- **Ollama** - Local LLM server (Qwen2.5:3b)
- **SentenceTransformers** - Multilingual embeddings (384d)
- **scikit-learn** - ML models (clustering, predictions)

### Frontend
- **React** - UI library
- **TypeScript** - Type-safe JavaScript
- **Vite** - Build tool & dev server
- **TailwindCSS** - Utility-first CSS
- **Plotly** - 3D visualizations

### Extraction Methods
1. **Aho-Corasick** - Multi-pattern matching (O(N+M))
2. **Regex** - Budget/amount extraction
3. **Qwen LLM** - Semantic concept enhancement

## Directory Structure

```
BDD2-LVMH/
├── server/                  # Backend processing
│   ├── __init__.py
│   ├── run_all.py          # Main pipeline orchestrator (10 stages)
│   ├── api_server.py       # FastAPI REST API
│   ├── ingest/             # Stage 1: Data ingestion
│   ├── extract/            # Stages 2-4: Concept extraction
│   │   ├── detect_concepts.py    # Rule-based (Aho-Corasick + Regex)
│   │   └── qwen_enhance.py       # LLM enhancement (Qwen2.5:3b)
│   ├── lexicon/            # Stage 3: Lexicon building
│   ├── privacy/            # GDPR anonymization
│   ├── embeddings/         # Stage 5: Vector embeddings
│   ├── profiling/          # Stage 6: Client segmentation
│   │   └── segment_clients.py    # KMeans + per-client tags
│   ├── actions/            # Stage 7: Action recommendations
│   ├── analytics/          # Stage 8: ML predictions
│   ├── db/                 # Database layer
│   │   ├── crud.py               # CRUD operations
│   │   ├── sync.py               # DB sync utilities
│   │   └── models.py             # SQLAlchemy models
│   ├── adaptive/           # Adaptive pipeline (any CSV)
│   └── shared/             # Shared utilities & config
│       ├── config.py       # Global configuration
│       ├── utils.py        # Helper functions
│       ├── knowledge_graph.py
│       └── generate_dashboard.py
│
├── dashboard/              # Frontend (React/TypeScript)
│   ├── src/
│   │   ├── App.tsx              # Main app component
│   │   ├── Client360Page.tsx    # Client detail view
│   │   ├── data.json            # Generated dashboard data
│   │   └── ...
│   ├── index.html
│   ├── package.json
│   ├── vite.config.ts
│   └── tsconfig.json
│
├── data/                    # Data files
│   ├── input/              # Raw input CSV files
│   ├── processed/          # Processed intermediates
│   ├── outputs/            # Final outputs
│   │   ├── note_concepts.csv           # 1,796 concepts
│   │   ├── client_profiles.csv         # 100 clients, 12 tags each
│   │   ├── ml_predictions.csv          # Purchase/Churn/CLV
│   │   └── client_summaries.json       # LLM summaries
│   └── qwen_checkpoints/   # LLM enhancement checkpoints
│
├── taxonomy/                # Taxonomy & vocabulary
│   ├── lexicon_v1.json     # 575 concepts, 9,003 aliases
│   ├── lexicon_v1.csv
│   └── taxonomy_v1.json    # Categorized concepts
│
├── tests/                   # Test suite
│   ├── test_big_o.py       # Big O complexity tests (11 tests)
│   ├── generate_big_o_graph.py
│   ├── results/            # Latest test results
│   └── archive/            # Historical test results
│
├── activations/             # Playbooks (YAML)
├── models/                  # ML models cache
│   └── sentence_transformers/
│
├── LVMH_Pipeline.command   # macOS launcher
├── start-server.sh         # Server startup script
├── main.py                  # Simple entry point
├── requirements.txt         # Python dependencies
├── .env                     # Environment variables
└── README.md                # Main documentation
```

## Running the Pipeline

### Option 1: Full Pipeline (10 Stages)
```bash
# Activate virtual environment
source .venv/bin/activate

# Run all stages
python -m server.run_all

# With specific CSV
python -m server.run_all --csv data/input/my_data.csv

# With Qwen enhancement (default: enabled)
ENABLE_QWEN_ENHANCEMENT=true python -m server.run_all

# Skip LLM enhancement (faster)
ENABLE_QWEN_ENHANCEMENT=false python -m server.run_all
```

### Option 2: Start API Server
```bash
# Start FastAPI server (port 8000)
./start-server.sh

# Or manually
source .venv/bin/activate
python -m uvicorn server.api_server:app --host 0.0.0.0 --port 8000
```

### Option 3: Start Dashboard
```bash
cd dashboard
npm install
npm run dev    # Development server (port 5173)
npm run build  # Production build
```

### Option 4: Using the launcher (macOS)
```bash
# Double-click or run:
./LVMH_Pipeline.command
```

## Pipeline Stages (Detailed)

### Stage 1: Data Ingestion
- **Input**: CSV file
- **Output**: `notes_clean.parquet`
- **Processing**: Load, validate, normalize dates/languages
- **Time**: ~0.5s

### Stage 2: Extract Candidates (Skip in v1.0)
- Replaced by direct lexicon-based extraction

### Stage 3: Lexicon Building
- **Input**: `taxonomy/vocabulary.json` (575 concepts)
- **Output**: `lexicon_v1.json`, `taxonomy_v1.json`
- **Processing**: Build alias map (9,003 aliases)
- **Time**: ~0.1s

### Stage 4: Concept Detection (Hybrid)
- **Input**: `notes_clean.parquet`, `lexicon_v1.json`
- **Output**: `note_concepts.csv` (1,796 concepts)
- **Processing**:
  1. **Rule-based** (Aho-Corasick): 1,454 concepts (~1ms/note)
  2. **Regex (Budget)**: 100 budget extractions (~1ms/note)
  3. **Qwen LLM**: 342 semantic concepts (~10s/note, optional)
- **Time**: ~15s (with LLM), ~1s (without LLM)

### Stage 5: Privacy/Anonymization (GDPR)
- **Input**: `notes_clean.parquet`
- **Output**: Anonymized text
- **Processing**: Remove PII (names, emails, phones, addresses)
- **Time**: ~2s

### Stage 6: Vector Building
- **Input**: `notes_clean.parquet`
- **Output**: `note_vectors.parquet` (384d embeddings)
- **Processing**: SentenceTransformer encoding
- **Time**: ~8s

### Stage 7: Client Segmentation
- **Input**: `note_vectors.parquet`, `note_concepts.csv`
- **Output**: `client_profiles.csv` (100 clients, 12 tags each)
- **Processing**:
  1. KMeans clustering (7 segments)
  2. Per-client tag extraction (budget first)
  3. Profile type labeling
- **Time**: ~3s

### Stage 8: Action Recommendations
- **Input**: `client_profiles.csv`, `activations/playbooks.yml`
- **Output**: `recommended_actions.csv`
- **Processing**: Playbook matching and ranking
- **Time**: ~1s

### Stage 9: ML Predictions
- **Input**: `client_profiles.csv`, `note_vectors.parquet`
- **Output**: `ml_predictions.csv`
- **Processing**:
  - Purchase probability (RandomForest)
  - Churn risk (GradientBoosting)
  - Customer Lifetime Value (Regression)
- **Time**: ~5s

### Stage 10: Dashboard Data Generation
- **Input**: All outputs
- **Output**: `dashboard/src/data.json`
- **Processing**: Aggregate all data for frontend
- **Time**: ~2s

**Total Pipeline Time**: ~65s (with LLM), ~35s (without LLM)

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
