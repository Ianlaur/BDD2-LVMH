# LVMH Voice-to-Tag Pipeline

## Overview

This repository implements a **deterministic (non-LLM)** multilingual pipeline that converts Client Advisor transcriptions into actionable client profiles and recommendations.

**Key constraint:** No generative LLMs (GPT/Claude/OpenAI) are used. Only deterministic NLP methods and pre-trained embeddings (SentenceTransformers) are allowed.

## Pipeline Stages

### 1. Ingest (`src/ingest/run_ingest.py`)
- Loads CSV from `data/raw/` (or `data/` fallback)
- Validates required columns: `ID`, `Date`, `Duration`, `Language`, `Length`, `Transcription`
- Normalizes dates, languages, and text
- **Output:** `data/processed/notes_clean.parquet`

### 2. Candidate Extraction (`src/extract/run_candidates.py`)
- Extracts keyphrases using YAKE, RAKE-NLTK, and TF-IDF n-grams
- Per-language and global extraction
- Normalizes candidates (lowercase, strip punctuation, collapse whitespace)
- **Output:** `data/processed/candidates.csv`

### 3. Lexicon Building (`src/lexicon/build_lexicon.py`)
- Embeds candidates using SentenceTransformer (multilingual MiniLM)
- Clusters candidates into concepts using Agglomerative Clustering (cosine distance)
- Selects concept labels (most frequent alias or nearest to centroid)
- Assigns taxonomy buckets via keyword rules
- **Outputs:**
  - `taxonomy/lexicon_v1.csv`
  - `taxonomy/taxonomy_v1.json`

### 4. Concept Detection (`src/extract/detect_concepts.py`)
- Matches lexicon aliases case-insensitively in transcriptions
- Records evidence spans (start/end indices)
- Limits overlapping matches (max 3 per alias per note)
- **Output:** `data/outputs/note_concepts.csv`

### 5. Vector Building (`src/embeddings/build_vectors.py`)
- Computes embeddings for each note using SentenceTransformer
- L2 normalized for cosine similarity
- **Output:** `data/outputs/note_vectors.parquet`

### 6. Client Segmentation (`src/profiling/segment_clients.py`)
- Aggregates note vectors to client level (mean for multiple notes)
- Clusters using KMeans with fixed random_state
- K determined by heuristic: `min(8, max(3, sqrt(n/2)))` or `CLUSTERS_K` env var
- Labels profiles with top concepts per cluster
- **Output:** `data/outputs/client_profiles.csv`

### 7. Action Recommendation (`src/actions/recommend_actions.py`)
- Loads playbooks from `activations/playbooks.yml`
- Matches clients to actions based on:
  - Profile type keywords
  - Detected concepts
  - Bucket-based triggers
- Ranks by priority (High > Medium > Low) and evidence strength
- **Output:** `data/outputs/recommended_actions.csv`

### 8. 3D Projection (Optional) (`src/embeddings/projection_3d.py`)
- Reduces embeddings to 3D using UMAP (or PCA fallback)
- Creates interactive Plotly visualization
- **Output:** `demo/embedding_space_3d.html`

## How to Run

### Prerequisites
- Python 3.9+
- Docker (optional, recommended for reproducibility)

### Local Development (venv)

```bash
# Create and activate virtual environment
make venv

# Download embedding model
make setup-models-local

# Run pipeline
make dev
```

### Docker (Recommended)

```bash
# Build image
make build

# Run full pipeline
make run

# Interactive shell
make shell
```

### Manual Execution

```bash
# Activate venv
source .venv/bin/activate

# Run full pipeline
python -m src.run_all

# Or run individual stages:
python -m src.ingest.run_ingest
python -m src.extract.run_candidates
python -m src.lexicon.build_lexicon
python -m src.extract.detect_concepts
python -m src.embeddings.build_vectors
python -m src.profiling.segment_clients
python -m src.actions.recommend_actions
python -m src.embeddings.projection_3d
```

## Input Data

Place your CSV file in `data/raw/`. The file must have these columns:

| Column | Description |
|--------|-------------|
| `ID` | Note identifier (used as both note_id and client_id in MVP) |
| `Date` | Date of the note (ISO format preferred) |
| `Duration` | Duration of interaction (categorical: "20 min", "35 min", etc.) |
| `Language` | Language code: FR, EN, IT, ES, DE |
| `Length` | Length category: short, medium, long |
| `Transcription` | Free text transcription content |

## Outputs

| File | Description |
|------|-------------|
| `data/processed/notes_clean.parquet` | Cleaned and normalized notes |
| `data/processed/candidates.csv` | Extracted candidate keyphrases |
| `taxonomy/lexicon_v1.csv` | Lexicon with concepts, aliases, and rules |
| `taxonomy/taxonomy_v1.json` | Taxonomy grouping concepts into buckets |
| `data/outputs/note_concepts.csv` | Concept matches per note with evidence spans |
| `data/outputs/note_vectors.parquet` | Note embeddings |
| `data/outputs/client_profiles.csv` | Client clusters and profile types |
| `data/outputs/recommended_actions.csv` | Action recommendations per client |
| `demo/embedding_space_3d.html` | Interactive 3D visualization |

## Taxonomy Buckets

Concepts are automatically assigned to one of these buckets:

- **intent**: Purchase intent, appointments, repairs
- **occasion**: Birthdays, anniversaries, events
- **preferences**: Colors, materials, styles, categories
- **constraints**: Budget, allergies, dietary restrictions
- **lifestyle**: Sports, hobbies, travel, profession
- **next_action**: Follow-ups, calls, invitations
- **other**: Unclassified concepts

## Configuration

Key parameters in `src/config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `RANDOM_SEED` | 42 | Global random seed for reproducibility |
| `CLUSTERS_K` | 0 (auto) | Number of clusters (0 = heuristic) |
| `CONCEPT_CLUSTER_DISTANCE_THRESHOLD` | 0.35 | Cosine distance for concept clustering |
| `MIN_CANDIDATE_FREQ` | 2 | Minimum note frequency for candidates |
| `MAX_ALIAS_MATCHES_PER_NOTE` | 3 | Max matches per alias per note |

Environment variables:
- `CLUSTERS_K`: Override number of clusters

## Reproducibility

The pipeline is fully deterministic:
- Fixed random seeds (NumPy, scikit-learn)
- Fixed UMAP random_state
- Sorted outputs for stability
- No external API calls

## Model Caching

The SentenceTransformer model is cached in `models/sentence_transformers/`. In Docker, `HF_HOME` and `TRANSFORMERS_CACHE` are set to `/app/models/`.

To pre-download the model:
```bash
make setup-models-local  # For venv
make setup-models        # For system Python
```

## Playbooks

Action playbooks are defined in `activations/playbooks.yml`. Each action specifies:
- Channel (CRM, Email, Event, Client Service)
- Priority (High, Medium, Low)
- KPIs
- Triggers (buckets, keywords, confidence threshold)

The default playbooks include 10 actions across all channels.

## Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   CSV       │───>│   Ingest    │───>│   Notes     │
│   Input     │    │             │    │   Parquet   │
└─────────────┘    └─────────────┘    └──────┬──────┘
                                             │
                   ┌─────────────────────────┴─────────────────────────┐
                   │                                                   │
                   v                                                   v
           ┌──────────────┐                                 ┌──────────────┐
           │  Candidates  │                                 │   Vectors    │
           │  Extraction  │                                 │   Building   │
           └──────┬───────┘                                 └──────┬───────┘
                  │                                                │
                  v                                                │
           ┌──────────────┐                                        │
           │   Lexicon    │                                        │
           │   Building   │                                        │
           └──────┬───────┘                                        │
                  │                                                │
                  v                                                │
           ┌──────────────┐                                        │
           │   Concept    │                                        │
           │   Detection  │                                        │
           └──────┬───────┘                                        │
                  │                                                │
                  └─────────────────────┬──────────────────────────┘
                                        │
                                        v
                                ┌──────────────┐
                                │   Client     │
                                │ Segmentation │
                                └──────┬───────┘
                                       │
                                       v
                                ┌──────────────┐
                                │   Action     │
                                │ Recommend.   │
                                └──────┬───────┘
                                       │
                                       v
                                ┌──────────────┐
                                │    3D        │
                                │  Projection  │
                                └──────────────┘
```
