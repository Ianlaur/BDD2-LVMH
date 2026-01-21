# LVMH Voice-to-Tag — Vector Profiles

A **deterministic (non-LLM)** multilingual pipeline that converts Client Advisor transcriptions into actionable client profiles and recommendations.

## Quick Start

### Local Development

```bash
# 1. Create virtual environment and install dependencies
make venv

# 2. Download the embedding model
make setup-models-local

# 3. Place your CSV in data/raw/

# 4. Run the pipeline
make dev
```

### Docker (Recommended for Reproducibility)

```bash
# 1. Build the Docker image
make build

# 2. Place your CSV in data/raw/

# 3. Run the pipeline
make run
```

## Features

- **No LLM required**: Uses only deterministic NLP (YAKE, RAKE, TF-IDF) and pre-trained embeddings
- **Multilingual support**: FR, EN, IT, ES, DE
- **Reproducible**: Fixed random seeds, Docker containerized
- **End-to-end pipeline**: From raw transcriptions to action recommendations

## Pipeline Outputs

| Output | Description |
|--------|-------------|
| `data/processed/notes_clean.parquet` | Cleaned and normalized notes |
| `data/processed/candidates.csv` | Extracted candidate keyphrases |
| `taxonomy/lexicon_v1.csv` | Auto-generated lexicon with concepts |
| `taxonomy/taxonomy_v1.json` | Taxonomy grouping concepts into buckets |
| `data/outputs/note_concepts.csv` | Concept matches per note |
| `data/outputs/note_vectors.parquet` | Note embeddings (384-dim) |
| `data/outputs/client_profiles.csv` | Client clusters and profile types |
| `data/outputs/recommended_actions.csv` | Action recommendations per client |
| `demo/embedding_space_3d.html` | Interactive 3D visualization |

## Input Format

CSV file with columns:
- `ID`: Note identifier
- `Date`: Date of interaction
- `Duration`: Duration (e.g., "35 min")
- `Language`: FR, EN, IT, ES, or DE
- `Length`: short, medium, or long
- `Transcription`: Free text content

## Documentation

See [docs/pipeline.md](docs/pipeline.md) for detailed pipeline documentation.

## License

Proprietary - LVMH
