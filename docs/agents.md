# agents.md

## Project: LVMH Client Intelligence Platform — Lexicon, Vector Profiling & Action Playbooks

This repository builds a deterministic (non-LLM) pipeline that:
1) extracts a **lexicon/taxonomy** from multilingual Client Advisor transcriptions (CSV),
2) embeds notes/concepts to create a **vector space** (with optional 3D projection for visualization),
3) segments each note/client into a **client type**,
4) generates **action plans** via rule-based playbooks.

> Constraint: **No generative LLMs** (no GPT/Claude/API prompt extraction).  
> Implementation must be reproducible via **Docker** to avoid OS/version drift.

---

## Agent Model (team responsibilities)

### 1) Data & Ingestion Agent
**Purpose:** Ensure data is loaded, validated, cleaned, and versioned.

**Responsibilities**
- Load CSV input (`ID`, `Date`, `Duration`, `Language`, `Length`, `Transcription`)
- Validate schema and constraints (no missing critical fields, date formats, language codes)
- Basic cleaning: whitespace normalization, punctuation normalization, safe text handling
- Produce standardized dataset files in `data/processed/` (Parquet recommended)
- Maintain data dictionaries and data lineage in `docs/`

**Outputs**
- `data/processed/notes_clean.parquet`
- `data/processed/notes_clean.jsonl` (optional)
- `docs/data_dictionary.md`

---

### 2) NLP Extraction Agent (Non-LLM)
**Purpose:** Extract candidate keywords/phrases and structured signals in a deterministic way.

**Responsibilities**
- Candidate extraction using:
  - TF-IDF (global + per-language)
  - Keyphrase extraction (YAKE/RAKE)
  - Optional noun-phrase extraction with spaCy models (FR/EN/IT/ES/DE) if available
- Entity extraction with rules/NER (dates, cities, money, sizes) where possible
- Generate a `candidates.csv` with scores, frequencies, and examples

**Outputs**
- `data/processed/candidates.csv`
- `data/processed/entities.csv` (optional)

---

### 3) Lexicon Builder Agent
**Purpose:** Convert candidates into a usable lexicon/taxonomy (built from the CSV only).

**Responsibilities**
- Normalize terms (lowercase, lemmatize, strip punctuation)
- Merge synonyms and multilingual variants:
  - Baseline: string normalization + frequency rules
  - Optional: embedding-based merging (fastText word vectors) if allowed
- Produce:
  - `lexicon_v1.csv` with concept labels, aliases, languages, rules, examples
  - `taxonomy_v1.json` grouping concepts into higher-level categories

**Outputs**
- `taxonomy/lexicon_v1.csv`
- `taxonomy/taxonomy_v1.json`

---

### 4) Embeddings & Vector Space Agent (Non-LLM)
**Purpose:** Build vectors for notes/clients and concept tags; create 3D projection for demo/interpretability.

**Responsibilities**
- Choose vector strategy (must be deterministic):
  - Option A: TF-IDF vectors (pure classical, no pretrained models)
  - Option B: fastText embeddings (pretrained, non-generative; confirm allowed)
- Build:
  - Note vectors
  - Concept vectors (from lexicon aliases)
  - Client vectors (aggregate by client_id; MVP can treat `ID` as client)
- Produce 3D projection using PCA/UMAP (3D is visualization only, not the decision space)

**Outputs**
- `data/outputs/note_vectors.(parquet|npy)`
- `data/outputs/concept_vectors.(parquet|npy)`
- `data/outputs/client_vectors.(parquet|npy)`
- `demo/embedding_space_3d.html` (optional)

---

### 5) Segmentation & Profile Typing Agent
**Purpose:** Assign “client types” from the vector space + lexicon signals.

**Responsibilities**
- Segment using:
  - k-means (baseline)
  - HDBSCAN (if available) for variable cluster density
- Label each segment using top concepts/features (interpretability required)
- Optionally implement a stable rule layer (cluster + rules hybrid)

**Outputs**
- `data/outputs/client_profiles.csv` with:
  - `client_id`, `profile_type`, `cluster_id`, `top_concepts`, `confidence`

---

### 6) Playbooks & Action Planning Agent
**Purpose:** Deterministically map profiles/concepts into actionable plans.

**Responsibilities**
- Design playbooks (YAML/CSV):
  - profile-based actions
  - concept-based triggers
  - channel + KPI + priority
- Implement recommender:
  - match rules + score actions by evidence strength and recency
- Ensure traceability: each recommendation must cite the concepts/evidence that triggered it

**Outputs**
- `activations/playbooks.yml`
- `data/outputs/recommended_actions.csv`

---

### 7) Evaluation & QA Agent
**Purpose:** Validate correctness, stability, and reproducibility.

**Responsibilities**
- Data QA: schema validation, language distribution checks
- Pipeline QA: deterministic outputs, schema checks for outputs
- Quality metrics:
  - coverage (% notes with ≥1 concept)
  - stability (same input => same output)
  - interpretability (segment labels explain clusters)
- Produce evaluation report artifacts

**Outputs**
- `docs/evaluation_report.md`
- `data/outputs/metrics.json`

---

## Communication Contracts (shared schemas)
All agents must adhere to these file formats to avoid coupling issues:
- Notes: `data/processed/notes_clean.parquet`
- Candidates: `data/processed/candidates.csv`
- Lexicon: `taxonomy/lexicon_v1.csv`
- Taxonomy: `taxonomy/taxonomy_v1.json`
- Vectors: `data/outputs/*_vectors.parquet`
- Profiles: `data/outputs/client_profiles.csv`
- Actions: `data/outputs/recommended_actions.csv`

---

## Determinism Requirements
To keep results reproducible:
- fixed random seeds in vectorization/clustering/projection
- pinned library versions (via Docker build)
- avoid non-deterministic parallelism unless controlled

---

## Docker Rules
- All development and runs must be possible inside Docker:
  - `docker build` creates a stable environment
  - `docker run ... make pipeline` produces the full output
- All dependencies must be pinned (`requirements.txt` or `poetry.lock`)
- Any downloadable model assets (spaCy models, fastText vectors) must be:
  - fetched in a deterministic way, OR
  - mounted from the host with checksums and documented steps

---

## Definition of Done
Project is complete when:
- Lexicon/taxonomy are produced **from the CSV** (no predefined lexicon)
- Pipeline runs end-to-end in Docker
- Client types are produced with explainable labels
- Action plans are generated deterministically with traceability
- Optional demo: 3D visualization showing clients and concept anchors