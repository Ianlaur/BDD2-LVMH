# CLAUDE.md — LVMH Client Intelligence Platform

## Project Overview

Full-stack hybrid NLP pipeline that transforms luxury retail sales advisor transcriptions (CSV) into rich, multi-layered client profiles with product recommendations. Built for LVMH as a real clienteling tool.

**Stack:** Python 3.14+ (FastAPI, PyTorch, scikit-learn) | React/TypeScript (Vite, TailwindCSS) | PostgreSQL (Neon) | Apple Silicon (MPS)

## Quick Commands

```bash
# Backend
make venv                              # Create virtual environment
source .venv/bin/activate              # Activate venv
python -m server.run_all               # Run full pipeline (12 stages)
python -m server.run_all --csv data/input/my_data.csv  # Run with specific CSV
./start-server.sh                      # Start FastAPI server (port 8000)

# Frontend
cd dashboard && npm install && npm run dev   # Dev server (port 5173)
cd dashboard && npm run build                # Production build

# Tests
pytest tests/test_big_o.py -v          # Big O complexity tests (11 tests)

# Vocabulary management
python -m server.server.train_vocabulary stats
python -m server.server.train_vocabulary add "term" "Label" "bucket" --aliases "alias1,alias2"
```

## Architecture

### Pipeline Stages (v2.0 — Intelligence Core Redesign)

```
Stage 0:  Scrape LV catalog (one-time / periodic)
Stage 1:  Ingest CSV → notes_clean.parquet
Stage 2:  GDPR anonymization (runs before any extraction)
Stage 3:  Layer 1 — Deterministic extraction (Aho-Corasick v2 + Regex + spaCy + negation detection)
Stage 4:  Layer 2 — Fine-tuned DeBERTa multi-task inference (concepts, sentiment, intent, cluster vector, evidence spans)
Stage 5:  Layer 3 — Fine-tuned SentenceTransformer (domain embeddings, product matching)
Stage 6:  Layer 4 — Cross-field reasoning (composite scores, commercial scoring, next best action)
Stage 7:  Profile assembly → full 84-field JSON per client (14 dimensions)
Stage 8:  Clustering → DBSCAN/KMeans on 12-dim behavioral cluster vectors
Stage 9:  Knowledge graph build → 15k+ nodes, 80k+ edges, 12 edge types
Stage 10: Product recommendations → top 10 LV products per client with match scores
Stage 11: Dashboard data generation
Stage 12: DB sync
```

### 4-Layer Extraction Stack (no external LLM APIs)

```
Layer 1: DETERMINISTIC    — Aho-Corasick, regex, spaCy deps, negation → "stated" fields
Layer 2: CONTEXTUAL       — Fine-tuned DeBERTa-v3-base (125M params) → "implied" + "inferred" fields
Layer 3: SEMANTIC          — Fine-tuned SentenceTransformer → embeddings, product matching, clustering
Layer 4: CROSS-FIELD      — Python rule engine → composite scores, commercial intelligence
```

All inference runs locally on Apple Silicon MPS. < 200ms/note total. No API calls.

### Fine-tuned Models (PyTorch)

| Model | Base | Params | Purpose | Training Time (Mac) |
|-------|------|--------|---------|-------------------|
| LuxuryProfileExtractor | DeBERTa-v3-base | 125M | Multi-task extraction (5 heads: concepts, sentiment, intent, cluster vector, evidence spans) | 2-4 hours |
| LuxuryEmbeddings | paraphrase-multilingual-MiniLM-L12-v2 | 33M | Domain-specific embeddings for clustering + product matching | 1-2 hours |

Training data: 1,700 augmented notes (100 real → augmented via translation round-trip, paraphrasing, synthetic generation).

### Key Directories

```
server/                  # Backend — pipeline + API
  run_all.py             # Pipeline orchestrator
  api_server.py          # FastAPI (40+ endpoints, port 8000)
  extract/               # Extraction (Aho-Corasick, regex, DeBERTa, Qwen)
  profiling/             # Client segmentation + profile assembly
  embeddings/            # Vector building (SentenceTransformer)
  db/                    # PostgreSQL layer (13 tables, schema.py + crud.py)
  shared/                # Config, utils, knowledge graph
  privacy/               # GDPR anonymization
dashboard/               # Frontend — React/TypeScript/Vite
  src/App.tsx            # Main app (needs decomposition — 1,743 lines)
  src/Client360Page.tsx  # Client detail view
  src/services/          # API + DB fallback services
taxonomy/                # Ontology, vocabulary, lexicon files
activations/             # Playbooks (YAML)
models/                  # ML model cache (gitignored)
tests/                   # Big O tests (11/11 passing)
data/                    # Input/processed/outputs (gitignored)
```

## Ontology Design (14 Dimensions)

The intelligence core uses a 3-level hierarchical ontology replacing the old flat 6-bucket taxonomy:

```
Domain → Dimension (14) → Category (~120) → Concept (~800-1000 leaf nodes)
```

| # | Dimension | What it captures |
|---|-----------|-----------------|
| 1 | product_affinity | Category interest: leather goods, watches, jewelry, fragrance... |
| 2 | maison_affinity | Brand loyalty/interest across 75 LVMH maisons |
| 3 | material_craftsmanship | Material preferences, craftsmanship sensitivity, sustainability |
| 4 | purchase_context | Self-purchase, gift, investment, decision stage, timeframe |
| 5 | occasion | Life events, calendar events, cultural events, travel |
| 6 | budget_intelligence | Stated amount, flexibility, spending tier, price sensitivity |
| 7 | lifestyle_signals | Profession, interests, travel, family (only what advisor reports) |
| 8 | client_relationship | Tenure, loyalty, engagement level, churn signals |
| 9 | behavioral_markers | Decision style, aesthetic confidence, brand knowledge, communication |
| 10 | cultural_context | Language, cultural background, aesthetic influences |
| 11 | sentiment_extraction | Valence, brand sentiment, pain points, delight signals, objections |
| 12 | service_mapping | Actions taken, follow-up needed, channel preferences |
| 13 | competitive_intelligence | Competitor mentions, switch probability, win strategy |
| 14 | commercial_scoring | Purchase intent, basket estimate, cross-sell, lifetime value, next best action |

### Concept Relationships

| Edge Type | Example |
|-----------|---------|
| implies | watch_collector → horological_knowledge |
| conflicts | budget_constrained ↔ no_budget_ceiling |
| amplifies | anniversary + jewelry → urgency boost |
| co_occurs | art_collector + japanese_denim |
| upgrades_to | fashion_entry → couture_client |
| substitutes | cartier_bridal_alternative ↔ tiffany_bridal |
| seasonal | chinese_new_year peaks Jan-Feb |

## Client Profile Schema (84 sub-fields)

Every field has:
- `score` or `value` — the extracted data
- `tier` — "stated" (explicitly said), "implied" (inferable), "inferred" (model predicted), or "absent"
- `evidence` — exact quote from advisor note (for stated/implied fields)

This ensures GDPR Article 22 compliance (right to explanation) and advisor trust.

**Zero PII in profiles.** All personal data anonymized at ingestion (Stage 2). Only behavioral signals retained.

## Product Matching Layer (Louis Vuitton)

Product catalog scraped from LV website (~2,000-4,000 products). Each product embedded with fine-tuned SentenceTransformer and auto-tagged with ontology concepts.

**Match formula:**
```
match_score = 0.25 × cosine(client_emb, product_emb)
            + 0.25 × concept_overlap(client_tags, product_tags)
            + 0.20 × budget_fit(client_budget, product_price)
            + 0.10 × occasion_boost(client_occasion, product_gifting)
            + 0.15 × style_alignment(client_style, product_style)
            - 0.30 × rejection_match(client_rejects, product_attrs)
```

**Future capability (not implemented):** Purchase history integration for repeat/cross-sell signals.

## Knowledge Graph

```
Node types:  DimensionNode (14), CategoryNode (~120), ConceptNode (~800),
             ClientNode (N), ProductNode (~3000), ClusterNode (K)

Edge types:  BELONGS_TO, HAS_CONCEPT, MATCHES_PRODUCT, IN_CLUSTER,
             SIMILAR_TO, IMPLIES, CONFLICTS, AMPLIFIES, CO_OCCURS,
             TAGGED_WITH, CONQUEST_FROM, RECOMMENDED

Scale:       ~15,000-20,000 nodes, ~80,000-120,000 edges
```

## Clustering

12-dimensional behavioral cluster vectors per client:

```
minimalist_vs_bold, discreet_vs_logo, functional_vs_decorative,
classic_vs_trendy, deliberate_vs_impulse, self_vs_gift,
premium_vs_ultra, engaged_vs_passive, conquest_vs_loyal,
travel_vs_local, creative_vs_corporate, solo_vs_family
```

All normalized 0-1. Used for KMeans/DBSCAN clustering and client similarity.

## GDPR Compliance

- Anonymization runs BEFORE any extraction (Stage 2)
- Zero PII in profiles — only behavioral signals
- Every automated score has `tier` + `evidence` (explainability)
- `human_review_required: true` on all profiles (Article 22)
- Audit log in PostgreSQL for all data access
- Right-to-erasure endpoint: `DELETE /api/rgpd/erase/{client_id}`
- Right-to-export endpoint: `GET /api/rgpd/export/{client_id}`
- Retention policy: 24 months from last interaction

## Data Augmentation Strategy

100 real notes → 1,700 training examples:

| Stage | Method | Output |
|-------|--------|--------|
| Translation round-trip | FR→EN→FR, FR→IT→FR | ~400 variants |
| Paraphrase generation | Local LLM rephrases in 3 styles | ~1,200 variants |
| Synthetic generation | Ontology concept combinations → new notes | ~500 synthetic |

Quality controls: deduplication (cosine > 0.95), human spot-check on ~50 samples.

## Database (PostgreSQL / Neon)

13 tables: users, clients, client_concepts, segments, client_actions, client_vectors, lexicon, audit_log, pipeline_runs, events, event_targets, client_scores, playbooks.

Schema extensions needed for v2: product catalog table, full profile JSON storage, ontology tables.

## Environment Variables

```bash
DATABASE_URL=postgresql://...          # Neon PostgreSQL (required for DB features)
HOST=0.0.0.0                           # FastAPI bind address
PORT=8000                              # FastAPI port
KAFKA_BOOTSTRAP_SERVERS=               # Optional event streaming
ENABLE_ANONYMIZATION=true              # GDPR anonymization (default: true)
ANONYMIZATION_AGGRESSIVE=false         # Aggressive mode (more false positives)
ENABLE_QWEN_ENHANCEMENT=true          # Qwen LLM extraction (legacy, being replaced)
ENABLE_BUDGET_EXTRACTION=true          # Regex budget detection
```

## Known Issues & Technical Debt

- `App.tsx` is 1,743 lines — needs decomposition into smaller components
- `api_server.py` is 1,796 lines — consider splitting into route modules
- `crud.py` is 2,120 lines — consider splitting by domain
- CORS is wide open (`*`) — restrict to dashboard URL in production
- `.env.example` contains a real-looking database URL — sanitize
- `server/extract/ml_detect.py` has `# TODO: Implement actual ML inference` — returns `[]`
- Only Big O tests exist — no unit tests for API, pipeline stages, or frontend
- Old phantom concept (CONCEPT_0000_401281) matches filler words in 53.8% of notes — root cause of tag repetition, being fixed by ontology redesign

## Coding Conventions

- **Python:** snake_case, type hints, `from server.` imports
- **TypeScript:** camelCase components, PascalCase for React components
- **Seeds:** All random operations use fixed seeds (42) for reproducibility
- **Logging:** Use `log_stage(stage_name, message)` from `server.shared.utils`
- **Config:** All constants in `server/shared/config.py` — no magic numbers in code
- **Data flow:** Each pipeline stage reads from previous stage's output file, writes its own
- **Fallback pattern:** API → DB → file-based data (smart degradation)
- **GDPR first:** Anonymization runs before any data processing

## Git Workflow

- **Main branch:** `main`
- **Dev branch:** `dev` (current working branch)
- **Commit style:** `type: description` (e.g., `feat:`, `fix:`, `refactor:`, `docs:`)
- **Data files gitignored:** CSV, parquet, model weights never committed
- **Taxonomy files:** Generated by pipeline, gitignored (regenerated on run)
