# PRD — Client Intelligence Platform (Hybrid: Rule-Based + LLM) Lexicon + 3D Vector Profiling + Action Plans

## 1. Overview
This project converts multilingual Client Advisor transcriptions (CSV) into:
1) a **lexicon/taxonomy** with 575 concepts and 9,003 multilingual aliases,
2) a **vector space** where notes/clients and concepts have coordinates,
3) interpretable **client profiles** with 12 personalized tags (budget first),
4) deterministic **action plans** via playbook rules,
5) **ML predictions** for purchase probability, churn risk, and CLV.

Key approach: **Hybrid extraction** combining rule-based (Aho-Corasick + Regex) with LLM semantic enhancement (Qwen2.5:3b).  
All workflows are **reproducible** via Docker and fixed seeds (except LLM).

### Key Features (Production v1.0)
- **Trainable Vocabulary**: 575 multilingual concepts (12+ languages), 9,003 aliases
- **Hybrid Extraction**: Rule-based (1,454) + Regex budgets (100) + LLM semantic (342) = **1,796 total concepts**
- **Per-Client Tags**: 12 personalized tags per client (budget always first)
- **Budget Detection**: Regex-based extraction (ranges, amounts, currencies) — 100% coverage
- **LLM Enhancement**: Qwen2.5:3b via Ollama (~10s/note, 99% success, 0.91 confidence)
- **Adaptive Pipeline**: Works with any CSV file structure
- **FastAPI Server**: REST API for frontend integration
- **PostgreSQL Database**: Neon cloud-hosted, 100 clients synced
- **React Dashboard**: TypeScript frontend with Client360 view
- **Big O Performance**: 11 complexity tests validating scalability

---

## 2. Goals & Success Metrics

### Goals
- Build a lexicon/taxonomy with **575 concepts** and **9,003 multilingual aliases**
- Support multilingual content: FR/EN/IT/ES/DE/PT/RU/AR/ZH/JA/KO/NL
- Combine rule-based precision with LLM semantic coverage
- Extract **100% of budget amounts** from client notes
- Assign each client **12 personalized tags** (budget first)
- Produce ML predictions (purchase probability, churn risk, CLV)
- Generate **ranked action plans** from deterministic playbooks
- Provide REST API for frontend integration

### Success Metrics (Production v1.0)
- ✅ **Pipeline reproducibility:** Fixed seeds ensure deterministic outputs (except LLM)
- ✅ **Concept coverage:** 1,796 concepts detected (1,454 rule + 342 LLM) = **100% of notes**
- ✅ **Budget extraction:** 100/100 notes with budgets captured (100% coverage)
- ✅ **LLM success rate:** 99% (99/100 notes enhanced successfully)
- ✅ **LLM confidence:** 0.91 average across all LLM-extracted concepts
- ✅ **Per-client tags:** 12 personalized tags per client (vs 3 cluster-level before)
- ✅ **Performance:** ~65s total pipeline (10s/note LLM, <1ms/note rule-based)
- ✅ **Big O validation:** All 11 complexity tests passing (5.15s)
- ✅ **Database sync:** 100 clients synced to PostgreSQL with full concept evidence
- ✅ **Profile interpretability:** Each profile has budget + top concepts + cluster label
- ✅ **Action traceability:** Every recommended action lists triggers (concepts/rules)

---

## 3. Users & Use Cases

### Primary users
- Data/AI team: building and validating the pipeline
- CRM / clienteling stakeholders: consuming profiles and actions

### Primary use cases
- Explore a client/note in a semantic space (3D demo)
- Identify client type from note(s)
- Generate recommended outreach/actions (CRM, email, event invite, service follow-up)

---

## 4. In Scope vs Out of Scope

### In Scope (✅ Implemented)
- ✅ CSV ingestion and cleaning
- ✅ Lexicon creation with 575 concepts, 9,003 aliases
- ✅ Hybrid extraction: Rule-based + Regex + LLM
  - Aho-Corasick multi-pattern matching
  - Regex budget extraction (amounts, ranges, currencies)
  - Qwen2.5:3b semantic enhancement via Ollama
- ✅ Vectorization (SentenceTransformers 384d)
- ✅ Segmentation (KMeans 7 clusters) + per-client tagging (12 tags)
- ✅ Playbook-based action recommendation
- ✅ ML predictions (purchase, churn, CLV)
- ✅ FastAPI REST API server (port 8000)
- ✅ PostgreSQL database (Neon cloud)
- ✅ React/TypeScript dashboard frontend
- ✅ Big O complexity testing (11 tests)
- ✅ GDPR/RGPD anonymization
- ✅ Dockerized, reproducible pipeline
- ✅ Outputs: tables + 3D HTML visualization

### Out of Scope
- ❌ Real-time streaming/websocket processing
- ❌ Production CRM integration (we output data via API)
- ❌ Personalized copy generation for emails (no generative LLM for content)
- ❌ Multi-tenant architecture (single workspace)
- ❌ Model retraining on new 1,796-feature set (future enhancement)

---

## 5. Input Data & Assumptions

### Standard CSV schema (LVMH Format)
- `ID` (note id; may be used as client id in MVP)
- `Date` (ISO date or parseable)
- `Duration` (categorical/number)
- `Language` (FR/EN/IT/ES/DE)
- `Length` (short/medium/long)
- `Transcription` (free text)

### Adaptive Mode (Any CSV)
The pipeline supports automatic column detection for any CSV:
```bash
python -m src.run_all --csv data/input/any_file.csv --analyze-only  # Analyze structure
python -m src.run_all --csv data/input/any_file.csv                  # Run with auto-detection
python -m src.run_all --csv data/input/any_file.csv --text-column "notes"  # Manual override
```

### Client identity assumption
If no explicit `client_id` exists:
- MVP treats `ID` as `client_id` (1 note = 1 client profile)
- Future enhancement: add mapping if CRM identifiers become available

---

## 6. Functional Requirements

### FR1 — Ingest & Validate
- Load CSV
- Validate schema and required columns
- Normalize dates and language codes
- Output standardized dataset (`notes_clean.parquet`)

### FR2 — Candidate Concept Extraction (Non-LLM)
Extract candidate terms/phrases from `Transcription` using deterministic methods:
- TF-IDF terms (global + per language)
- Keyphrase extraction (YAKE/RAKE)
- Optional noun-phrase extraction (spaCy models)
Return a scored candidate list.

### FR3 — Lexicon & Taxonomy Generation
From candidates:
- Normalize and deduplicate
- Merge synonyms (string rules + optional embedding similarity)
- Create lexicon concepts with:
  - label
  - aliases (multilingual: EN, FR, IT, ES, DE, ZH, JA, KO, RU, AR, PT, NL)
  - languages
  - scoring stats (freq, tfidf)
  - examples (note ids + snippets)
  - rule (how to detect concept: alias match / lemma match / pattern)
Create taxonomy grouping (e.g., intent, preferences, occasions, constraints, lifestyle, next_action).

### FR3b — Vocabulary Training System (v2.0)
Train and manage vocabulary via CLI:
```bash
python -m src.server.train_vocabulary stats                    # View statistics
python -m src.server.train_vocabulary add "term" "Label" "bucket" --aliases "alias1,alias2"
python -m src.server.train_vocabulary import keywords.json      # Batch import
python -m src.server.train_vocabulary list --bucket preferences # List by category
```

Current vocabulary: **384 concepts** across 6 buckets (preferences, intent, lifestyle, occasion, constraints, next_action).

### FR4 — Concept Detection on Notes
Given lexicon:
- Detect which concepts appear in each note
- Output `extracted_concepts` per note with counts and evidence spans (string indices or matched alias)

### FR5 — Vector Space Creation
Create vectors for:
- notes (and/or clients) and concepts
Options:
- TF-IDF vectors (pure classical)
- fastText embeddings (pretrained, non-generative) if allowed
Vectors must be reproducible (seeded, pinned versions).

### FR6 — Segmentation & Client Typing
- Cluster client/note vectors into segments
- Generate human-readable `profile_type` labels using top concepts/features
- Output `client_profiles.csv`

### FR7 — Action Recommendation (Playbooks)
- Maintain playbooks (YAML/CSV) mapping:
  - profile types and/or required concepts → actions
- For each client:
  - match playbooks
  - rank actions by priority and evidence strength
  - output `recommended_actions.csv` with traceability

### FR8 — Visualization (Optional but Recommended)
- Project vectors to 3D using PCA/UMAP for demo
- Create an HTML visualization:
  - show points: clients/notes + concept anchors
  - clicking a client shows top concepts + recommended actions (lightweight)

---

## 7. Non-Functional Requirements

### NFR1 — Reproducibility
- Docker build creates identical environment
- Pinned dependency versions
- Fixed random seeds for clustering/projection
- Deterministic output file naming/versioning

### NFR2 — Privacy & Data Safety
- No raw sensitive data committed to git
- Provide anonymization guidance
- Outputs should not include unnecessary PII

### NFR3 — Performance
- Must handle at least 300 notes on a standard laptop in reasonable time
- Avoid heavy GPU dependencies unless required

---

## 8. Proposed System Design

### Pipeline Stages
1. **Ingest** → `notes_clean.parquet`
2. **Candidate extraction** → `candidates.csv`
3. **Lexicon build** → `lexicon_v1.json` + `taxonomy_v1.json`
4. **Concept detection** → `extracted_concepts.jsonl` + flat CSV
5. **Vectorization** → vectors for notes/clients + concepts
6. **Segmentation** → `client_profiles.csv`
7. **Playbook recommendation** → `recommended_actions.csv`
8. **3D projection** (optional) → `embedding_space_3d.html`

### Architecture (v2.0)
```
src/
├── run_all.py              # Main orchestrator
├── shared/                 # Shared configuration & utilities
│   ├── config.py
│   └── utils.py
├── server/                 # Training & vocabulary management
│   ├── train_vocabulary.py # CLI for vocabulary training
│   ├── vocabulary_manager.py
│   ├── vocabulary_learner.py
│   ├── adaptive_pipeline.py
│   └── flexible_extraction.py
└── client/                 # Processing pipeline
    ├── ingest/
    ├── extract/
    ├── lexicon/
    ├── embeddings/
    ├── profiling/
    ├── actions/
    └── visualization/
```

### Data Model (minimal)
- `notes_clean`: `note_id`, `client_id`, `date`, `language`, `text`
- `lexicon`: `concept_id`, `label`, `aliases`, `languages`, `freq`, `rule`, `examples`
- `note_concepts`: `note_id`, `concept_id`, `evidence`, `count`
- `client_profiles`: `client_id`, `profile_type`, `cluster_id`, `top_concepts`, `confidence`
- `actions`: `client_id`, `action_id`, `title`, `channel`, `priority`, `triggers`, `kpi`

---

## 9. Playbooks Specification (Deterministic)
Playbooks must include:
- `id`
- `applies_to` (profile_types and/or concepts)
- `action` (channel, title, priority, kpi)
- `reason` / `rationale` (static text)
- optional thresholds (min count, min recency, etc.)

Example:
```yaml
- id: "PB_HIGH_INTENT_APPOINTMENT"
  applies_to:
    profile_types: ["HighIntent"]
    required_concepts: ["INTENT_APPOINTMENT", "TIMEFRAME_SOON"]
  action:
    channel: "CRM"
    title: "Propose appointment + curated selection"
    priority: "High"
    kpi: "Appointment booked"
  rationale: "Client shows near-term intent and appointment signals."
```

## 12. Milestones

### Milestone 1 — MVP Lexicon (Week 1)
	- Ingest + cleaning
	- Candidate extraction
	- Lexicon v1 + taxonomy v1

### Milestone 2 — Vector + Segmentation (Week 2)
	- Concept detection
	- Note/client vectors
	- Baseline clustering + profile labels

### Milestone 3 — Actions + Demo (Week 3)
	- Playbooks + recommender
	- Traceable recommended actions
	- 3D visualization prototype

### Milestone 4 — Scale + Stabilize (Week 4)
	- Apply to additional notes (Wave 2)
	- Lexicon v2 refinements (merge/split)
	- Comparative report + final demo readiness

---

## 13. Risks & Mitigations

Risk: Multilingual synonym explosion

Mitigation: concept clustering (TF-IDF space baseline; fastText optional) + alias lists per concept.

Risk: No client_id available

Mitigation: MVP uses ID as client; architecture supports later mapping.

Risk: Lexicon becomes too large / noisy

Mitigation: enforce thresholds (min frequency, min tfidf) and cap per-category.

Risk: Clusters hard to interpret

Mitigation: label clusters using top concepts + provide “why” (feature contributions).

---

## 14. Acceptance Criteria
	- Running in Docker produces:
	- taxonomy/lexicon_v1.json and taxonomy/taxonomy_v1.json
	- taxonomy/vocabulary.json (trained vocabulary)
	- data/outputs/client_profiles.csv
	- data/outputs/recommended_actions.csv
	- optional demo/embedding_space_3d.html
	- Pipeline is deterministic (seeded) and documented
	- Each client type is interpretable with top concepts
	- Each action recommendation includes triggers/evidence
	- Vocabulary training CLI is functional
	- Pipeline works with any CSV structure (adaptive mode)