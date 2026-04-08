# Intelligence Core Redesign — Design Specification

**Date:** 2026-03-28
**Status:** Approved
**Scope:** Redesign the extraction pipeline, ontology, knowledge graph, vectorization, and add product matching

---

## 1. Problem Statement

The current extraction pipeline produces repetitive, low-quality tags across all clients:
- A phantom concept (CONCEPT_0000_401281) matches conversational filler words ("tipo", "menos", "like") in 53.8% of all detections
- All 300 clients share the same 6 tags — zero differentiation
- Knowledge graph uses only 242 of 1,038 concepts (23%)
- Flat 6-bucket taxonomy with no concept relationships
- 30% of lexicon concepts have zero aliases (dead weight)
- No diversity enforcement in tag selection
- Silhouette score 0.057 (nearly random clustering)

## 2. Goals

- Each client gets a **unique, rich profile** with 84+ sub-fields across 14 dimensions
- Every extracted field is **traceable** (stated/implied/inferred tier + evidence quote)
- **Fine-tuned models** trained on luxury retail domain (no external LLM API dependency)
- **Product matching** against real Louis Vuitton catalog with explainable match scores
- **GDPR compliant** — zero PII, full explainability, right to erasure/export
- **< 200ms/note** total extraction (vs ~10s/note current Qwen approach)
- **Clustering silhouette > 0.3** (vs 0.057 current)
- **Graceful degradation** — each layer falls back if subsequent layers fail

## 3. Architecture Overview

### 3.1 Extraction Stack (4 Layers)

```
Input: Anonymized advisor note
         │
         ▼
┌─────────────────────────────────────────┐
│ Layer 1: DETERMINISTIC EXTRACTION       │
│  Aho-Corasick v2 (800+ concepts)       │
│  Regex (budget, dates, brands, amounts) │
│  Negation detection (rule-based:        │
│    "pas de", "sans", "no", "never",     │
│    "nicht", "non", "sin" + scope rules) │
│                                         │
│  Output: stated fields + evidence spans │
│  Tier: "stated" (confidence 0.95+)      │
├─────────────────────────────────────────┤
│ Layer 2: CONTEXTUAL INFERENCE           │
│  Fine-tuned DeBERTa-v3-base (125M)     │
│  Multi-task heads:                      │
│    Head A: multi-label concept tags     │
│    Head B: sentiment scores             │
│    Head C: purchase intent + urgency    │
│    Head D: clustering vector (12-dim)   │
│    Head E: evidence span extraction     │
│                                         │
│  Output: implied + inferred fields      │
│  Tier: "implied" (0.7+) / "inferred"   │
├─────────────────────────────────────────┤
│ Layer 3: SEMANTIC SIMILARITY            │
│  Fine-tuned SentenceTransformer         │
│  Cosine similarity:                     │
│    note ↔ concept descriptions          │
│    note ↔ product descriptions          │
│    client ↔ client (clustering)         │
│                                         │
│  Output: product matches, clusters      │
├─────────────────────────────────────────┤
│ Layer 4: CROSS-FIELD REASONING          │
│  Python rule engine:                    │
│    concept combinations → scores        │
│    budget + reaction → stretch willing  │
│    competitor mention + positive LV     │
│      reaction → conquest probability    │
│                                         │
│  Output: composite scores, next action  │
└─────────────────────────────────────────┘
         │
         ▼
Full Profile (14 dimensions, 84 sub-fields)
```

### 3.2 Pipeline Stages (v2.0)

```
Stage 0:  Scrape LV catalog (one-time / periodic)
Stage 1:  Ingest CSV → notes_clean.parquet
Stage 2:  GDPR anonymization
Stage 3:  Layer 1 — Deterministic extraction
Stage 4:  Layer 2 — DeBERTa multi-task inference
Stage 5:  Layer 3 — SentenceTransformer embeddings + product matching
Stage 6:  Layer 4 — Cross-field reasoning
Stage 7:  Profile assembly → full 84-field JSON per client
Stage 8:  Clustering → DBSCAN/KMeans on 12-dim vectors
Stage 9:  Knowledge graph build
Stage 10: Product recommendations (top 10 per client)
Stage 11: Dashboard data generation
Stage 12: DB sync
```

## 4. Hierarchical Ontology (14 Dimensions)

### 4.1 Structure

```
Domain (Level 0)          →  "Luxury Retail Clienteling"
├── Dimension (Level 1)   →  14 dimensions
│   ├── Category (Level 2)→  ~120 categories
│   │   ├── Concept       →  ~800-1000 leaf nodes
│   │   │   ├── aliases[] →  multilingual (12+ languages)
│   │   │   ├── weight    →  differentiating power (IDF-like)
│   │   │   └── relationships[] → edges to other concepts
```

### 4.2 Dimensions

| # | Dimension | Categories (examples) | Example Concepts |
|---|-----------|----------------------|-----------------|
| 1 | product_affinity | watches, leather goods, jewelry, fragrance, fashion, cosmetics, eyewear, shoes, home | tourbillon_interest, exotic_leather_preference, high_jewelry_collector |
| 2 | maison_affinity | 75 LVMH maisons (LV, Dior, Tiffany, Bulgari, Hennessy, TAG Heuer, Fendi, Givenchy, Loewe, Celine, Rimowa, Berluti...) | louis_vuitton_loyalist, dior_couture_client, celine_minimalist |
| 3 | material_craftsmanship | leathers, metals, stones, fabrics, techniques, finishes | crocodile_leather, rose_gold_preference, hand_stitched, sustainability_interest |
| 4 | purchase_context | gift, self-purchase, investment, replacement, impulse, corporate, collection building | anniversary_gift_buyer, investment_piece_seeker, collection_completist |
| 5 | occasion | life events, calendar events, cultural events, social events, travel | ramadan_gift, destination_wedding, graduation_milestone, gallery_opening |
| 6 | budget_intelligence | amount range, flexibility, payment style, value perception | ultra_high_net_worth, aspirational_stretch, no_budget_ceiling |
| 7 | lifestyle_signals | profession, hobbies, sport, travel, dining, culture, wellness | finance_executive, art_collector, equestrian, yacht_owner |
| 8 | client_relationship | tenure, loyalty, engagement, acquisition channel | first_visit, decade_loyal, vip_by_invitation, lapsed_12mo |
| 9 | behavioral_markers | decision style, communication, aesthetic confidence, brand knowledge | detail_oriented_researcher, impulse_emotional, status_driven |
| 10 | cultural_context | regional preferences, cultural norms, aesthetic influences | middle_east_modest_fashion, japanese_minimalism, european_understated |
| 11 | sentiment_extraction | satisfaction, frustration, excitement, hesitation, trust, urgency | high_urgency_deadline, brand_switching_signal, delighted_repeat |
| 12 | service_mapping | repair, personalization, appointment, follow-up, concierge | monogram_engraving, bespoke_request, vip_private_showing |
| 13 | competitive_intelligence | competitor mentions, market positioning, cross-brand signals | hermes_comparison, chanel_defector, rolex_collector_adjacent |
| 14 | commercial_scoring | cross-sell, upsell, seasonal timing, trend alignment | cross_sell_fragrance_to_fashion, upsell_to_haute_horlogerie, limited_edition_fomo |

### 4.3 Concept Relationships

| Relationship | Example | Use |
|-------------|---------|-----|
| implies | yacht_owner → ultra_high_net_worth | Infer unstated attributes |
| conflicts | budget_constrained ↔ no_budget_ceiling | Catch contradictions |
| amplifies | anniversary + jewelry → urgency +0.3 | Score boosting |
| co_occurs | art_collector + japanese_denim | Pattern discovery |
| upgrades_to | fashion_entry → couture_client | Journey prediction |
| substitutes | cartier_bridal_alternative ↔ tiffany_bridal | Competitive positioning |
| seasonal | chinese_new_year peaks Jan-Feb | Timing triggers |

## 5. Client Profile Schema

### 5.1 Extraction Tiers

Every field has a tier indicating provenance:

| Tier | Meaning | Confidence | Example |
|------|---------|------------|---------|
| stated | Client or advisor said it explicitly | 0.95+ | "budget autour de 3000€" |
| implied | Strongly inferable from context | 0.70-0.94 | Architect → high aesthetic sensibility |
| inferred | Model prediction | 0.50-0.69 | Travel frequency from single trip mention |
| absent | Not mentioned, not guessed | null | Field left empty |

### 5.2 Full Schema (14 dimensions, 84 sub-fields)

```json
{
    "product_intelligence": {
        "category_affinity": {"leather_goods": {"score": 0.92, "tier": "stated", "evidence": "..."}},
        "style_signals": {"minimalist": {"score": 0.95, "tier": "stated", "evidence": "..."}},
        "functional_needs": {"travel_ready": {"value": true, "tier": "stated", "evidence": "..."}},
        "color_mentions": {"preferred": [], "rejected": []},
        "product_shown": {"items": [], "reaction": "positive", "tier": "stated"}
    },
    "maison_intelligence": {
        "affinity_signals": {"louis_vuitton": {"sentiment": "warming", "tier": "implied"}},
        "conquest_context": {"is_conquest": true, "from_brand": "celine", "barrier_stated": "logo_perception"}
    },
    "material_craftsmanship": {
        "mentioned_preferences": {"materials": [], "craftsmanship_signals": []},
        "rejection_signals": {"materials": ["monogram_canvas"], "tier": "stated"}
    },
    "purchase_context": {
        "type": {"value": "self_purchase", "tier": "stated"},
        "trigger": {"value": "upcoming_travel", "tier": "stated"},
        "decision_stage": {"value": "active_considering", "tier": "implied"},
        "timeframe": {"value": "weeks", "tier": "stated"},
        "gift_context": null,
        "corporate_context": null
    },
    "occasion": {
        "primary": {"value": "travel", "tier": "stated"},
        "destination": {"value": "tokyo_japan", "tier": "stated"},
        "cultural_relevance": {"value": "japanese_aesthetic_alignment", "tier": "inferred"},
        "recurrence": {"value": "likely_regular_traveler", "tier": "inferred"}
    },
    "budget_intelligence": {
        "stated_amount": {"value": 3000, "currency": "EUR", "tier": "stated"},
        "flexibility": {"value": "moderate", "tier": "implied"},
        "range_behavior": {"floor": 2500, "ceiling": 4000, "stretch_willing": true},
        "spending_tier": {"value": "premium", "tier": "inferred"},
        "discount_seeking": {"value": false, "tier": "implied"}
    },
    "lifestyle_signals": {
        "profession": {"value": "architect", "sector": "creative_design", "tier": "stated"},
        "interests_detected": {"architecture_design": {"score": 0.95, "tier": "stated"}},
        "travel_signals": {"destination_mentioned": "tokyo", "frequency": {"value": "regular", "tier": "inferred"}},
        "family_signals": {"detected": false, "tier": "absent"}
    },
    "client_relationship": {
        "status": {"value": "prospect_warm", "tier": "implied"},
        "loyalty_indicator": {"value": "fidele_other_brand", "tier": "stated"},
        "engagement_this_visit": {"value": "high", "tier": "implied"},
        "previous_visits_mentioned": {"value": false, "tier": "absent"}
    },
    "behavioral_markers": {
        "decision_style": {"value": "researched_deliberate", "tier": "implied"},
        "aesthetic_confidence": {"value": "high", "tier": "implied"},
        "openness_to_suggestion": {"value": "selective", "tier": "stated"},
        "communication_style": {"value": "direct_specific", "tier": "implied"},
        "brand_knowledge": {"value": "informed", "tier": "implied"},
        "patience_level": {"value": "patient", "tier": "implied"}
    },
    "cultural_context": {
        "language_of_interaction": "FR",
        "cultural_background": {"value": "western_european", "tier": "inferred"},
        "aesthetic_influences_detected": {"values": ["minimalism", "japanese_sensibility"], "tier": "inferred"}
    },
    "sentiment_extraction": {
        "overall_valence": {"value": 0.75, "tier": "implied"},
        "toward_lv": {"value": "shifting_positive", "tier": "stated"},
        "pain_points_stated": [{"issue": "logo_perception", "evidence": "trop logotés"}],
        "delight_signals": [{"signal": "product_match", "evidence": "bien aimé"}],
        "objections_stated": [{"objection": "brand_too_visible", "resolved": true, "resolution": "shown_aerogramme"}]
    },
    "service_mapping": {
        "advisor_actions_taken": {"actions": ["product_demonstration"], "products_shown": ["aerogramme_line"]},
        "follow_up_stated": {"channel": "whatsapp", "action": "recontact", "tier": "stated"},
        "additional_interests_noted": {"items": ["silk_scarves"], "tier": "stated"},
        "service_requests": {"appointment": null, "personalization": null}
    },
    "competitive_intelligence": {
        "brands_mentioned": [{"brand": "celine", "relationship": "current_loyal", "sentiment": "positive"}],
        "comparison_made": {"lv_vs": "celine", "lv_weakness_stated": "too_logo_heavy"},
        "switch_signals": {"probability": {"value": 0.60, "tier": "inferred"}}
    },
    "commercial_scoring": {
        "purchase_intent": {"value": 0.70, "tier": "inferred"},
        "basket_estimate": {"value": 3200, "currency": "EUR"},
        "cross_sell_detected": {"categories": ["fashion_accessories"], "tier": "stated"},
        "growth_potential": {"category_expansion": ["shoes", "rtw"], "trade_up": "moderate"},
        "lifetime_value": {"bracket": "high", "trajectory": "ascending"},
        "next_best_action": {"action": "whatsapp_curated_selection", "products": ["aerogramme_bags", "epi_leather"]}
    },
    "clustering_vector": {
        "dimensions": {
            "minimalist_vs_bold": 0.95,
            "discreet_vs_logo": 0.95,
            "functional_vs_decorative": 0.80,
            "classic_vs_trendy": 0.65,
            "deliberate_vs_impulse": 0.85,
            "self_vs_gift": 0.95,
            "premium_vs_ultra": 0.45,
            "engaged_vs_passive": 0.85,
            "conquest_vs_loyal": 0.90,
            "travel_vs_local": 0.85,
            "creative_vs_corporate": 0.80,
            "solo_vs_family": 0.70
        }
    },
    "extraction_metadata": {
        "model_version": "lvmh_cip_v2.0",
        "fields_stated": 28,
        "fields_implied": 19,
        "fields_inferred": 15,
        "fields_absent": 22,
        "confidence_overall": 0.78,
        "tier_distribution": {"stated": 0.33, "implied": 0.23, "inferred": 0.18, "absent": 0.26}
    },
    "gdpr_compliance": {
        "data_basis": "legitimate_interest_clienteling",
        "consent_scope": "profiling_for_personalization",
        "retention_policy": "24_months_from_last_interaction",
        "automated_decision": false,
        "human_review_required": true,
        "right_to_explanation": true,
        "pii_status": "zero_pii_all_behavioral_signals",
        "data_minimization": true
    }
}
```

## 6. Fine-tuned Models

### 6.1 DeBERTa Multi-task Classifier

**Base:** microsoft/deberta-v3-base (125M params, ~500MB)
**Purpose:** Extract the full 84-field profile from raw note text

**Architecture:**
```
Input: tokenized advisor note (512 tokens max)
          │
          ▼
    DeBERTa-v3-base (frozen first 6 layers, fine-tune last 6)
          │
    ┌─────┼──────┬──────────┬──────────┐
    ▼     ▼      ▼          ▼          ▼
 Head A  Head B  Head C   Head D    Head E
Concepts Sentim. Intent   Cluster   Evidence
(sigmoid)(regr.) (mixed)  (regr.)   (token)
```

**Training strategy:**
- Phase 1 (warm-up): Freeze backbone, train heads only — 500 steps, LR 1e-3
- Phase 2 (fine-tune): Unfreeze last 6 layers — 2,000 steps, LR 2e-5
- Phase 3 (calibration): Full model, low LR — 500 steps, LR 5e-6

**Loss function:**
```
loss = 0.35 * concept_bce + 0.15 * sentiment_mse + 0.20 * intent_mixed
     + 0.15 * cluster_mse + 0.15 * evidence_token_ce
```

**Training:** ~2-4 hours on Apple Silicon MPS, batch size 16
**Inference:** ~50ms/note

### 6.2 SentenceTransformer Domain Embeddings

**Base:** paraphrase-multilingual-MiniLM-L12-v2 (33M params, ~130MB, already in project)
**Purpose:** Domain-specific embeddings for clustering + product matching

**Training approach:** Contrastive learning with domain pairs
- ~10,000 pairs generated from 1,700 augmented notes × their labels
- Cross-lingual pairs from multilingual aliases
- Product descriptions paired with matching concepts

**Training:** ~1-2 hours on Apple Silicon MPS
**Inference:** ~20ms/note

## 7. Data Augmentation

100 real notes → ~2,200 training examples:

| Stage | Method | Input | Output |
|-------|--------|-------|--------|
| Ground-truth labeling | Semi-automated: Layer 1 (deterministic) pre-fills stated fields, human reviews and completes implied/inferred fields | 100 real notes | 100 labeled notes |
| Translation round-trip | Translate FR→EN→FR, FR→IT→FR via Ollama local model (Qwen or NLLB-200) | 100 labeled notes | ~400 variants (labels inherited) |
| Paraphrase generation | Local LLM rephrases in 3 styles: formal, conversational, brief | 500 notes | ~1,200 variants (labels inherited) |
| Synthetic generation | Pick concept combinations from ontology → generate notes via templates + local LLM | Ontology | ~500 synthetic (labels by construction) |

**Total: ~2,200 training examples** (100 real + 400 translated + 1,200 paraphrased + 500 synthetic)

**Ground-truth labeling workflow:**
1. Run Layer 1 (deterministic) on all 100 real notes → auto-fills stated fields
2. Human reviewer completes implied/inferred fields using a labeling UI or spreadsheet
3. Estimated effort: ~5 min/note × 100 = ~8 hours of labeling
4. Quality: spot-check 20 notes for inter-reviewer consistency

**Quality controls:**
- Augmented notes inherit ground-truth labels from parent (stages 2-3)
- Synthetic notes labeled by construction — concepts picked first, note generated to match
- Deduplication via cosine similarity > 0.95
- Human spot-check on ~50 random samples before training

**Label format per training example:** Full 84-field profile schema (Section 5.2)

**DeBERTa training risk mitigation:**
- With ~2,200 examples and ~200 concept output classes (not 800 — many concepts grouped into category-level classifiers), the data-to-class ratio is ~11:1, which is workable
- Phase 1 (head-only training) validates convergence before unfreezing backbone
- Per-head validation: if any head's val loss diverges, freeze it and continue training others
- Fallback: if DeBERTa fails to converge, reduce to 3 heads (concepts + sentiment + cluster vector) and use Layer 1 deterministic output for the remaining fields

## 8. Louis Vuitton Product Scraper

### 8.1 Target

Louis Vuitton website product listing pages (JSON API under the hood).

### 8.2 Categories

```
women/handbags, women/small-leather-goods, women/shoes,
women/accessories, women/ready-to-wear, women/jewelry,
women/watches, women/fragrances,
men/bags, men/small-leather-goods, men/shoes,
men/ready-to-wear, men/accessories, men/watches, men/fragrances,
travel/luggage, travel/travel-accessories,
gifts/for-women, gifts/for-men,
home/decorative-objects, home/art-of-living
```

### 8.3 Product Schema

```json
{
    "product_id": "nvprod4900086v",
    "name": "Capucines MM",
    "collection": "Capucines",
    "category": "handbags",
    "subcategory": "shoulder_bags",
    "price": 5150,
    "currency": "EUR",
    "gender": "women",
    "materials": ["taurillon_leather", "python_handle"],
    "colors": ["noir", "magnolia"],
    "description": "full product description text",
    "dimensions": "31.5 x 20 x 11 cm",
    "url": "...",
    "image_urls": ["..."],
    "availability": true,
    "tags_auto": [],
    "embedding": []
}
```

### 8.4 Post-scrape Enrichment

1. Embed each product description with fine-tuned SentenceTransformer
2. Auto-tag with ontology concepts via Aho-Corasick on description + name
3. Assign style signals, material affinity, occasion fit, price tier

### 8.5 Product-Client Match Formula

```
match_score(client, product) = clamp(0, 1,
    0.25 * cosine(client_embedding, product_embedding)
  + 0.25 * concept_overlap(client_tags, product_tags)
  + 0.20 * budget_fit(client_budget, product_price)
  + 0.10 * occasion_boost(client_occasion, product_gifting)
  + 0.15 * style_alignment(client_style, product_style)
  - 0.30 * rejection_match(client_rejects, product_attrs)
)
# All component scores normalized to [0, 1] before weighting
# Final score clamped to [0, 1]
```

**Future capability (not implemented):** Purchase history integration.

### 8.6 Expected Catalog Size

~2,000-4,000 products across all categories.

## 9. Knowledge Graph (Redesigned)

### 9.1 Node Types

| Type | Count | Description |
|------|-------|-------------|
| DimensionNode | 14 | Top-level ontology dimensions |
| CategoryNode | ~120 | Ontology categories |
| ConceptNode | ~800 | Leaf concepts with scores |
| ClientNode | N | Client profiles |
| ProductNode | ~3,000 | LV catalog items |
| ClusterNode | K | Client segments |

### 9.2 Edge Types

| Edge | From → To | Description |
|------|-----------|-------------|
| BELONGS_TO | concept → category → dimension | Ontology hierarchy |
| HAS_CONCEPT | client → concept | With score + tier |
| MATCHES_PRODUCT | client → product | With match_score |
| IN_CLUSTER | client → cluster | Cluster membership |
| SIMILAR_TO | client ↔ client | Cosine > threshold |
| IMPLIES | concept → concept | Logical implication |
| CONFLICTS | concept ↔ concept | Mutual exclusion |
| AMPLIFIES | concept + concept → score | Combined boost |
| CO_OCCURS | concept ↔ concept | Frequency-based |
| TAGGED_WITH | product → concept | Product tagging |
| CONQUEST_FROM | client → maison | Competitive opportunity |
| RECOMMENDED | client → product + action | Recommendations |

### 9.3 Scale

~15,000-20,000 nodes, ~80,000-120,000 edges (vs current 4,734 nodes, 4,175 edges).

## 10. Clustering

### 10.1 Cluster Vector (12 dimensions)

Each client gets a 12-dimensional behavioral vector, all normalized 0-1:

```
minimalist_vs_bold        — aesthetic spectrum
discreet_vs_logo          — branding preference
functional_vs_decorative  — product purpose
classic_vs_trendy         — style orientation
deliberate_vs_impulse     — decision behavior
self_vs_gift              — purchase motivation
premium_vs_ultra          — spending tier
engaged_vs_passive        — interaction depth
conquest_vs_loyal         — brand relationship
travel_vs_local           — lifestyle mobility
creative_vs_corporate     — professional identity
solo_vs_family            — social context
```

### 10.2 Algorithm

KMeans or DBSCAN on 12-dim vectors. Target silhouette > 0.3.

## 11. GDPR Compliance

- Anonymization runs before any extraction (Stage 2)
- Zero PII in profiles — only behavioral signals derived from interaction
- Every field has tier (stated/implied/inferred) + evidence (explainability)
- human_review_required: true on all profiles (Article 22 compliance)
- Audit log in PostgreSQL for all data access
- Right-to-erasure: DELETE /api/rgpd/erase/{client_id}
- Right-to-export: GET /api/rgpd/export/{client_id}
- Retention: 24 months from last interaction
- Data minimization: only behavioral signals retained

## 12. API Additions

```
GET  /api/clients/{id}/profile      — full 84-field profile
GET  /api/clients/{id}/products     — top product matches with scores
GET  /api/products                  — LV catalog (browsable, filterable)
GET  /api/products/{id}             — product detail + matched clients
GET  /api/clusters                  — cluster definitions with 12-dim descriptions
GET  /api/ontology                  — full hierarchical ontology (browsable)
POST /api/scraper/run               — trigger LV catalog refresh
GET  /api/graph/explore             — knowledge graph traversal
```

## 13. Integration with Existing System

### What stays
- FastAPI server (new endpoints added)
- React dashboard (enhanced with profile views + product recommendations)
- PostgreSQL / Neon (schema extended)
- Pipeline orchestrator run_all.py (stages reordered)
- GDPR anonymization module
- Docker support

### What changes
- Flat taxonomy → hierarchical ontology
- Aho-Corasick v1 (filler words) → v2 (800+ real concepts, stopword filtered)
- Pretrained SentenceTransformer → fine-tuned on luxury domain
- No ML inference → fine-tuned DeBERTa multi-task
- 6 tags per client → 84-field profile
- 242-node knowledge graph → 15k+ node graph
- No product layer → LV catalog with matching

### What's removed
- Phantom concept (CONCEPT_0000_401281)
- Qwen2.5:3b runtime dependency (replaced by fine-tuned DeBERTa)
- Flat taxonomy buckets
- Frequency-only tag selection

### Migration strategy
- Database: add new tables (products, ontology, full_profiles) alongside existing schema — no destructive changes
- Existing clients re-profiled through new pipeline on next run
- Dashboard: feature-flag new profile views, old views remain functional during transition
- No big-bang cutover — old and new pipelines can coexist until new one is validated

## 14. Error Handling & Degradation

Each extraction layer degrades gracefully:

| Layer | Failure Mode | Fallback |
|-------|-------------|----------|
| Layer 1 (Deterministic) | Unlikely — pure rule-based | Return empty stated fields |
| Layer 2 (DeBERTa) | Model load failure, inference error | Use Layer 1 results only — all fields marked "stated" or "absent" |
| Layer 3 (SentenceTransformer) | Model load failure | Skip product matching, use Layer 1+2 for clustering |
| Layer 4 (Cross-field) | Rule error | Return raw Layer 1+2+3 fields without composite scores |
| Product scraper | Site unavailable, anti-bot | Use cached catalog, manual CSV upload fallback |

Per-layer timeouts: Layer 1 (50ms), Layer 2 (200ms), Layer 3 (100ms), Layer 4 (50ms).
Circuit breaker: if >10% of notes fail in any layer, skip that layer for the batch and log warning.

## 15. Negation Handling

**Detection patterns (multilingual):**
- FR: "pas de", "sans", "ne...pas", "ne...jamais", "aucun"
- EN: "no", "not", "never", "without", "don't", "doesn't"
- IT: "non", "senza", "nessun"
- ES: "no", "sin", "ningún", "nunca"
- DE: "nicht", "kein", "ohne", "nie"

**Scope rules:** Negation applies to the next noun phrase only (dependency-free heuristic using window of 3-5 tokens after negation word).

**Behavior:** Concepts detected under negation are added to `rejection_signals` with `tier: "stated"`, not to affinity scores. Example: "pas de monogramme" → `material_craftsmanship.rejection_signals.materials: ["monogram_canvas"]`.

## 16. Naming Conventions

Canonical dimension names used everywhere (ontology, profile schema, API, DB):

```
product_intelligence, maison_intelligence, material_craftsmanship,
purchase_context, occasion, budget_intelligence, lifestyle_signals,
client_relationship, behavioral_markers, cultural_context,
sentiment_extraction, service_mapping, competitive_intelligence,
commercial_scoring
```

The clustering vector dimensions use a separate namespace (`minimalist_vs_bold`, etc.) as they represent behavioral spectrums, not ontology dimensions.

## 17. Testing Strategy

| Level | What | Target |
|-------|------|--------|
| Unit tests | Each extraction layer independently | 90% coverage on Layer 1 rules, Layer 4 rules |
| Integration tests | Full pipeline end-to-end on 10 test notes | All 14 dimensions populated |
| Model validation | DeBERTa: per-head F1 on held-out set (20% of training data) | Concept head F1 > 0.6, sentiment MSE < 0.1 |
| Embedding validation | Silhouette score on clustering | > 0.3 |
| Product matching | Match relevance on 20 manually scored client-product pairs | Precision > 0.7 |
| Performance | End-to-end latency per note | < 200ms |
| GDPR | Verify zero PII in all profile outputs | 100% pass on PII regex scan |
| Regression | Compare new vs old pipeline on same 100 notes | More unique tags, higher silhouette |
