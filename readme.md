# LVMH Voice-to-Tag — Vector Profiles

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.14+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/NLP-Hybrid-green.svg" alt="NLP">
  <img src="https://img.shields.io/badge/Languages-12+-orange.svg" alt="Languages">
  <img src="https://img.shields.io/badge/LLM-Qwen2.5-purple.svg" alt="Qwen LLM">
  <img src="https://img.shields.io/badge/Vocabulary-575%20concepts-teal.svg" alt="Vocabulary">
  <img src="https://img.shields.io/badge/Concepts-1796%20detected-success.svg" alt="Detected">
</p>

Pipeline **hybride (Rule-based + LLM)** et multilingue qui transforme les transcriptions des Conseillers de Vente en profils clients actionnables et recommandations personnalisées.

**✨ Nouveau (Production v1.0):** 
- **Qwen LLM Enhancement**: Extraction sémantique avec Qwen2.5:3b (342 concepts supplémentaires)
- **Budget Detection**: Extraction regex des montants (100% coverage)
- **Per-Client Tags**: 12 tags personnalisés par client (budget en premier)
- **Big O Performance**: Tests de complexité validés (11/11 passing)
- **Anonymisation RGPD/GDPR automatique** 🔒

---

## 🎯 Objectifs

Transformer automatiquement les notes vocales des conseillers en :
- **Tags structurés** (préférences, occasions, contraintes, budgets)
- **Profils clients** segmentés par similarité (12 tags personnalisés par client)
- **Actions recommandées** personnalisées
- **Visualisation 3D** interactive de l'espace client
- **Extraction sémantique** via Qwen LLM (au-delà des règles)

### 🧠 Approche Hybride

Le système combine trois méthodes complémentaires:

1. **Rule-Based Matching** (Aho-Corasick) — 1,454 concepts
   - Correspondance rapide d'alias multilingues
   - Déterministe et explicable
   - Complexité O(N+M) — linéaire

2. **Regex Patterns** — Extraction de budgets
   - Montants, ranges, devises
   - 100% coverage sur notes avec budget
   - Pattern: `budget/price 3-4k`, `€1500`, `25K+`

3. **Qwen LLM Enhancement** (Qwen2.5:3b) — 342 concepts supplémentaires
   - Extraction sémantique des concepts que les règles manquent
   - Optimisé: ~10s/note, 99% success rate
   - Filtres qualité: anti-hallucination, snake_case, duplicates
   - Confidence moyenne: 0.91

**Total: 1,796 concepts détectés** (1,454 rule-based + 342 LLM)

## 🔒 Conformité RGPD/GDPR

Le pipeline inclut un **module d'anonymisation automatique** qui détecte et supprime les informations personnelles sensibles :
- Noms, emails, téléphones
- Adresses postales
- Cartes bancaires, IBAN
- Numéros d'identité
- Dates de naissance

Les insights métier (préférences produits, intentions, contextes) sont **préservés** pour l'analyse.

📖 **Documentation complète:** [docs/ANONYMIZATION.md](docs/ANONYMIZATION.md)

```bash
# Activer/désactiver l'anonymisation (activée par défaut)
ENABLE_ANONYMIZATION=true python -m server.run_all

# Mode agressif (détecte plus de noms, plus de faux positifs)
ANONYMIZATION_AGGRESSIVE=true python -m server.run_all
```

---

## 🚀 Démarrage Rapide

### Option 1: Exécutable (Recommandé)

**macOS:**
```bash
# Double-cliquez sur le fichier ou exécutez:
./LVMH_Pipeline.command
```

**Windows:**
```cmd
# Double-cliquez sur le fichier ou exécutez:
LVMH_Pipeline.bat
```

### Option 2: Ligne de commande

```bash
# 1. Créer l'environnement virtuel
make venv

# 2. Télécharger le modèle d'embedding
make setup-models-local

# 3. Placer votre CSV dans data/raw/

# 4. Lancer le pipeline
make dev
```

### Option 3: Docker (Reproductibilité garantie)

```bash
# Construire et lancer
make build && make run
```

---

## 🧠 Entraînement du Vocabulaire

Le pipeline utilise un vocabulaire entraînable de **575 concepts** avec **9,003 aliases** multilingues (12+ langues).

### Statistiques Actuelles (Production v1.0)

| Bucket | Concepts | Exemples | LLM Boost |
|--------|----------|----------|-----------|
| **preferences** | 167 | marques, matériaux, styles | +34% detection |
| **intent** | 71 | émotions, intentions d'achat | +28% detection |
| **lifestyle** | 71 | famille, personnalité, VIP | +31% detection |
| **occasion** | 36 | fêtes, événements, étapes | +25% detection |
| **constraints** | 20 | **budget**, délais, canaux | **100% coverage** |
| **next_action** | 19 | rendez-vous, réparation | +22% detection |

### Extraction de Budgets

Le système détecte automatiquement les montants budgétaires via regex:

```python
# Patterns détectés:
"budget 3-4k"           → BUDGET_AMOUNT: budget 3-4k
"presupuesto 25-30k"    → BUDGET_AMOUNT: budget 25-30k  
"prix 1500€"            → BUDGET_AMOUNT: budget 1500€
"price around 40K+"     → BUDGET_AMOUNT: budget 40k+
```

**Coverage: 100/100 notes** | Range: €1,500 - €40,000+ | Moyenne: €13,070

### Commandes CLI

```bash
# Voir les statistiques du vocabulaire
python -m server.server.train_vocabulary stats

# Ajouter un mot-clé manuellement
python -m server.server.train_vocabulary add "terme" "Label FR" "bucket" --aliases "alias1,alias2,别名"

# Importer des mots-clés depuis un fichier JSON
python -m server.server.train_vocabulary import taxonomy/training_keywords.json

# Charger les mots-clés prédéfinis
python -m server.server.train_vocabulary load-predefined

# Lister les concepts d'un bucket
python -m server.server.train_vocabulary list --bucket preferences
```

### Format JSON pour Import

```json
[
  {
    "term": "hermès",
    "label": "Hermès",
    "bucket": "preferences",
    "aliases": ["hermes", "エルメス", "爱马仕", "에르메스"]
  },
  {
    "term": "anniversary",
    "label": "Anniversaire",
    "bucket": "occasion",
    "aliases": ["anniversaire", "compleanno", "cumpleaños", "記念日", "기념일"]
  }
]
```

### Langues Supportées dans le Vocabulaire

| Code | Langue | Code | Langue |
|------|--------|------|--------|
| EN | English | ZH | 中文 |
| FR | Français | JA | 日本語 |
| IT | Italiano | KO | 한국어 |
| ES | Español | RU | Русский |
| DE | Deutsch | AR | العربية |
| PT | Português | NL | Nederlands |

---

## 📊 Architecture du Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    LVMH Voice-to-Tag Pipeline (Hybrid v1.0)                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │  STAGE 1 │───▶│  STAGE 2 │───▶│  STAGE 3 │───▶│  STAGE 4 │              │
│  │  Ingest  │    │ Extract  │    │  Lexicon │    │ Concepts │              │
│  │          │    │          │    │          │    │          │              │
│  │ CSV ──▶  │    │Rule-Based│    │ Embedding│    │ Aho-     │              │
│  │ Parquet  │    │  + LLM   │    │Clustering│    │Corasick  │              │
│  │          │    │  Qwen2.5 │    │          │    │+ Regex   │              │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘              │
│       │               │               │               │                     │
│       ▼               ▼               ▼               ▼                     │
│  notes_clean     qwen_enhanced   lexicon_v1.csv  note_concepts             │
│   .parquet      _concepts.csv   taxonomy_v1.json   .csv                    │
│                  (342 LLM)        (575 concepts)  (1,796 total)            │
│                                                                              │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │  STAGE 5 │───▶│  STAGE 6 │───▶│  STAGE 7 │───▶│  STAGE 8 │              │
│  │ Vectors  │    │ Profiles │    │ Actions  │    │Predictions│             │
│  │          │    │          │    │          │    │          │              │
│  │ Sentence │    │  KMeans  │    │ Playbook │    │ Purchase │              │
│  │Transformer│   │ Per-Client│   │ Matching │    │Churn/CLV │              │
│  │          │    │  12 Tags │    │          │    │          │              │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘              │
│       │               │               │               │                     │
│       ▼               ▼               ▼               ▼                     │
│  note_vectors    client_profiles  recommended    ml_predictions            │
│   .parquet          .csv         _actions.csv       .csv                   │
│                  (budget 1st)                                               │
│                                                                              │
│  ┌──────────┐    ┌──────────┐                                               │
│  │  STAGE 9 │───▶│ STAGE 10 │                                               │
│  │   K.G.   │    │Dashboard │                                               │
│  │          │    │          │                                               │
│  │ Neo4j /  │    │ React UI │                                               │
│  │ Cytoscape│    │ + API    │                                               │
│  └──────────┘    └──────────┘                                               │
│       │               │                                                     │
│       ▼               ▼                                                     │
│  knowledge_graph  dashboard/                                                │
│     .json        src/data.json                                              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 🔬 Extraction Methods Comparison

| Method | Concepts | Speed | Precision | Coverage |
|--------|----------|-------|-----------|----------|
| **Aho-Corasick** | 1,454 | ⚡ <1ms/note | ✅ 100% | ⚠️ 75% (missed semantic) |
| **Regex (Budget)** | 100 | ⚡ <1ms/note | ✅ 100% | ✅ 100% (budget notes) |
| **Qwen LLM** | 342 | 🐢 ~10s/note | ⚠️ 91% | ✅ 99% (semantic gaps) |
| **Combined** | **1,796** | ⚡ ~10s/note | ✅ 96% | ✅ 100% |

---

## 📁 Structure des Fichiers

```
BDD2-LVMH/
├── 📄 LVMH_Pipeline.command    # Lanceur macOS (double-clic)
├── 📄 LVMH_Pipeline.bat        # Lanceur Windows (double-clic)
├── 📄 makefile                 # Commandes make
├── 📄 requirements.txt         # Dépendances Python
├── 📄 Dockerfile               # Container Docker
├── 📄 ARCHITECTURE.md          # Documentation architecture
│
├── 📂 data/
│   ├── 📂 input/               # ← Placez votre CSV ici
│   ├── 📂 processed/           # Données intermédiaires
│   └── 📂 outputs/             # Résultats finaux
│
├── 📂 taxonomy/                # Lexique et vocabulaire entraîné
│   ├── vocabulary.json         # Vocabulaire (384 concepts)
│   ├── lexicon_v1.json         # Lexique synchronisé
│   └── taxonomy_v1.json        # Taxonomie par catégories
│
├── 📂 activations/             # Playbooks d'actions (YAML)
├── 📂 models/                  # Modèle SentenceTransformer
│
├── 📂 server/                  # 🔧 Backend - Traitement de données
│   ├── run_all.py              # Orchestrateur principal (10 stages)
│   ├── api_server.py           # FastAPI REST API (port 8000)
│   ├── 📂 shared/              # Config & utilitaires
│   │   ├── config.py           # Configuration centrale
│   │   ├── utils.py            # Fonctions helper
│   │   ├── knowledge_graph.py  # Construction du graphe
│   │   └── generate_dashboard.py # Génération dashboard
│   ├── 📂 ingest/              # Étape 1: Ingestion CSV
│   ├── 📂 extract/             # Étapes 2-4: Extraction concepts
│   │   ├── detect_concepts.py  # Rule-based (Aho-Corasick + Regex)
│   │   └── qwen_enhance.py     # LLM enhancement (Qwen2.5:3b)
│   ├── 📂 lexicon/             # Étape 3: Construction lexique
│   ├── 📂 embeddings/          # Étapes 5 & 8: Vecteurs & UMAP 3D
│   ├── 📂 profiling/           # Étape 6: Segmentation clients
│   │   └── segment_clients.py  # KMeans + per-client tags (12 tags)
│   ├── 📂 actions/             # Étape 7: Recommandations
│   ├── 📂 analytics/           # Étape 8: ML predictions
│   ├── 📂 db/                  # Database (Neon PostgreSQL)
│   │   ├── crud.py             # CRUD operations
│   │   └── sync.py             # DB sync (100 clients)
│   └── 📂 adaptive/            # Pipeline adaptatif
│
├── 📂 client/                  # 🎨 Frontend - Interface utilisateur
│   └── 📂 app/                 # Application dashboard
│       ├── dashboard.html      # Dashboard unifié (KG + 3D)
│       ├── kg_obsidian.html    # Graphe de connaissance
│       ├── embedding_space_3d.html # Espace vectoriel 3D
│       └── cytoscape.min.js    # Bibliothèque visualisation
│
└── 📂 docs/                    # Documentation
```

---

## 📥 Format d'Entrée

### Mode Standard (Format LVMH)

Fichier CSV avec les colonnes suivantes:

| Colonne | Type | Description | Exemple |
|---------|------|-------------|---------|
| `ID` | string | Identifiant unique | `CA_001` |
| `Date` | string | Date d'interaction | `2024-01-15` |
| `Duration` | string | Durée | `35 min` |
| `Language` | string | Langue (FR/EN/IT/ES/DE) | `FR` |
| `Length` | string | Longueur | `medium` |
| `Transcription` | string | Contenu textuel | `Mme Dupont...` |

### Mode Adaptatif (N'importe quel CSV)

Le pipeline peut analyser automatiquement n'importe quel fichier CSV:

```bash
# Analyser la structure d'un CSV
python -m server.run_all --csv data/input/my_data.csv --analyze-only

# Lancer avec détection automatique des colonnes
python -m server.run_all --csv data/input/my_data.csv

# Spécifier les colonnes manuellement
python -m server.run_all --csv data/input/my_data.csv --text-column "description" --id-column "client_id"
```

Le système détecte automatiquement:
- **Colonne texte**: Plus longue moyenne de caractères
- **Colonne ID**: Noms comme `id`, `client_id`, `code`
- **Colonne langue**: Noms comme `lang`, `language`, `langue`
- **Colonne date**: Formats date détectés automatiquement

---

## 📤 Fichiers de Sortie

### Données Traitées

| Fichier | Description |
|---------|-------------|
| `notes_clean.parquet` | Notes nettoyées et normalisées (100 notes) |
| `note_concepts_rules_only.csv` | Concepts rule-based uniquement (1,454) |
| `note_concepts_enhanced.csv` | Concepts rule-based + LLM (1,796) |
| `qwen_checkpoints/*.json` | Checkpoints Qwen par note (resume capability) |

### Taxonomie

| Fichier | Description |
|---------|-------------|
| `vocabulary.json` | **Vocabulaire entraîné** (575 concepts, 9,003 aliases) |
| `lexicon_v1.json` | Lexique synchronisé avec alias et fréquences |
| `lexicon_v1.csv` | Version CSV du lexique |
| `taxonomy_v1.json` | Taxonomie par catégories (6 buckets) |

### Résultats

| Fichier | Description |
|---------|-------------|
| `note_concepts.csv` | **1,796 concepts** (1,454 rule + 342 LLM) avec positions |
| `note_vectors.parquet` | Embeddings 384 dimensions par note |
| `client_profiles.csv` | Segments clients avec **12 tags personnalisés** (budget 1st) |
| `client_summaries.json` | 99 résumés LLM avec urgency/sentiment |
| `recommended_actions.csv` | Actions recommandées par client |
| `ml_predictions.csv` | Purchase probability, churn risk, CLV |
| `embedding_space_3d.html` | Visualisation 3D interactive |

---

## 🔧 Technologies Utilisées

### Extraction Hybride (Rule-Based + LLM)

**Rule-Based (Déterministe)**:
- **Aho-Corasick** - Correspondance multi-pattern O(N+M)
- **Regex** - Extraction de budgets (montants, ranges, devises)
- **TF-IDF** - Term Frequency-Inverse Document Frequency

**LLM Enhancement (Sémantique)**:
- **Qwen2.5:3b** via Ollama - 1.9GB, 3.1B params
  - Context: 2048 tokens
  - Output: 512 tokens  
  - Timeout: 45s × 2 retries
  - Format: JSON strict
  - Performance: ~10s/note, 99% success
  - Confidence: 0.91 avg

### Embeddings & Clustering
- **SentenceTransformers** - `paraphrase-multilingual-MiniLM-L12-v2`
  - Support multilingue (50+ langues)
  - 384 dimensions
  - Optimisé pour similarité sémantique
- **KMeans** - Segmentation clients (7 clusters)
- **Silhouette Score** - Validation clustering (0.0573)

### Machine Learning
- **scikit-learn** - RandomForestClassifier, GradientBoostingRegressor
  - Purchase probability prediction
  - Churn risk prediction
  - Customer Lifetime Value (CLV) estimation

### Database & API
- **Neon PostgreSQL** - Cloud-hosted database
  - 100 clients synced
  - 1,919 concept extractions
  - Per-client tags (budget first)
- **FastAPI** - REST API server (port 8000)
  - `/api/clients` - List clients
  - `/api/clients/{id}` - Client 360° view
  - `/api/predictions` - ML predictions
  - `/api/dashboard` - Dashboard data

### Visualisation
- **UMAP** - Réduction dimensionnelle non-linéaire
- **Plotly** - Graphiques 3D interactifs
- **React/TypeScript** - Dashboard frontend (Vite)

### Performance Testing
- **pytest** - Unit testing framework
- **Big O Complexity Tests** - 11 tests validating scalability
  - All passing in 5.15s
  - Linear algorithms: O(N) with b < 1.6
  - Quadratic operations: O(N²) with b < 2.8

### Déterminisme
- Seeds fixes: `RANDOM_SEED=42`, `NUMPY_SEED=42`
- Reproductibilité garantie (hors LLM)

---

## ⚙️ Configuration

Fichier `server/shared/config.py`:

```python
# Seeds pour reproductibilité
RANDOM_SEED = 42
NUMPY_SEED = 42
SKLEARN_RANDOM_STATE = 42
UMAP_RANDOM_STATE = 42

# Modèle d'embedding
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Clustering
N_CLUSTERS = 7                      # Segments clients (KMeans)
SILHOUETTE_THRESHOLD = 0.0          # Validation clustering

# Extraction
MIN_CANDIDATE_FREQ = 2              # Fréquence minimale

# LLM Enhancement (Qwen2.5:3b)
QWEN_MODEL = "qwen2.5:3b"
QWEN_TIMEOUT = 45                   # seconds per retry
QWEN_RETRIES = 2
QWEN_NUM_CTX = 2048                 # context window
QWEN_NUM_PREDICT = 512              # max output tokens
QWEN_TEMPERATURE = 0.15             # low for consistency
QWEN_REPEAT_PENALTY = 1.3           # avoid repetition
```

### Variables d'Environnement (.env)

```bash
# Database
DATABASE_URL=postgresql://user:pass@host/db

# Ollama
OLLAMA_BASE_URL=http://localhost:11434

# Anonymization (GDPR)
ENABLE_ANONYMIZATION=true
ANONYMIZATION_AGGRESSIVE=false

# Features
ENABLE_QWEN_ENHANCEMENT=true        # LLM semantic extraction
ENABLE_BUDGET_EXTRACTION=true       # Regex budget detection
```

---

## 🎨 Visualisation 3D

La visualisation interactive (`demo/embedding_space_3d.html`) offre:

- **Carte de similarité**: Clients proches = profils similaires
- **Couleurs par segment**: 7 segments distincts avec légende
- **Hover détaillé**: Client ID, profil complet, note ID
- **Filtrage**: Cliquez sur la légende pour afficher/masquer des segments
- **Rotation 3D**: Explorez l'espace client sous tous les angles

### Axes Sémantiques
| Axe | Signification |
|-----|---------------|
| **X** | ← Classique \| Moderne → |
| **Y** | ← Quotidien \| Événements → |
| **Z** | ← Budget \| Premium → |

---

## 📋 Playbooks d'Actions

Les recommandations sont basées sur 10 playbooks configurables (`activations/playbooks.yml`):

| Action | Déclencheurs |
|--------|--------------|
| VIP Event Invitation | client_vip, events |
| New Collection Preview | style, fashion |
| Gift Occasion Follow-up | cadeau, anniversaire |
| Follow-up Appointment | rappeler, next_action |
| Budget-Sensitive Presentation | budget |
| Family Package Offer | famille, enfants |
| Travel Collection | voyage, travel |
| Anniversary Special | anniversaire, mariage |
| Dietary Accommodation | allergie, végétarien, végan |
| Personalized Recommendation | lifestyle, preferences |

---

## 🌍 Langues Supportées

### Traitement des Transcriptions

| Code | Langue | Exemple de Transcription |
|------|--------|--------------------------|
| FR | Français | "Mme Dupont cherche un cadeau pour son mari" |
| EN | English | "Mrs. Anderson is looking for elegant pieces" |
| IT | Italiano | "Signora Rossi cerca regali per la famiglia" |
| ES | Español | "Sra. García busca artículos de lujo" |
| DE | Deutsch | "Frau Schmidt sucht Geschenke für ihren Mann" |

### Vocabulaire Multilingue (12+ langues)

Le vocabulaire entraîné supporte des alias dans:
- 🇬🇧 English, 🇫🇷 Français, 🇮🇹 Italiano, 🇪🇸 Español, 🇩🇪 Deutsch
- 🇵🇹 Português, 🇳🇱 Nederlands, 🇷🇺 Русский, 🇸🇦 العربية
- 🇨🇳 中文, 🇯🇵 日本語, 🇰🇷 한국어

> **Note**: Les sorties sont **standardisées en français** pour cohérence.

---

## 🐳 Docker

### Construction
```bash
docker build -t lvmh-pipeline .
```

### Exécution
```bash
docker run -v $(pwd)/data:/app/data -v $(pwd)/taxonomy:/app/taxonomy lvmh-pipeline
```

---

## 📈 Performance

| Métrique | Valeur |
|----------|--------|
| **Pipeline total** | ~65s (100 notes, 10 stages) |
| **Qwen Enhancement** | ~10s/note (99% success) |
| **Rule-based Detection** | <1ms/note |
| **Budget Extraction** | <1ms/note (100% coverage) |
| **Vocabulaire** | **575 concepts, 9,003 aliases** |
| **Concepts détectés** | **1,796 total** (1,454 rule + 342 LLM) |
| **LLM Confidence** | 0.91 average |
| **Budget Range** | €1,500 - €40,000+ (avg €13,070) |
| **Segments clients** | 7 clusters (silhouette 0.0573) |
| **Tags per client** | 12 personalized (budget first) |
| **Big O Tests** | 11/11 passing (5.15s) |
| **Database** | 100 clients synced (Neon PostgreSQL) |

---

## 🔍 Exemple de Résultat

### Entrée (Transcription FR)
```
Mme Rousseau, avocate d'affaires, cliente occasionnelle. Cherche cadeau 
anniversaire mari, mars prochain. Budget flexible autour de 5000€. 
Mari collectionne montres vintage, joue au golf...
```

### Sortie (Profil avec Tags Personnalisés)
```yaml
Client: CA_001
Segment: "Budget Amount | Mariage | Europe Travel"
Cluster: 0 (13 clients similaires)

Tags Personnalisés (12):
  1. budget 5000€           ← Budget détecté en premier
  2. anniversaire           ← Occasion
  3. cadeau                 ← Intent
  4. collectionne           ← Lifestyle
  5. golf                   ← Lifestyle/Preferences
  6. vintage                ← Preferences
  7. mari                   ← Context
  8. mars                   ← Timeframe
  9. flexible               ← Constraints
  10. avocate               ← Profession
  11. occasionnelle         ← Client Type
  12. montres               ← Product Interest

Concepts détectés (total: 15):
  Rule-based (12):
    - anniversaire (OCCASION)
    - cadeau (INTENT)
    - budget (CONSTRAINTS) → extracted: 5000€
    - collectionne (LIFESTYLE)
    - golf (PREFERENCES)
    - vintage (PREFERENCES)
    ...
  
  LLM-enhanced (3):
    - "luxury timepiece interest" (conf: 0.92)
    - "spring event planning" (conf: 0.88)
    - "professional woman gifting" (conf: 0.85)

Actions recommandées:
  1. Gift Occasion Follow-up (score: 0.95)
     Triggers: anniversaire, cadeau, mars
  2. VIP Watch Collection Preview (score: 0.92)
     Triggers: collectionne, montres, vintage, budget 5000€
  3. Anniversary Special (score: 0.90)
     Triggers: anniversaire, mari
  4. Golf Accessories Recommendation (score: 0.85)
     Triggers: golf, lifestyle, budget

ML Predictions:
  - Purchase Probability: 0.78 (High)
  - Churn Risk: 0.12 (Low)
  - Estimated CLV: €12,450
```

---

## 🛠️ Dépannage

### Le pipeline ne trouve pas le modèle
```bash
make setup-models-local
```

### Erreur de mémoire avec UMAP
Le pipeline bascule automatiquement sur PCA si UMAP échoue.

### Permissions sur macOS
```bash
chmod +x LVMH_Pipeline.command
```

### Le launcher ne s'ouvre pas (macOS)
```bash
xattr -d com.apple.quarantine LVMH_Pipeline.command
```

---

## 📚 Documentation Complémentaire

- [docs/prd.md](docs/prd.md) - Product Requirements Document
- [docs/agents.md](docs/agents.md) - Documentation des agents

---

## 📄 Licence

Proprietary - LVMH © 2026

---

<p align="center">
  <i>Développé pour LVMH - Transformation des interactions clients en insights actionnables</i>
</p>
