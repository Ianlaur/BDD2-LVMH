# LVMH Voice-to-Tag — Vector Profiles

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/NLP-Deterministic-green.svg" alt="NLP">
  <img src="https://img.shields.io/badge/Languages-FR%20|%20EN%20|%20IT%20|%20ES%20|%20DE-orange.svg" alt="Languages">
  <img src="https://img.shields.io/badge/LLM-None%20Required-red.svg" alt="No LLM">
</p>

Pipeline **déterministe (sans LLM)** et multilingue qui transforme les transcriptions des Conseillers de Vente en profils clients actionnables et recommandations personnalisées.

---

## 🎯 Objectifs

Transformer automatiquement les notes vocales des conseillers en :
- **Tags structurés** (préférences, occasions, contraintes)
- **Profils clients** segmentés par similarité
- **Actions recommandées** personnalisées
- **Visualisation 3D** interactive de l'espace client

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

## 📊 Architecture du Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           LVMH Voice-to-Tag Pipeline                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │  STAGE 1 │───▶│  STAGE 2 │───▶│  STAGE 3 │───▶│  STAGE 4 │              │
│  │  Ingest  │    │ Candidates│    │  Lexicon │    │ Concepts │              │
│  │          │    │           │    │          │    │          │              │
│  │ CSV ──▶  │    │ YAKE/RAKE │    │ Embedding│    │  Alias   │              │
│  │ Parquet  │    │  TF-IDF   │    │ Clustering│   │ Matching │              │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘              │
│       │               │               │               │                     │
│       ▼               ▼               ▼               ▼                     │
│  notes_clean     candidates.csv  lexicon_v1.csv  note_concepts             │
│   .parquet                       taxonomy_v1.json   .csv                   │
│                                                                              │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │  STAGE 5 │───▶│  STAGE 6 │───▶│  STAGE 7 │───▶│  STAGE 8 │              │
│  │ Vectors  │    │ Profiles │    │ Actions  │    │   3D     │              │
│  │          │    │          │    │          │    │Projection│              │
│  │ Sentence │    │  KMeans  │    │ Playbook │    │   UMAP   │              │
│  │Transformer│   │ Clustering│   │ Matching │    │  Plotly  │              │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘              │
│       │               │               │               │                     │
│       ▼               ▼               ▼               ▼                     │
│  note_vectors    client_profiles  recommended     embedding_               │
│   .parquet          .csv         _actions.csv    space_3d.html             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Structure des Fichiers

```
BDD2-LVMH/
├── 📄 LVMH_Pipeline.command    # Lanceur macOS (double-clic)
├── 📄 LVMH_Pipeline.bat        # Lanceur Windows (double-clic)
├── 📄 makefile                 # Commandes make
├── 📄 requirements.txt         # Dépendances Python
├── 📄 Dockerfile               # Container Docker
│
├── 📂 data/
│   ├── 📂 raw/                 # ← Placez votre CSV ici
│   ├── 📂 processed/           # Données intermédiaires
│   └── 📂 outputs/             # Résultats finaux
│
├── 📂 taxonomy/                # Lexique et taxonomie générés
├── 📂 activations/             # Playbooks d'actions (YAML)
├── 📂 demo/                    # Visualisation 3D interactive
├── 📂 models/                  # Modèle SentenceTransformer
│
├── 📂 src/                     # Code source
│   ├── config.py               # Configuration centrale
│   ├── utils.py                # Utilitaires
│   ├── run_all.py              # Orchestrateur principal
│   ├── 📂 ingest/              # Étape 1: Ingestion
│   ├── 📂 extract/             # Étapes 2 & 4: Extraction
│   ├── 📂 lexicon/             # Étape 3: Construction lexique
│   ├── 📂 embeddings/          # Étapes 5 & 8: Vecteurs & 3D
│   ├── 📂 profiling/           # Étape 6: Segmentation
│   └── 📂 actions/             # Étape 7: Recommandations
│
└── 📂 docs/                    # Documentation
```

---

## 📥 Format d'Entrée

Fichier CSV avec les colonnes suivantes:

| Colonne | Type | Description | Exemple |
|---------|------|-------------|---------|
| `ID` | string | Identifiant unique | `CA_001` |
| `Date` | string | Date d'interaction | `2024-01-15` |
| `Duration` | string | Durée | `35 min` |
| `Language` | string | Langue (FR/EN/IT/ES/DE) | `FR` |
| `Length` | string | Longueur | `medium` |
| `Transcription` | string | Contenu textuel | `Mme Dupont...` |

---

## 📤 Fichiers de Sortie

### Données Traitées

| Fichier | Description |
|---------|-------------|
| `notes_clean.parquet` | Notes nettoyées et normalisées |
| `candidates.csv` | Candidats extraits (mots-clés, entités) |

### Taxonomie

| Fichier | Description |
|---------|-------------|
| `lexicon_v1.csv` | Lexique avec ~130 concepts, alias, fréquences |
| `taxonomy_v1.json` | Taxonomie par catégories (intent, occasion, preferences, constraints, lifestyle, next_action) |

### Résultats

| Fichier | Description |
|---------|-------------|
| `note_concepts.csv` | Correspondances concept ↔ note avec positions |
| `note_vectors.parquet` | Embeddings 384 dimensions par note |
| `client_profiles.csv` | Segments clients avec profils et confiance |
| `recommended_actions.csv` | Actions recommandées par client |
| `embedding_space_3d.html` | Visualisation 3D interactive |

---

## 🔧 Technologies Utilisées

### Extraction de Mots-Clés (Sans LLM)
- **YAKE** - Yet Another Keyword Extractor
- **RAKE-NLTK** - Rapid Automatic Keyword Extraction
- **TF-IDF** - Term Frequency-Inverse Document Frequency

### Embeddings & Clustering
- **SentenceTransformers** - `paraphrase-multilingual-MiniLM-L12-v2`
  - Support multilingue (50+ langues)
  - 384 dimensions
  - Optimisé pour similarité sémantique
- **Agglomerative Clustering** - Regroupement hiérarchique (distance cosinus)
- **KMeans** - Segmentation clients

### Visualisation
- **UMAP** - Réduction dimensionnelle non-linéaire
- **Plotly** - Graphiques 3D interactifs

### Déterminisme
- Seeds fixes: `RANDOM_SEED=42`, `NUMPY_SEED=42`
- Pas d'appels API externes
- Reproductibilité garantie

---

## ⚙️ Configuration

Fichier `src/config.py`:

```python
# Seeds pour reproductibilité
RANDOM_SEED = 42
NUMPY_SEED = 42
SKLEARN_RANDOM_STATE = 42
UMAP_RANDOM_STATE = 42

# Modèle d'embedding
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Clustering
CLUSTER_DISTANCE_THRESHOLD = 0.35  # Seuil similarité cosinus
N_CLUSTERS = 7                      # Segments clients

# Extraction
MIN_CANDIDATE_FREQ = 2              # Fréquence minimale
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

| Code | Langue | Exemple de Transcription |
|------|--------|--------------------------|
| FR | Français | "Mme Dupont cherche un cadeau pour son mari" |
| EN | English | "Mrs. Anderson is looking for elegant pieces" |
| IT | Italiano | "Signora Rossi cerca regali per la famiglia" |
| ES | Español | "Sra. García busca artículos de lujo" |
| DE | Deutsch | "Frau Schmidt sucht Geschenke für ihren Mann" |

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
| Temps d'exécution | ~18 secondes (100 notes) |
| Couverture concepts | 100% des notes |
| Concepts extraits | ~130 |
| Candidats filtrés | ~480 |
| Segments clients | 7 |

---

## 🔍 Exemple de Résultat

### Entrée (Transcription FR)
```
Mme Rousseau, avocate d'affaires, cliente occasionnelle. Cherche cadeau 
anniversaire mari, mars prochain. Budget flexible autour de 5000€. 
Mari collectionne montres vintage, joue au golf...
```

### Sortie (Profil)
```yaml
Client: CA_001
Segment: "Élégant | Follow | Pratique"
Concepts détectés:
  - anniversaire
  - cadeau
  - budget @0 (5000€)
  - collectionne
  - golf
  - vintage
Actions recommandées:
  - Gift Occasion Follow-up (score: 0.95)
  - Anniversary Special (score: 0.90)
  - VIP Event Invitation (score: 0.85)
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
