# LVMH Voice-to-Tag — Vector Profiles

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/NLP-Deterministic-green.svg" alt="NLP">
  <img src="https://img.shields.io/badge/Languages-12+-orange.svg" alt="Languages">
  <img src="https://img.shields.io/badge/LLM-None%20Required-red.svg" alt="No LLM">
  <img src="https://img.shields.io/badge/Vocabulary-384%20concepts-purple.svg" alt="Vocabulary">
</p>

Pipeline **déterministe (sans LLM)** et multilingue qui transforme les transcriptions des Conseillers de Vente en profils clients actionnables et recommandations personnalisées.

**✨ Nouveau:** 
- Support de n'importe quel fichier CSV
- Système d'entraînement de vocabulaire
- **Anonymisation RGPD/GDPR automatique** 🔒

---

## 🎯 Objectifs

Transformer automatiquement les notes vocales des conseillers en :
- **Tags structurés** (préférences, occasions, contraintes)
- **Profils clients** segmentés par similarité
- **Actions recommandées** personnalisées
- **Visualisation 3D** interactive de l'espace client

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

Le pipeline utilise un vocabulaire entraînable de **384 concepts** avec support multilingue (12+ langues).

### Statistiques Actuelles

| Bucket | Concepts | Exemples |
|--------|----------|----------|
| **preferences** | 167 | marques, matériaux, styles |
| **intent** | 71 | émotions, intentions d'achat |
| **lifestyle** | 71 | famille, personnalité, indicateurs VIP |
| **occasion** | 36 | fêtes, événements, étapes de vie |
| **constraints** | 20 | budget, délais, canaux |
| **next_action** | 19 | rendez-vous, réparation, livraison |

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
│   ├── run_all.py              # Orchestrateur principal
│   ├── 📂 shared/              # Config & utilitaires
│   │   ├── config.py           # Configuration centrale
│   │   ├── utils.py            # Fonctions helper
│   │   ├── knowledge_graph.py  # Construction du graphe
│   │   └── generate_dashboard.py # Génération dashboard
│   ├── 📂 ingest/              # Étape 1: Ingestion CSV
│   ├── 📂 extract/             # Étapes 2 & 4: Extraction concepts
│   ├── 📂 lexicon/             # Étape 3: Construction lexique
│   ├── 📂 embeddings/          # Étapes 5 & 8: Vecteurs & UMAP 3D
│   ├── 📂 profiling/           # Étape 6: Segmentation clients
│   └── 📂 actions/             # Étape 7: Recommandations
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
| `notes_clean.parquet` | Notes nettoyées et normalisées |
| `candidates.csv` | Candidats extraits (mots-clés, entités) |

### Taxonomie

| Fichier | Description |
|---------|-------------|
| `vocabulary.json` | Vocabulaire entraîné (384 concepts, multilingue) |
| `lexicon_v1.json` | Lexique synchronisé avec alias et fréquences |
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

Fichier `src/shared/config.py`:

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
| Temps d'exécution | ~18 secondes (100 notes) |
| Couverture concepts | 100% des notes |
| **Vocabulaire entraîné** | **384 concepts** |
| Langues supportées | 12+ |
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
