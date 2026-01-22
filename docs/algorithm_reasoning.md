# LVMH Voice-to-Tag: Raisonnement Algorithmique

## Document de Conception Technique

**Version:** 1.0  
**Date:** Janvier 2026  
**Auteur:** Équipe Data Science LVMH

---

## Table des Matières

1. [Contexte et Contraintes](#1-contexte-et-contraintes)
2. [Philosophie de Conception](#2-philosophie-de-conception)
3. [Choix Architecturaux](#3-choix-architecturaux)
4. [Raisonnement par Étape](#4-raisonnement-par-étape)
5. [Compromis et Limitations](#5-compromis-et-limitations)
6. [Alternatives Considérées](#6-alternatives-considérées)
7. [Évolutions Futures](#7-évolutions-futures)

---

## 1. Contexte et Contraintes

### 1.1 Problématique Métier

Les Conseillers de Vente LVMH enregistrent quotidiennement des notes vocales sur leurs interactions clients. Ces notes contiennent des informations précieuses mais non structurées :

- Préférences stylistiques ("aime le cuir noir, style classique")
- Occasions à venir ("anniversaire de mariage en mars")
- Contraintes alimentaires ("végétarienne, allergie aux noix")
- Relations familiales ("cadeau pour sa fille qui termine ses études")
- Budget et comportement d'achat ("budget flexible autour de 5000€")

**Objectif:** Transformer ce texte libre en profils clients structurés et actionnables.

### 1.2 Contraintes Techniques Imposées

| Contrainte | Raison | Impact |
|------------|--------|--------|
| **Pas de LLM** (GPT, Claude, etc.) | Confidentialité des données clients, coûts API, latence | Utiliser des méthodes NLP classiques |
| **Déterminisme** | Reproductibilité des résultats, auditabilité | Seeds fixes, pas d'aléatoire |
| **Multilingue** | Boutiques dans 5 pays (FR, EN, IT, ES, DE) | Modèle multilingue obligatoire |
| **Temps réel viable** | Intégration future dans CRM | < 1 minute pour 100 notes |
| **Explicabilité** | Comprendre pourquoi un profil est assigné | Concepts et scores traçables |

### 1.3 Caractéristiques des Données

- **Volume:** ~100-1000 notes par batch
- **Longueur:** 100-500 mots par note
- **Langues:** Distribution FR (30%), EN (25%), IT (15%), ES (15%), DE (15%)
- **Structure:** Texte libre, pas de format imposé
- **Qualité:** Erreurs de transcription vocale possibles

---

## 2. Philosophie de Conception

### 2.1 Principe Fondamental: "Embeddings First"

Notre approche repose sur l'hypothèse que **la similarité sémantique capture l'essentiel de l'information client**.

```
Intuition:
  "Mme Dupont cherche un cadeau pour son mari"
  ≈ "Mrs. Smith is looking for a gift for her husband"
  ≈ "Signora Rossi cerca un regalo per suo marito"

→ Ces trois phrases ont des embeddings très proches
→ Elles doivent produire des profils similaires
```

**Pourquoi cette approche?**

1. **Robustesse linguistique:** Un modèle multilingue bien entraîné capture les équivalences sémantiques automatiquement
2. **Généralisation:** Pas besoin de règles explicites pour chaque langue
3. **Évolutivité:** Fonctionne avec de nouvelles langues sans modification

### 2.2 Principe de Séparation des Préoccupations

Le pipeline est divisé en 8 étapes indépendantes car :

1. **Débogage facilité:** Chaque étape peut être testée isolément
2. **Flexibilité:** On peut remplacer une étape sans affecter les autres
3. **Paralélisation future:** Certaines étapes pourront tourner en parallèle
4. **Traçabilité:** Chaque fichier intermédiaire est inspectable

```
┌─────────┐    ┌─────────┐    ┌─────────┐
│ Étape N │───▶│ Fichier │───▶│Étape N+1│
└─────────┘    │  .csv   │    └─────────┘
               │ .parquet│
               └─────────┘
                    ↑
            Inspectable humainement
```

### 2.3 Principe de Standardisation Française

**Décision:** Toutes les sorties sont en français.

**Raisonnement:**
- Le siège LVMH est en France
- Les équipes CRM centrales travaillent en français
- Un vocabulaire unifié facilite les rapports et dashboards
- Évite la confusion des synonymes multilingues

```
Entrée: "gift", "regalo", "geschenk", "cadeau"
       ↓ Standardisation
Sortie: "cadeau" (toujours)
```

---

## 3. Choix Architecturaux

### 3.1 Pourquoi SentenceTransformers et pas Word2Vec/FastText?

| Critère | Word2Vec/FastText | SentenceTransformers |
|---------|-------------------|---------------------|
| Granularité | Mot par mot | Phrase entière |
| Contexte | Aucun | Oui (attention) |
| Multilingue | Modèles séparés | Un seul modèle |
| "King - Man + Woman = Queen" | Approximatif | Excellent |
| Phrases longues | Moyenne des mots (perte d'info) | Embedding natif |

**Exemple concret:**

```python
# Word2Vec: moyenne des mots
"cadeau pour mari" = (embed("cadeau") + embed("pour") + embed("mari")) / 3
# Problème: "pour" dilue le sens

# SentenceTransformer: embedding de la phrase
"cadeau pour mari" = model.encode("cadeau pour mari")
# Le modèle comprend la relation "cadeau → destinataire: mari"
```

### 3.2 Pourquoi `paraphrase-multilingual-MiniLM-L12-v2`?

Ce modèle spécifique a été choisi pour :

| Propriété | Valeur | Importance |
|-----------|--------|------------|
| Langues supportées | 50+ | ✅ Couvre nos 5 langues |
| Dimension | 384 | ✅ Bon compromis (pas 768 ou 1024) |
| Taille | 471 MB | ✅ Utilisable localement |
| Tâche d'entraînement | Paraphrase | ✅ Idéal pour similarité |
| Performance | Top 10 MTEB | ✅ État de l'art accessible |

**Alternatives rejetées:**
- `all-MiniLM-L6-v2`: Anglais seulement
- `multilingual-e5-large`: 2.2 GB, trop lourd
- `LaBSE`: Moins performant sur les phrases courtes

### 3.3 Pourquoi Agglomerative Clustering et pas DBSCAN/HDBSCAN?

Pour regrouper les candidats en concepts (Étape 3):

| Algorithme | Avantages | Inconvénients | Verdict |
|------------|-----------|---------------|---------|
| **KMeans** | Rapide | Nombre de clusters fixe a priori | ❌ |
| **DBSCAN** | Détecte le bruit | Sensible à epsilon, formes sphériques | ❌ |
| **HDBSCAN** | Hiérarchique, robuste | Implémentation complexe | ⚠️ |
| **Agglomerative** | Seuil de distance explicite, hiérarchique | Plus lent sur gros corpus | ✅ |

**Raisonnement:**

1. **Seuil explicite (0.35):** On veut contrôler précisément quand deux candidats sont "assez similaires" pour être le même concept
2. **Hiérarchique:** Permet d'inspecter la dendrogramme si besoin
3. **Linkage "average":** Plus robuste que "single" (chaînage) ou "complete" (diamètre)

```python
# Seuil = 0.35 signifie:
# cos_distance("cadeau", "gift") < 0.35 → même concept ✅
# cos_distance("cadeau", "voyage") > 0.35 → concepts différents ✅
```

### 3.4 Pourquoi KMeans pour la segmentation clients (et pas Agglomerative)?

Pour regrouper les clients (Étape 6), on utilise KMeans:

| Critère | Agglomerative | KMeans |
|---------|---------------|--------|
| Scalabilité | O(n²) mémoire | O(n) mémoire |
| Nombre de clusters | Déterminé par seuil | Fixé a priori |
| Interprétabilité | Dendrogramme | Centroïdes |
| Cas d'usage | Petit corpus, exploration | Production, grand corpus |

**Raisonnement:**

1. **Nombre de segments connu:** LVMH veut ~7 segments marketing
2. **Centroïdes interprétables:** On peut analyser "le client moyen" de chaque segment
3. **Performance:** KMeans scale mieux si on passe à 10k clients

### 3.5 Pourquoi UMAP et pas t-SNE/PCA pour la visualisation?

| Méthode | Préservation | Vitesse | Déterminisme |
|---------|--------------|---------|--------------|
| **PCA** | Variance globale | ⚡⚡⚡ | ✅ |
| **t-SNE** | Structure locale | ⚡ | ❌ (stochastique) |
| **UMAP** | Locale + globale | ⚡⚡ | ✅ (avec seed) |

**Raisonnement:**

1. **PCA écrase les clusters:** En haute dimension, les clusters sont souvent dans des sous-espaces que PCA ne capture pas bien
2. **t-SNE est lent et non déterministe:** Chaque run donne un résultat différent
3. **UMAP est le meilleur compromis:** Préserve la topologie locale (clusters visibles) tout en étant déterministe

```
384D → UMAP → 3D
       ↓
Les clients similaires restent proches
Les clusters sont visuellement séparés
```

---

## 4. Raisonnement par Étape

### 4.1 Étape 1: Ingestion — Pourquoi Parquet?

**Question:** Pourquoi convertir le CSV en Parquet?

**Réponse:**

| Format | Taille | Lecture | Types | Compression |
|--------|--------|---------|-------|-------------|
| CSV | 100% | Lente (parsing) | Inférés | Non |
| Parquet | ~30% | Rapide (binaire) | Préservés | Oui (snappy) |

```python
# CSV: on reparse les types à chaque lecture
df = pd.read_csv("notes.csv")  # "42" → str ou int?

# Parquet: types stockés dans le schéma
df = pd.read_parquet("notes.parquet")  # Types garantis
```

**Bonus:** Parquet permet la lecture partielle de colonnes (utile si on ajoute des métadonnées).

### 4.2 Étape 2: Extraction — Pourquoi 3 extracteurs?

**Question:** Pourquoi combiner YAKE + RAKE + TF-IDF au lieu d'en utiliser un seul?

**Réponse:** Chaque extracteur a des forces différentes:

| Extracteur | Force | Faiblesse | Exemple capturé |
|------------|-------|-----------|-----------------|
| **YAKE** | N-grams statistiques | Mots composés rares | "cuir noir" |
| **RAKE** | Phrases-clés longues | Trop de candidats | "cadeau anniversaire mari" |
| **TF-IDF** | Termes discriminants | Ignore la position | "végétarien" (rare = important) |

```
Texte: "Mme Dupont, cliente VIP, cherche cadeau anniversaire 
        pour son mari. Style classique, cuir noir."

YAKE:  ["cliente vip", "cuir noir", "style classique"]
RAKE:  ["cadeau anniversaire mari", "style classique cuir noir"]
TF-IDF: ["vip", "anniversaire", "classique", "cuir"]

Union → Plus de couverture, moins d'oublis
```

**Filtre de fréquence (≥2):** Un candidat apparaissant dans une seule note est probablement du bruit ou trop spécifique.

### 4.3 Étape 3: Lexique — Pourquoi un seuil de 0.35?

**Question:** Comment a-t-on déterminé le seuil de clustering à 0.35?

**Réponse:** Par expérimentation sur des paires connues:

```python
# Paires qui DOIVENT être dans le même concept:
cos_dist("cadeau", "gift") = 0.18     # ✅ < 0.35
cos_dist("anniversaire", "birthday") = 0.22  # ✅ < 0.35
cos_dist("mari", "husband") = 0.15    # ✅ < 0.35

# Paires qui NE DOIVENT PAS être ensemble:
cos_dist("cadeau", "voyage") = 0.52   # ✅ > 0.35
cos_dist("anniversaire", "allergie") = 0.61  # ✅ > 0.35
cos_dist("mari", "fille") = 0.41      # ✅ > 0.35

# Cas limites (validation manuelle nécessaire):
cos_dist("élégant", "raffiné") = 0.32  # ✅ Même cluster (style)
cos_dist("moderne", "contemporain") = 0.28  # ✅ Même cluster
```

**Compromis:**
- Seuil trop bas (0.20): Clusters trop petits, synonymes séparés
- Seuil trop haut (0.50): Clusters trop gros, concepts mélangés
- **0.35:** Équilibre entre précision et rappel

### 4.4 Étape 4: Détection — Pourquoi matcher par alias et pas par embedding?

**Question:** Pourquoi utiliser du regex matching au lieu de la similarité embedding?

**Réponse:**

| Approche | Avantages | Inconvénients |
|----------|-----------|---------------|
| **Regex** | Rapide, positions exactes, explicable | Manque les paraphrases |
| **Embedding** | Capture les paraphrases | Lent, pas de positions |

**Raisonnement:**

1. **Besoin des positions (spans):** Pour le highlighting dans l'UI future
2. **Explicabilité:** "Ce concept a été détecté car 'regalo' apparaît à la position 45"
3. **Performance:** Regex sur 100 notes = ~1s, embedding search = ~10s

```python
# Regex matching (notre choix):
"Cherche regalo pour mari" 
  → match "regalo" at position 8-14
  → concept_id: CONCEPT_001 (cadeau)
  → evidence: {"start": 8, "end": 14}  # Traçable!

# Embedding matching (alternative):
embed("Cherche regalo pour mari") ≈ embed("cadeau")
  → concept_id: CONCEPT_001
  → evidence: ???  # Pas de position exacte
```

**Compromis accepté:** On manque les paraphrases non présentes dans les alias. Solution future: enrichir les alias avec des paraphrases générées.

### 4.5 Étape 5: Vecteurs — Pourquoi embedder le texte complet?

**Question:** Pourquoi ne pas utiliser la moyenne des embeddings de concepts détectés?

**Réponse:**

```python
# Option A: Moyenne des concepts
concepts = ["cadeau", "anniversaire", "mari"]
client_vector = mean([embed(c) for c in concepts])
# Problème: Perd les nuances, tous les "cadeau anniversaire" sont identiques

# Option B: Embedding du texte complet (notre choix)
text = "Mme Dupont, avocate d'affaires, cherche cadeau anniversaire mari..."
client_vector = embed(text)
# Avantage: Capture le contexte ("avocate d'affaires" → professionnel haut de gamme)
```

**Le texte complet contient des signaux implicites:**
- Ton formel vs informel
- Niveau de détail (client engagé vs pressé)
- Mentions indirectes ("club de golf de Paris" → lifestyle haut de gamme)

### 4.6 Étape 6: Segmentation — Pourquoi k=7 clusters?

**Question:** Comment déterminer le nombre optimal de segments?

**Réponse:** Combinaison de critères métier et statistiques:

**Critères métier:**
- Marketing LVMH travaille avec 5-10 segments historiquement
- Trop peu (3): Pas assez granulaire pour personnaliser
- Trop (15): Impossible à opérationnaliser

**Critères statistiques:**
```python
# Méthode du coude (Elbow method)
for k in range(3, 12):
    kmeans = KMeans(n_clusters=k)
    inertia[k] = kmeans.inertia_  # Somme des distances²

# Plot: on cherche le "coude" où l'amélioration ralentit
# Pour nos données: k=7 est le coude
```

**Score Silhouette:**
```python
silhouette_score(X, labels)  # Entre -1 et 1
# Notre score: 0.048 (faible mais acceptable pour des données textuelles)
# Interprétation: Les clusters existent mais se chevauchent un peu
```

**Décision finale:** k=7 car c'est le compromis entre:
- Granularité suffisante pour la personnalisation
- Taille de cluster suffisante (>10 clients par segment)
- Interprétabilité (on peut nommer chaque segment)

### 4.7 Étape 7: Actions — Pourquoi des playbooks YAML?

**Question:** Pourquoi des règles manuelles plutôt qu'un modèle de recommandation?

**Réponse:**

| Approche | Avantages | Inconvénients |
|----------|-----------|---------------|
| **ML (collaborative filtering)** | Personnalisé, apprend | Besoin de feedback, cold start |
| **Règles YAML** | Explicable, contrôlable, rapide à ajuster | Moins personnalisé |

**Raisonnement:**

1. **Pas de données de feedback (encore):** On ne sait pas quelles actions ont fonctionné
2. **Explicabilité réglementaire:** "Cette action a été recommandée car le client a mentionné 'allergie'" est auditable
3. **Contrôle métier:** Le marketing peut ajuster les règles sans coder

```yaml
# Le marketing peut modifier directement:
- action_id: ACT_003
  title: "Gift Occasion Follow-up"
  triggers:
    keywords: ["birthday", "anniversary", "gift"]
    min_confidence: 0.3  # ← Ajustable sans code
```

**Évolution prévue:** Quand on aura du feedback (clics, conversions), on pourra apprendre les poids des triggers automatiquement.

### 4.8 Étape 8: Visualisation — Pourquoi masquer les valeurs des axes?

**Question:** Les axes UMAP n'ont pas de valeurs. Pourquoi?

**Réponse:**

Les dimensions UMAP **n'ont pas de signification intrinsèque**. Ce sont des projections optimisées pour préserver les distances, pas pour représenter des variables.

```
Dimension 1 UMAP ≠ "niveau de luxe"
Dimension 2 UMAP ≠ "âge du client"
```

**Ce qui est significatif:**
- **Distance entre points:** Clients proches = profils similaires
- **Densité:** Zone dense = segment cohérent
- **Outliers:** Points isolés = clients atypiques

**Labels sémantiques choisis:**
```
X: "← Classique | Moderne →"
Y: "← Quotidien | Événements →"
Z: "← Budget | Premium →"
```

Ces labels sont des **interprétations approximatives** basées sur l'observation des clusters, pas des mesures exactes.

---

## 5. Compromis et Limitations

### 5.1 Compromis Acceptés

| Compromis | Ce qu'on gagne | Ce qu'on perd |
|-----------|----------------|---------------|
| Pas de LLM | Confidentialité, coût, vitesse | Compréhension profonde du contexte |
| Regex matching | Vitesse, positions | Paraphrases non détectées |
| k=7 fixe | Stabilité | Adaptation automatique |
| Seuil 0.35 fixe | Reproductibilité | Optimisation par langue |
| Français uniquement en sortie | Cohérence | Perte de nuances |

### 5.2 Limitations Connues

#### A. Sensibilité aux erreurs de transcription
```
Transcription: "cherche un cardeau pour son mary"
                        ↑             ↑
                    "cadeau"        "mari"
# Regex ne matche pas les fautes d'orthographe
```
**Mitigation future:** Ajouter une étape de correction orthographique (symspell).

#### B. Manque de compréhension contextuelle
```
"Son mari n'aime PAS le cuir"
→ Le pipeline détecte "mari" et "cuir" comme positifs
→ Il manque la négation
```
**Mitigation future:** Analyse de sentiment/polarité sur les spans détectés.

#### C. Nouveaux concepts non détectés
```
Note: "Elle adore le padel" (nouveau sport tendance)
→ "padel" n'est pas dans le lexique initial
→ Concept non détecté
```
**Mitigation future:** Relancer l'étape 2-3 périodiquement pour enrichir le lexique.

### 5.3 Hypothèses Non Validées

| Hypothèse | Validation nécessaire |
|-----------|----------------------|
| Les clusters UMAP correspondent à des segments business | Entretiens avec le marketing |
| Le seuil 0.35 est optimal pour toutes les langues | Test par langue |
| 7 segments suffisent | A/B test sur les recommandations |
| Les playbooks actuels sont pertinents | Mesure du taux de conversion |

---

## 6. Alternatives Considérées

### 6.1 Alternative: Fine-tuning d'un modèle de classification

```python
# Approche rejetée:
model = BertForSequenceClassification(num_labels=7)
model.fit(notes, manual_labels)  # Besoin de labels manuels!
```

**Pourquoi rejetée:**
- Besoin de 1000+ notes labellisées manuellement
- Rigide: ajouter un segment = ré-entraîner
- Le clustering est plus exploratoire

### 6.2 Alternative: Named Entity Recognition (NER)

```python
# Approche rejetée:
ner_model = spacy.load("fr_core_news_lg")
entities = ner_model("Mme Dupont cherche un cadeau")
# → Détecte "Mme Dupont" comme PERSON, mais pas "cadeau" comme GIFT
```

**Pourquoi rejetée:**
- Les entités métier (cadeau, voyage, allergie) ne sont pas des NER standard
- Faudrait entraîner un NER custom = besoin de données annotées

### 6.3 Alternative: Topic Modeling (LDA)

```python
# Approche rejetée:
lda = LatentDirichletAllocation(n_components=10)
topics = lda.fit_transform(tfidf_matrix)
```

**Pourquoi rejetée:**
- LDA suppose un "bag of words" = perd l'ordre des mots
- Les topics sont souvent difficiles à interpréter
- Pas multilingue nativement

### 6.4 Alternative: Graph-based (TextRank)

```python
# Approche considérée mais non retenue:
import networkx as nx
# Construire un graphe de co-occurrence
# Appliquer PageRank pour extraire les mots importants
```

**Pourquoi non retenue:**
- Plus complexe à implémenter
- YAKE utilise déjà des principes similaires
- Pas d'avantage clair sur notre corpus

---

## 7. Évolutions Futures

### 7.1 Court terme (< 3 mois)

| Amélioration | Impact | Effort |
|--------------|--------|--------|
| Correction orthographique (symspell) | +10% de détection | Faible |
| Enrichissement des alias (synonymes) | +15% de couverture | Moyen |
| Dashboard interactif (Streamlit) | Meilleure adoption | Moyen |

### 7.2 Moyen terme (3-6 mois)

| Amélioration | Impact | Effort |
|--------------|--------|--------|
| Feedback loop sur les actions | Apprentissage continu | Élevé |
| Détection de négation | +5% de précision | Moyen |
| Clustering dynamique (k auto) | Adaptation aux données | Moyen |

### 7.3 Long terme (6-12 mois)

| Amélioration | Impact | Effort |
|--------------|--------|--------|
| LLM local (Mistral 7B) pour enrichissement | Compréhension profonde | Élevé |
| Intégration temps réel (Kafka) | Profils live | Très élevé |
| Modèle de recommandation appris | Personnalisation poussée | Très élevé |

---

## Annexe A: Formules Mathématiques

### Cosine Distance
$$
d_{cos}(A, B) = 1 - \frac{A \cdot B}{\|A\| \|B\|}
$$

Où $A \cdot B$ est le produit scalaire et $\|A\|$ la norme L2.

### TF-IDF
$$
\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)
$$

$$
\text{TF}(t, d) = \frac{f_{t,d}}{\sum_{t' \in d} f_{t',d}}
$$

$$
\text{IDF}(t, D) = \log \frac{|D|}{1 + |\{d \in D : t \in d\}|}
$$

### Score YAKE
$$
S(kw) = \frac{T_{Rel} \cdot T_{Pos}}{T_{Case} + \frac{T_{Freq}}{T_{Rel}} + \frac{T_{DiffSent}}{T_{Freq}}}
$$

Plus le score est **bas**, plus le mot-clé est pertinent.

### Silhouette Score
$$
s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}
$$

Où:
- $a(i)$ = distance moyenne aux points du même cluster
- $b(i)$ = distance moyenne aux points du cluster le plus proche

---

## Annexe B: Références

1. **SentenceTransformers:** Reimers & Gurevych (2019). "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"

2. **YAKE:** Campos et al. (2020). "YAKE! Keyword extraction from single documents using multiple local features"

3. **UMAP:** McInnes et al. (2018). "UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction"

4. **Agglomerative Clustering:** Müllner (2011). "Modern hierarchical, agglomerative clustering algorithms"

---

*Document généré pour le projet LVMH Voice-to-Tag — Version 1.0*
