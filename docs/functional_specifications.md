# Functional Specifications

## Table of Contents

<details open>
<summary>Table of Contents</summary>

 - [1. Overview](#1-overview)
   - [1.1 Document Purpose](#11-document-purpose)
   - [1.2 Context](#12-context)
   - [1.3 Data Source](#13-data-source)
 - [2. Product Goal](#2-product-goal)
   - [2.1 Project Scope](#21-project-scope)
   - [2.2 Constraints](#22-constraints)
   - [2.3 Risks and Mitigations](#23-risks-and-mitigations)
 - [3. Algorithm Goal](#3-algorithm-goal)
 - [4. Product Details](#4-product-details)
   - [4.1 Memory Requirements](#41-memory-requirements)
   - [4.2 Non-Functional Requirements](#42-non-functional-requirements)
   - [4.3 Minimum Viable Product](#43-minimum-viable-product)
   - [4.4 Acceptance Criteria](#44-acceptance-criteria)
   - [4.5 Out of Scope](#45-out-of-scope)
 - [5. Pipeline Features](#5-pipeline-features)
 - [6. Security](#6-security)
   - [6.1 Security Measures](#61-security-measures)
   - [6.2 Error Handling](#62-error-handling)
 - [7. Glossary](#7-glossary)
 
</details>

## Document Evolution
| Author| Équipe Data Science LVMH |
|---------------|------------|
| Created | 01/20/2025 |
| Last modified | 01/20/2025 |
| Document deadline | 02/07/2025 |

## 1. Overview

### 1.1. Document Purpose

Ce document a pour objectif de fournir des instructions claires et détaillées sur le produit et ses objectifs pour les développeurs, les designers et les parties prenantes.

Il doit décrire comment le produit fonctionnera tout en déclarant son comportement prévu et ses fonctionnalités implémentées.

Le document fonctionne comme un contrat entre l'équipe de développement et les parties prenantes, où les exigences et contraintes qui doivent être satisfaites seront définies.

### 1.2. Context

Le projet imaginé par LVMH est de développer une solution logicielle haute performance avec un algorithme<sup><a href="#1">[1]</a></sup> qui transforme automatiquement les transcriptions vocales des Conseillers de Vente en profils clients structurés et actionnables.

LVMH souhaitait encourager l'exploration et l'implémentation d'algorithmes<sup><a href="#1">[1]</a></sup> efficaces adaptés pour gérer des jeux de données<sup><a href="#5">[5]</a></sup> multilingues tout en considérant des contraintes réelles comme la vitesse, la précision et la reproductibilité.

Le pipeline est **déterministe** (sans LLM) et traite les notes dans 5 langues (FR, EN, IT, ES, DE) pour produire des tags structurés, des profils clients segmentés et des recommandations d'actions personnalisées.

### 1.3. Data Source

Pour le processus de test du projet, chaque équipe reçoit un fichier CSV contenant des notes de transcription avec les colonnes suivantes : ***ID, Date, Duration, Language, Length, Transcription***.

Dans ce fichier, chaque ligne représente une interaction entre un Conseiller de Vente et un client, avec une transcription textuelle exprimée en texte libre dans l'une des langues supportées.

Les transcriptions peuvent contenir :
- Des préférences stylistiques ("aime le cuir noir, style classique")
- Des occasions à venir ("anniversaire de mariage en mars")
- Des contraintes alimentaires ("végétarienne, allergie aux noix")
- Des relations familiales ("cadeau pour sa fille")
- Des budgets et comportements d'achat ("budget flexible autour de 5000€")

Le fichier CSV doit être placé dans le dossier **data/raw/** avant l'exécution du pipeline.

## 2. Product Goal

### 2.1. Project Scope

Notre produit sera développé avec les fonctionnalités suivantes :

- Un **pipeline de traitement NLP**<sup><a href="#1">[1]</a></sup> performant et efficace qui reçoit un fichier CSV en entrée<sup><a href="#13">[13]</a></sup> et produit plusieurs fichiers de sortie<sup><a href="#18">[18]</a></sup> :
  - **Tags structurés** : concepts extraits et normalisés (lexique)
  - **Profils clients** : segments avec types de profils et scores de confiance
  - **Actions recommandées** : suggestions personnalisées basées sur des playbooks
  - **Visualisation 3D** : espace d'embedding<sup><a href="#8">[8]</a></sup> interactif

- Un **pipeline en 8 étapes** qui traite séquentiellement les données :
  1. **Ingestion** : Chargement et validation du CSV
  2. **Extraction de candidats** : Extraction de mots-clés avec YAKE, RAKE et TF-IDF
  3. **Construction du lexique** : Regroupement des candidats en concepts via clustering<sup><a href="#9">[9]</a></sup>
  4. **Détection de concepts** : Correspondance des concepts dans les notes
  5. **Construction de vecteurs** : Embeddings<sup><a href="#8">[8]</a></sup> avec SentenceTransformer
  6. **Segmentation clients** : Clustering<sup><a href="#9">[9]</a></sup> KMeans pour créer des profils
  7. **Recommandation d'actions** : Matching avec playbooks YAML
  8. **Projection 3D** : Visualisation interactive avec UMAP<sup><a href="#22">[22]</a></sup> et Plotly

- Un **système de validation** qui vérifie l'intégrité des données fournies. Le code source avec les instructions d'utilisation doit être dans le dépôt GitHub<sup><a href="#7">[7]</a></sup>. Avant d'utiliser le jeu de données<sup><a href="#5">[5]</a></sup>, nous devons également nous assurer que le fichier contient les colonnes requises et que les transcriptions ne sont pas vides.

### 2.2. Constraints

D'abord, le produit doit être implémenté en **Python 3.10+** pour bénéficier d'un écosystème riche en bibliothèques NLP et permettre une intégration facile avec les outils existants.

Ensuite, le pipeline doit être capable de traiter des fichiers avec un grand nombre de notes tout en maintenant des performances acceptables, encourageant une optimisation continue de l'algorithme<sup><a href="#1">[1]</a></sup> et permettant l'exécution de nombreux tests critiques.

Le pipeline doit être **déterministe** : tous les résultats doivent être reproductibles grâce à des seeds fixes (RANDOM_SEED=42, NUMPY_SEED=42).

Le pipeline ne doit **pas utiliser de LLM** (GPT, Claude, etc.) pour des raisons de confidentialité, de coûts et de latence.

Le pipeline doit supporter **5 langues** : Français (FR), Anglais (EN), Italien (IT), Espagnol (ES) et Allemand (DE).

Toutes les sorties doivent être **standardisées en français** pour la cohérence.

Enfin : le pipeline doit gérer gracieusement les erreurs (fichiers manquants, données corrompues, mémoire insuffisante) sans planter.

### 2.3. Risks and Mitigations

| Risks | Mitigations | 
| ------| ----------- |
| Problèmes de performance dus à la taille importante des embeddings<sup><a href="#8">[8]</a></sup> et du clustering<sup><a href="#9">[9]</a></sup>. | Utilisation de batch processing, fallback sur PCA si UMAP échoue, optimisation de la mémoire avec Parquet. | 
| Apparition de vulnérabilités lors du traitement de données sensibles clients. | Traitement local uniquement, pas d'appels API externes, données stockées localement dans des fichiers sécurisés. | 
| Mises à jour constantes qui déclenchent une réduction indirecte de la qualité du code et de la vitesse de l'algorithme<sup><a href="#1">[1]</a></sup>, arrêtant potentiellement tout progrès réel. | Établissement d'un plan de développement approprié pour indiquer les changements, et suivi des tests de l'algorithme<sup><a href="#1">[1]</a></sup> et de sa progression dans le temps. | 
| Manque de connaissances avancées concernant les embeddings<sup><a href="#8">[8]</a></sup> multilingues dans l'équipe. | Recherche approfondie sur SentenceTransformer et les modèles multilingues serait bénéfique. Documentation technique détaillée dans algorithm_reasoning.md. | 
| Erreurs de transcription vocale affectant la qualité des concepts extraits. | Utilisation de plusieurs méthodes d'extraction (YAKE, RAKE, TF-IDF) pour robustesse, filtrage par fréquence minimale. | 

## 3. Algorithm Goal

L'algorithme<sup><a href="#1">[1]</a></sup> transforme **les transcriptions vocales en profils clients structurés**, en effectuant des calculs basés sur la similarité sémantique et les concepts détectés.

D'abord, il charge et valide le fichier CSV fourni, puis extrait des candidats (mots-clés) de chaque transcription en utilisant plusieurs méthodes complémentaires (YAKE, RAKE, TF-IDF).

Ensuite, il regroupe ces candidats en concepts via clustering<sup><a href="#9">[9]</a></sup> hiérarchique sur les embeddings<sup><a href="#8">[8]</a></sup>, créant un lexique normalisé.

Après avoir détecté les concepts dans chaque note, l'algorithme<sup><a href="#1">[1]</a></sup> calcule des embeddings<sup><a href="#8">[8]</a></sup> pour chaque transcription et segmente les clients en groupes similaires.

Enfin, il génère des recommandations d'actions personnalisées en faisant correspondre les profils clients avec des playbooks configurables.

Le pipeline fournit rapidement et efficacement :
- **Les concepts détectés** avec leurs positions dans le texte
- **Les profils clients** avec segments et scores de confiance
- **Les actions recommandées** avec scores de matching
- **La visualisation 3D** interactive de l'espace client

## 4. Product Details

### 4.1. Memory Requirements

Comprendre les types d'entrée<sup><a href="#13">[13]</a></sup> et de sortie<sup><a href="#18">[18]</a></sup> est crucial lors de l'utilisation d'embeddings<sup><a href="#8">[8]</a></sup> pour représenter les notes, car cela aide à déterminer la mémoire minimale requise pour notre produit.

D'abord, pour **100 notes** avec des transcriptions moyennes de 200 mots chacune :

- **Données brutes** : ~20 000 mots × 4 bytes (UTF-8) = **80 KB**

- **Candidats extraits** : ~500 candidats × 50 bytes (moyenne) = **25 KB**

- **Lexique** : ~130 concepts × 200 bytes = **26 KB**

- **Embeddings<sup><a href="#8">[8]</a></sup> de notes** : 100 notes × 384 dimensions × 4 bytes (float32) = **153.6 KB**

- **Vecteurs clients** : 100 clients × 384 dimensions × 4 bytes = **153.6 KB**

- **Matrices de clustering<sup><a href="#9">[9]</a></sup>** : ~10 KB (matrices temporaires)

En tenant compte des structures diverses telles que les DataFrames pandas, les buffers d'entrée<sup><a href="#13">[13]</a></sup>/sortie<sup><a href="#18">[18]</a></sup>, et la gestion des fichiers intermédiaires, nous pouvons estimer une exigence mémoire supplémentaire d'environ **50 MB** pour les bibliothèques Python et les structures temporaires.

Au final, la mémoire minimale requise serait d'environ **50-100 MB** pour 100 notes, avec une croissance linéaire pour des volumes plus importants.

### 4.2. Non-Functional Requirements

Voici les différents critères pour les exigences non fonctionnelles :

#### Functionality
L'algorithme<sup><a href="#1">[1]</a></sup> doit être stable, précis, efficace en espace et résoudre le problème correctement.

#### Scalability
L'algorithme<sup><a href="#1">[1]</a></sup> doit gérer des entrées<sup><a href="#13">[13]</a></sup> grandes et complexes, y compris des fichiers CSV avec jusqu'à **1000 notes** tout en maintenant les performances.

#### Performance
L'algorithme<sup><a href="#1">[1]</a></sup> doit utiliser une mémoire minimale de **100 MB** pour 100 notes et répondre à tous les traitements **en moins de 1 minute** sur n'importe quel type d'ordinateur.

#### Robustness
L'algorithme<sup><a href="#1">[1]</a></sup> doit gérer les entrées<sup><a href="#13">[13]</a></sup> invalides, les cas limites et les erreurs de manière fiable. Les solutions heuristiques ne doivent pas dépasser une marge d'erreur de **10%** par rapport aux résultats attendus.

#### Integrity
L'algorithme<sup><a href="#1">[1]</a></sup> doit fournir un pipeline propre qui supporte une utilisation réelle et produit des fichiers de sortie<sup><a href="#18">[18]</a></sup> dans des formats standards (**CSV**, **Parquet**, **JSON**<sup><a href="#15">[15]</a></sup>).

#### Maintainability 
L'algorithme<sup><a href="#1">[1]</a></sup> doit permettre des mises à jour et modifications basées sur les retours utilisateurs, supportant un développement à long terme.

#### Determinism
L'algorithme<sup><a href="#1">[1]</a></sup> doit produire des résultats reproductibles grâce à des seeds fixes et l'absence d'appels API externes.

### 4.3. Minimum Viable Product

Voici une liste des différentes phases de développement potentielles de notre produit. Chaque phase est mise à jour en fonction de la progression de l'algorithme<sup><a href="#1">[1]</a></sup> selon la spécification non fonctionnelle.

|**Phase** |**Targeted Non-Functional Requirements** | **Algorithm Improvements** | **Version** |
|:------- |:--------- | :--------- |:-------------|
|**Phase 1** |Core Functionality & Scalability | Pipeline<sup><a href="#1">[1]</a></sup> fournissant des sorties<sup><a href="#18">[18]</a></sup> correctes, utilisant des structures complexes de grandes tailles variées.| 0.2 |
|**Phase 2** |Performance| Temps d'exécution amélioré et utilisation mémoire réduite grâce à des techniques d'optimisation avancées. |0.4 (Alpha) |
|**Phase 3** |Integrity | Implémentation complète du pipeline<sup><a href="#1">[1]</a></sup> avec validation des données et gestion d'erreurs robuste. |0.6|
|**Phase 4** |Robustness | Meilleure versatilité et fiabilité, capable de fournir des sorties<sup><a href="#18">[18]</a></sup> dans plusieurs formats (CSV, Parquet, JSON<sup><a href="#15">[15]</a></sup>). | 0.8 (Beta) |
|**Phase 5** | Maintainability | Algorithme<sup><a href="#1">[1]</a></sup> raffiné selon les retours utilisateurs et insights supplémentaires, avec améliorations ou fonctionnalités optionnelles. | 1.0 (Final)|

### 4.4. Acceptance Criteria

Pour déterminer si ce projet IT est réussi, le produit doit satisfaire tous les critères suivants :

- L'algorithme<sup><a href="#1">[1]</a></sup> traite correctement et produit des sorties<sup><a href="#18">[18]</a></sup> valides pour des entrées<sup><a href="#13">[13]</a></sup> jusqu'à 1000 notes en moins de 1 minute.
- L'algorithme<sup><a href="#1">[1]</a></sup> gère les entrées<sup><a href="#13">[13]</a></sup> invalides (par exemple, colonnes manquantes, données corrompues) gracieusement en retournant des messages d'erreur appropriés.
- Les fichiers de sortie<sup><a href="#18">[18]</a></sup> respectent la structure de payload spécifiée pour les formats CSV, Parquet et JSON<sup><a href="#15">[15]</a></sup>.
- La solution atteint au moins **90% de couverture** pour la détection de concepts (notes avec au moins un concept détecté).
- Le produit est déployé avec succès dans un pipeline<sup><a href="#1">[1]</a></sup> qui peut être exécuté avec des commandes efficaces (make, scripts shell, Docker).
- Les résultats sont **reproductibles** : deux exécutions avec les mêmes données produisent les mêmes résultats.

### 4.5. Out of Scope

Voici la liste des fonctionnalités qui seront mises de côté pour ce projet ou qui ont été simplement abandonnées pendant le développement :

- Compatibilité avec d'autres formats de sortie<sup><a href="#18">[18]</a></sup> (XML<sup><a href="#21">[21]</a></sup>, YAML pour les profils).
- Interface utilisateur graphique créée comme une page HTML<sup><a href="#10">[10]</a></sup> où l'utilisateur entre les données et visualise les résultats.
- API REST<sup><a href="#20">[20]</a></sup> pour accéder au pipeline<sup><a href="#1">[1]</a></sup> via HTTP<sup><a href="#11">[11]</a></sup>.
- Traitement en temps réel : le pipeline<sup><a href="#1">[1]</a></sup> fonctionne en mode batch uniquement.
- Support de langues supplémentaires au-delà des 5 langues initiales (FR, EN, IT, ES, DE).
- Intégration directe avec des systèmes CRM externes.
- Génération automatique de rapports PDF ou Excel.

## 5. Pipeline Features

Nous construirons un pipeline<sup><a href="#1">[1]</a></sup> en 8 étapes qui traite séquentiellement les données pour transformer les transcriptions en profils clients actionnables.

Le pipeline<sup><a href="#1">[1]</a></sup> peut être exécuté via :
- **Scripts shell** : `make dev` (local venv) ou `make run` (Docker)
- **Fichiers batch** : `LVMH_Pipeline.bat` (Windows) ou `LVMH_Pipeline.command` (macOS)
- **Ligne de commande** : `python -m src.run_all`

Parmi toutes les fonctionnalités principales possibles, le pipeline<sup><a href="#1">[1]</a></sup> fournira les fonctionnalités suivantes :

#### Stage 1: Ingestion
- **Entrée** : Fichier CSV avec colonnes ID, Date, Duration, Language, Length, Transcription
- **Sortie** : `notes_clean.parquet` (données nettoyées et normalisées)
- **Fonctionnalités** : Validation des colonnes, normalisation des langues, nettoyage du texte

#### Stage 2: Extraction de Candidats
- **Entrée** : `notes_clean.parquet`
- **Sortie** : `candidates.csv` (mots-clés extraits)
- **Fonctionnalités** : YAKE, RAKE-NLTK, TF-IDF, extraction d'entités (budgets, allergies)

#### Stage 3: Construction du Lexique
- **Entrée** : `candidates.csv`
- **Sortie** : `lexicon_v1.csv` et `taxonomy_v1.json`
- **Fonctionnalités** : Clustering<sup><a href="#9">[9]</a></sup> hiérarchique sur embeddings<sup><a href="#8">[8]</a></sup>, assignation de buckets taxonomiques

#### Stage 4: Détection de Concepts
- **Entrée** : `notes_clean.parquet` et `lexicon_v1.csv`
- **Sortie** : `note_concepts.csv` (correspondances concept ↔ note)
- **Fonctionnalités** : Matching d'aliases avec positions (spans) dans le texte

#### Stage 5: Construction de Vecteurs
- **Entrée** : `notes_clean.parquet`
- **Sortie** : `note_vectors.parquet` (embeddings<sup><a href="#8">[8]</a></sup> 384 dimensions)
- **Fonctionnalités** : SentenceTransformer multilingue, agrégation client si plusieurs notes

#### Stage 6: Segmentation Clients
- **Entrée** : `note_vectors.parquet` et `note_concepts.csv`
- **Sortie** : `client_profiles.csv` (segments avec profils et confiance)
- **Fonctionnalités** : Clustering<sup><a href="#9">[9]</a></sup> KMeans, calcul de scores de confiance, assignation de types de profils

#### Stage 7: Recommandation d'Actions
- **Entrée** : `client_profiles.csv`, `note_concepts.csv`, `playbooks.yml`
- **Sortie** : `recommended_actions.csv` (actions recommandées par client)
- **Fonctionnalités** : Matching avec playbooks YAML, scoring par triggers, tri par priorité

#### Stage 8: Projection 3D (Optionnel)
- **Entrée** : `note_vectors.parquet` et `client_profiles.csv`
- **Sortie** : `embedding_space_3d.html` (visualisation interactive)
- **Fonctionnalités** : Réduction dimensionnelle UMAP<sup><a href="#22">[22]</a></sup> (ou PCA fallback), graphique Plotly 3D interactif

De plus, toutes ces fonctionnalités doivent être compatibles avec Python<sup><a href="#3">[3]</a></sup>, efficaces, rapides et robustes. Cela garantit que l'algorithme<sup><a href="#1">[1]</a></sup> fonctionne efficacement dans divers cas d'usage.

Les exemples de structure de sortie<sup><a href="#18">[18]</a></sup> suivants démontrent le format des fichiers générés :

#### Structure de sortie<sup><a href="#18">[18]</a></sup> client_profiles.csv

```csv
client_id,cluster_id,profile_type,top_concepts,confidence
CA_001,2,"Anniversaire | Cadeau | Budget","anniversaire|cadeau|budget",0.85
CA_002,5,"Voyage | Premium | Lifestyle","voyage|premium|lifestyle",0.92
```

#### Structure de sortie<sup><a href="#18">[18]</a></sup> recommended_actions.csv

```csv
client_id,action_id,title,channel,priority,kpi,triggers,rationale
CA_001,ACT_003,"Gift Occasion Follow-up",Email,High,"Response rate, gift purchase","bucket:occasion | keyword:anniversaire","Proactive outreach before known gift occasions"
```

#### Structure de sortie<sup><a href="#18">[18]</a></sup> taxonomy_v1.json

```json
{
  "intent": ["CONCEPT_0001_abc123", "CONCEPT_0002_def456"],
  "occasion": ["CONCEPT_0010_xyz789"],
  "preferences": ["CONCEPT_0020_uvw012"],
  "constraints": ["CONCEPT_0030_rst345"],
  "lifestyle": ["CONCEPT_0040_mno678"],
  "next_action": ["CONCEPT_0050_pqr901"],
  "other": []
}
```

## 6. Security

### 6.1. Security Measures

Pour empêcher quiconque d'utiliser le pipeline<sup><a href="#1">[1]</a></sup> de manière malveillante, comme la fuite de données personnelles ou l'implémentation de virus, nous emploierons plusieurs mesures de sécurité.

- Plus important encore, nous traiterons toutes les données **localement** sans appels API externes, empêchant les attaquants d'intercepter, manipuler ou voler les données transmises ou reçues par le pipeline<sup><a href="#1">[1]</a></sup>.
- Nous nous assurerons de garder nos bibliothèques à jour via `requirements.txt`.
- Nous utiliserons des **seeds fixes** pour garantir la reproductibilité et éviter les comportements non déterministes.
- Nous implémenterons un programme de gestion d'erreurs pour anticiper toute erreur d'entrée<sup><a href="#13">[13]</a></sup> et éviter d'exposer des informations sensibles via les messages d'erreur.
- Les données sensibles ne seront jamais loggées dans la console ou écrites dans des fichiers temporaires non sécurisés.

### 6.2. Error Handling

Voici les cas d'erreur que le programme gérera :

| **Case** | **Error Handling** |
| ----- | ----------- |
| Fichier CSV manquant dans `data/raw/`. | Le programme affiche le message d'erreur : "No CSV files found in data/raw/. Please place your input CSV in data/raw/" |
| Colonnes requises manquantes dans le CSV. | Le programme affiche le message d'erreur : "Missing required columns: [liste]. Expected columns: ID, Date, Duration, Language, Length, Transcription" |
| Langue non supportée dans le CSV. | Le programme affiche un avertissement : "Warning: Unknown language codes: [liste]" et continue le traitement |
| Le modèle SentenceTransformer n'est pas disponible localement. | Le programme tente de télécharger le modèle, ou affiche : "Failed to load SentenceTransformer model. Run 'make setup-models-local'" |
| Mémoire insuffisante pour exécuter le clustering<sup><a href="#9">[9]</a></sup> UMAP. | Le programme bascule automatiquement sur PCA et affiche : "UMAP failed: [erreur], trying PCA" |
| Le fichier de sortie<sup><a href="#18">[18]</a></sup> ne peut pas être écrit (permissions). | Le programme affiche : "Permission denied: [chemin]. Please check file permissions." |
| Transcription vide dans une note. | Le programme affiche un avertissement : "Warning: [N] rows with empty transcription" et continue le traitement |
| Aucun candidat extrait (fichier trop petit ou texte trop court). | Le programme crée des fichiers de sortie<sup><a href="#18">[18]</a></sup> vides avec les colonnes appropriées et affiche : "WARNING: No candidates found. Creating empty lexicon." |

## 7. Glossary

| **Terms** | **Definitions** |
| ----- | ----------- |
| <span id="0">**Adjacency List**</span> | Structure de données utilisée pour représenter un graphe<sup><a href="#8">[8]</a></sup> où chaque nœud du graphe<sup><a href="#8">[8]</a></sup> stocke une liste de ses sommets voisins. |
| <span id="1">**Algorithm**</span> | Ensemble fini de règles ou d'instructions qui spécifient une séquence d'étapes computationnelles pour résoudre un problème spécifique efficacement. |
| <span id="2">**API**</span> | Signifie *Application Programming Interface*, un ensemble de fonctions et procédures permettant la création d'applications qui accèdent aux fonctionnalités ou données d'un système d'exploitation, d'une application ou d'un autre service. |
| <span id="3">**C++**</span> | Langage compilé populaire pour créer des programmes informatiques et largement utilisé dans le développement de jeux. Développé comme extension de C, il partage presque la même syntaxe. |
| <span id="4">**DAG**</span> | Signifie *Directed Acyclic Graph*, un type de graphe<sup><a href="#8">[8]</a></sup> constitué de sommets et d'arêtes dirigées (arcs), où chaque arête pointe d'un sommet à un autre, et aucun cycle n'existe. |
| <span id="5">**Dataset**</span> | Collection de données provenant d'une seule source ou destinée à un seul projet. |
| <span id="6">**Endpoint**</span> | Emplacement spécifique dans une API<sup><a href="#2">[2]</a></sup> qui accepte des requêtes et renvoie des réponses. |
| <span id="7">**Github**</span> | Plateforme cloud qui permet à ses utilisateurs de créer des projets de codage via un dépôt et de travailler ensemble pour stocker et partager du code. En effet, cela nous permet de suivre et gérer nos changements dans le temps. |
| <span id="8">**Graph**</span> | Diagramme qui représente une collection de sommets et d'arêtes qui joignent des paires de sommets. |
| <span id="9">**Heuristics**</span> | Raccourcis mentaux ou approches pragmatiques utilisés pour résoudre des problèmes rapidement et efficacement lorsque le temps, les ressources ou les informations sont limités. Ces méthodes ne garantissent pas une solution optimale ou parfaite mais fournissent des résultats qui sont "assez bons" pour être utiles dans des situations pratiques. |
| <span id="10">**HTML**</span> | Signifie *Hypertext Markup Language*, format basé sur le texte décrivant comment le contenu est structuré dans un fichier, et affichant du texte, des images et d'autres formes de multimédia sur une page web. |
| <span id="11">**HTTP**</span> | Signifie *Hypertext Transfer Protocol*, un ensemble de règles ajustant le transfert de fichiers (textes, images, sons...) sur le Web, et indirectement utilisé lorsque nous nous connectons et ouvrons un navigateur web. |
| <span id="12">**HTTPS**</span> | Signifie *Hyper Text Transfer Protocol Secure*, une extension sécurisée du protocole HTTP<sup><a href="#11">[11]</a></sup>, permettant aux données transférées entre l'utilisateur et le serveur web d'être cryptées et ne peuvent pas être divulguées ou modifiées. |
| <span id="13">**Input**</span> | Fait référence aux données, signaux ou instructions fournis à un système (par exemple, un programme informatique, un appareil ou un processus) pour déclencher des opérations spécifiques ou produire un résultat. |
| <span id="14">**Integer**</span> | Fait référence à un nombre entier (pas fractionnaire) qui peut être positif, négatif ou zéro. |
| <span id="15">**JSON**</span> | Signifie *JavaScript Object Notation*, format de fichier et d'échange de données standard ouvert qui utilise du texte lisible par l'homme pour stocker et transmettre des objets de données contenant des paires nom-valeur et des tableaux (ou d'autres valeurs sérialisables). |
| <span id="16">**Landmark**</span> | Quelque chose utilisé pour marquer la limite d'un terrain. |
| <span id="17">**Localhost**</span> | Environnement serveur local qui peut être utilisé pour tester et exécuter des scripts côté serveur sur un ordinateur. |
| <span id="18">**Output**</span> | Fait référence au résultat, à la réponse ou aux données générées par un système après le traitement de l'entrée<sup><a href="#13">[13]</a></sup>. |
| <span id="19">**Query**</span> | Requête de données que nous pouvons accéder, manipuler, supprimer ou récupérer depuis une base de données. |
| <span id="20">**REST API**</span> | Signifie *Representational State Transfer (REST) API*, elle suit le style architectural REST, définit des règles pour créer des APIs<sup><a href="#2">[2]</a></sup> web légères et flexibles. Une API<sup><a href="#2">[2]</a></sup> RESTful doit se conformer à une architecture client-serveur, une communication sans état, des données en cache, une interface uniforme permettant la manipulation de ressources et la navigation hypermédia, une conception de système en couches, et optionnellement, du code à la demande pour une fonctionnalité client étendue. |
| <span id="21">**TLS**</span> | Signifie *Transport layer Security*, système de cryptage de données entre clients et serveurs qui protège les informations sensibles comme les clés API<sup><a href="#2">[2]</a></sup> et les jetons d'accès contre l'interception. |
| <span id="22">**XML**</span> | Signifie *Extensible Markup Language*, langage de balisage et format de fichier pour stocker, transmettre et reconstruire des données. |
| <span id="23">**Embedding**</span> | Représentation vectorielle dense d'un texte dans un espace de grande dimension (typiquement 384 dimensions pour SentenceTransformer), où des textes sémantiquement similaires ont des vecteurs proches. |
| <span id="24">**Clustering**</span> | Technique d'apprentissage non supervisé qui regroupe des données similaires en clusters basés sur leur similarité (distance cosinus, euclidienne, etc.). |
| <span id="25">**Pipeline**</span> | Série d'étapes de traitement séquentielles où la sortie<sup><a href="#18">[18]</a></sup> d'une étape devient l'entrée<sup><a href="#13">[13]</a></sup> de l'étape suivante. |
| <span id="26">**YAKE**</span> | Yet Another Keyword Extractor, algorithme<sup><a href="#1">[1]</a></sup> d'extraction de mots-clés sans supervision qui identifie les termes importants dans un texte. |
| <span id="27">**RAKE**</span> | Rapid Automatic Keyword Extraction, méthode d'extraction de phrases-clés basée sur la fréquence des mots et leur co-occurrence. |
| <span id="28">**TF-IDF**</span> | Term Frequency-Inverse Document Frequency, mesure statistique qui évalue l'importance d'un terme dans un document par rapport à une collection de documents. |
| <span id="29">**UMAP**</span> | Uniform Manifold Approximation and Projection, technique de réduction dimensionnelle non-linéaire qui préserve la structure locale des données. |
| <span id="30">**KMeans**</span> | Algorithme<sup><a href="#1">[1]</a></sup> de clustering<sup><a href="#24">[24]</a></sup> qui partitionne les données en k clusters en minimisant la variance intra-cluster. |
| <span id="31">**SentenceTransformer**</span> | Modèle de deep learning qui génère des embeddings<sup><a href="#23">[23]</a></sup> de phrases en transformant le texte en vecteurs numériques denses. |
| <span id="32">**Playbook**</span> | Fichier YAML contenant des règles et déclencheurs pour recommander des actions marketing ou commerciales basées sur les profils clients. |
