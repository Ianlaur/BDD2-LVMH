"""
Lexicon building stage: Group candidates into concepts using embeddings.

This module:
- Embeds candidate phrases using SentenceTransformer
- Clusters candidates using Agglomerative Clustering with cosine distance
- Builds lexicon with concept labels and aliases
- Assigns taxonomy buckets using keyword rules

Output:
- taxonomy/lexicon_v1.csv
- taxonomy/taxonomy_v1.json
"""
import sys
import json
import hashlib
from collections import defaultdict
from typing import List, Dict, Set, Optional, Tuple
import pandas as pd
import numpy as np

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances

from server.shared.config import (
    DATA_PROCESSED, TAXONOMY_DIR, 
    CONCEPT_CLUSTER_DISTANCE_THRESHOLD, MIN_ALIASES_PER_CONCEPT,
    TAXONOMY_RULES, SKLEARN_RANDOM_STATE
)
from server.shared.utils import log_stage, set_all_seeds, slugify
from server.shared.model_cache import get_sentence_transformer

# ---------------------------------------------------------------------------
# Known synonyms that the embedding model may or may not group correctly.
# Each set contains words that are legitimate aliases of each other.
# Used during post-clustering pruning to avoid ejecting real synonyms.
# ---------------------------------------------------------------------------
_KNOWN_SYNONYM_SETS: list[set[str]] = [
    # Colors (FR/EN/DE/ES/IT/PT equivalents)
    {"noir", "black", "schwarz", "negro", "nero", "preto"},
    {"blanc", "white", "weiß", "weiss", "blanco", "bianco", "branco"},
    {"rouge", "red", "rot", "rojo", "rosso", "vermelho"},
    {"bleu", "blue", "blau", "azul", "blu"},
    {"vert", "green", "grün", "gruen", "verde"},
    {"rose", "pink", "rosa"},
    {"marron", "brown", "braun", "marrón", "marrone", "castanho"},
    {"gris", "grey", "gray", "grau"},
    {"beige"},
    {"taupe"},
    {"bordeaux", "burgundy"},
    {"navy", "marine"},
    {"crème", "cream", "creme"},
    {"ivoire", "ivory"},
    {"doré", "gold", "golden", "or"},
    {"argenté", "silver", "argent"},
    # Materials
    {"cuir", "leather", "leder", "cuero", "pelle", "couro"},
    {"soie", "silk", "seide", "seda", "seta"},
    {"cachemire", "cashmere", "kaschmir"},
    {"lin", "linen", "leinen", "lino"},
    {"coton", "cotton", "baumwolle", "algodón", "cotone"},
    {"laine", "wool", "wolle", "lana"},
    {"voile", "veil", "chiffon"},
    # Diet / lifestyle
    {"végan", "vegan", "vegano"},
    {"végétarien", "vegetarian", "vegetarisch", "vegetariano"},
    {"keto", "kéto", "cétogène"},
    {"pilates"},
    {"yoga"},
    # Accessories
    {"sac", "bag", "tasche", "bolso", "borsa"},
    {"montre", "watch", "uhr", "reloj", "orologio"},
    {"parfum", "perfume", "fragrance", "duft"},
    {"écharpe", "scarf", "foulard"},
    # Other
    {"mari", "husband", "ehemann", "marido", "marito"},
    {"femme", "wife", "woman"},
    {"anniversaire", "birthday", "anniversary", "geburtstag", "cumpleaños"},
    {"mariage", "wedding", "hochzeit", "boda", "matrimonio"},
    {"voyage", "travel", "reise", "viaje", "viaggio"},
    {"cadeau", "gift", "geschenk", "regalo"},
]

# Build a fast lookup: word → frozenset index
_SYNONYM_INDEX: dict[str, int] = {}
for _i, _syn_set in enumerate(_KNOWN_SYNONYM_SETS):
    for _w in _syn_set:
        _SYNONYM_INDEX[_w.lower()] = _i


def _are_known_synonyms(a: str, b: str) -> bool:
    """Return True if a and b belong to the same synonym set."""
    idx_a = _SYNONYM_INDEX.get(a.lower())
    idx_b = _SYNONYM_INDEX.get(b.lower())
    if idx_a is not None and idx_b is not None and idx_a == idx_b:
        return True
    return False


def _share_stem(a: str, b: str, min_overlap: int = 4) -> bool:
    """Return True if a and b share a character substring of length >= min_overlap."""
    a_low, b_low = a.lower(), b.lower()
    if a_low in b_low or b_low in a_low:
        return True
    shorter = a_low if len(a_low) <= len(b_low) else b_low
    longer = b_low if shorter == a_low else a_low
    for k in range(min_overlap, len(shorter) + 1):
        for start in range(len(shorter) - k + 1):
            if shorter[start:start + k] in longer:
                return True
    return False


def prune_cluster(
    label: str,
    aliases: list[str],
    embeddings_map: dict[str, np.ndarray],
    distance_matrix: np.ndarray | None = None,
) -> tuple[str, list[str], list[str]]:
    """
    Validate every alias in a cluster against the label.

    A short alias (≤2 words) is ejected when **both** the label and alias
    are short and:
    1. They are NOT known synonyms, AND
    2. They share NO word overlap AND NO 4-char stem overlap.

    Longer aliases (≥3 words) almost always carry enough semantic context
    to be trustworthy, so they are kept unconditionally.

    Returns (label, kept_aliases, ejected_aliases).
    """
    kept: list[str] = []
    ejected: list[str] = []

    label_words = set(label.lower().split())
    label_word_count = len(label_words)

    for alias in aliases:
        alias_words = set(alias.lower().split())
        alias_word_count = len(alias_words)

        # Long aliases (≥3 words) are almost always fine — keep them
        if alias_word_count >= 3:
            kept.append(alias)
            continue

        # Long label with short alias — usually fine (e.g. "luxury writing set" + "luxury")
        if label_word_count >= 3:
            kept.append(alias)
            continue

        # --- Both label and alias are short (1-2 words): high-risk zone ---

        # Check known synonyms
        if _are_known_synonyms(label.lower(), alias.lower()):
            kept.append(alias)
            continue

        # Check for any word in common (e.g. "sac" ↔ "duffle sac")
        if label_words & alias_words:
            kept.append(alias)
            continue

        # Check for stem overlap between any pair of words
        has_stem = False
        for lw in label_words:
            for aw in alias_words:
                if _share_stem(lw, aw):
                    has_stem = True
                    break
            if has_stem:
                break

        if has_stem:
            kept.append(alias)
            continue

        # No lexical, word-overlap, or synonym evidence → eject
        ejected.append(alias)

    return label, kept, ejected


def embed_candidates(
    model,
    candidates: List[str],
    batch_size: int = 64
) -> np.ndarray:
    """
    Embed candidate phrases using SentenceTransformer.
    Returns: numpy array of shape (n_candidates, embedding_dim)
    """
    log_stage("lexicon", f"Embedding {len(candidates)} candidates...")
    
    embeddings = model.encode(
        candidates,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True  # For cosine similarity
    )
    
    return embeddings


def cluster_candidates(
    embeddings: np.ndarray,
    distance_threshold: float = CONCEPT_CLUSTER_DISTANCE_THRESHOLD
) -> np.ndarray:
    """
    Cluster candidate embeddings using Agglomerative Clustering.
    Returns: array of cluster labels
    """
    log_stage("lexicon", f"Clustering with distance threshold={distance_threshold}...")
    
    # Compute cosine distance matrix
    distances = cosine_distances(embeddings)
    
    # Agglomerative clustering with precomputed distances
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric='precomputed',
        linkage='average'
    )
    
    labels = clustering.fit_predict(distances)
    n_clusters = len(set(labels))
    log_stage("lexicon", f"Found {n_clusters} concept clusters")
    
    return labels


def select_cluster_label(
    candidates: List[str],
    embeddings: np.ndarray,
    freq_map: Dict[str, int]
) -> str:
    """
    Select the best label for a cluster.
    Strategy: use most frequent alias, break ties by embedding nearest to centroid.
    """
    if len(candidates) == 1:
        return candidates[0]
    
    # Sort by frequency
    sorted_by_freq = sorted(candidates, key=lambda c: freq_map.get(c, 0), reverse=True)
    max_freq = freq_map.get(sorted_by_freq[0], 0)
    
    # Get all candidates with max frequency
    top_freq_candidates = [c for c in sorted_by_freq if freq_map.get(c, 0) == max_freq]
    
    if len(top_freq_candidates) == 1:
        return top_freq_candidates[0]
    
    # Break tie: nearest to centroid
    centroid = embeddings.mean(axis=0, keepdims=True)
    distances = cosine_distances(embeddings, centroid).flatten()
    
    # Find index of nearest among top_freq_candidates
    best_idx = None
    best_dist = float('inf')
    for i, c in enumerate(candidates):
        if c in top_freq_candidates and distances[i] < best_dist:
            best_dist = distances[i]
            best_idx = i
    
    return candidates[best_idx] if best_idx is not None else sorted_by_freq[0]


def assign_taxonomy_bucket(label: str, aliases: List[str]) -> str:
    """
    Assign a taxonomy bucket to a concept based on keyword rules.
    Returns bucket name or 'other'.
    """
    # Combine label and aliases for matching
    all_terms = [label.lower()] + [a.lower() for a in aliases]
    combined_text = " ".join(all_terms)
    
    # Check each bucket's keywords
    for bucket, keywords in TAXONOMY_RULES.items():
        for kw in keywords:
            kw_lower = kw.lower()
            # Check for keyword in any alias or label
            if kw_lower in combined_text:
                return bucket
            # Check if any alias starts with keyword
            for term in all_terms:
                if term.startswith(kw_lower) or kw_lower in term.split():
                    return bucket
    
    return "other"


def generate_concept_id(label: str, index: int) -> str:
    """Generate a stable concept ID."""
    # Use hash for stability + index for uniqueness
    hash_part = hashlib.md5(label.encode()).hexdigest()[:6]
    return f"CONCEPT_{index:04d}_{hash_part}"


def build_lexicon() -> Tuple[pd.DataFrame, Dict]:
    """
    Main lexicon building function.
    
    Returns:
        Tuple of (lexicon_df, taxonomy_dict)
        
    Side effects:
        Writes taxonomy/lexicon_v1.csv
        Writes taxonomy/taxonomy_v1.json
    """
    set_all_seeds()
    
    log_stage("lexicon", "Starting lexicon building...")
    
    # Load candidates
    candidates_path = DATA_PROCESSED / "candidates.csv"
    if not candidates_path.exists():
        raise FileNotFoundError(f"Candidates file not found: {candidates_path}. Run candidates extraction first.")
    
    candidates_df = pd.read_csv(candidates_path)
    log_stage("lexicon", f"Loaded {len(candidates_df)} candidates")
    
    if len(candidates_df) == 0:
        log_stage("lexicon", "WARNING: No candidates found. Creating empty lexicon.")
        empty_lexicon = pd.DataFrame(columns=[
            "concept_id", "label", "aliases", "languages", "freq_notes", "examples", "rule"
        ])
        empty_lexicon.to_csv(TAXONOMY_DIR / "lexicon_v1.csv", index=False)
        
        empty_taxonomy = {
            "intent": [], "occasion": [], "preferences": [],
            "constraints": [], "lifestyle": [], "next_action": [], "other": []
        }
        with open(TAXONOMY_DIR / "taxonomy_v1.json", "w") as f:
            json.dump(empty_taxonomy, f, indent=2)
        
        return empty_lexicon, empty_taxonomy
    
    # Build frequency map
    freq_map = dict(zip(candidates_df["candidate"], candidates_df["freq_notes"]))
    
    # Get unique example notes per candidate
    example_map = dict(zip(candidates_df["candidate"], candidates_df["example_note_ids"]))
    
    # Get languages per candidate
    lang_map = dict(zip(candidates_df["candidate"], candidates_df["language"]))
    
    # Load model and embed candidates
    model = get_sentence_transformer()
    candidate_list = candidates_df["candidate"].tolist()
    embeddings = embed_candidates(model, candidate_list)
    
    # Cluster candidates
    cluster_labels = cluster_candidates(embeddings)
    
    # Group candidates by cluster
    cluster_candidates_map: Dict[int, List[int]] = defaultdict(list)
    for idx, cluster_id in enumerate(cluster_labels):
        cluster_candidates_map[cluster_id].append(idx)
    
    # Build concepts from clusters
    concepts = []
    ejected_total = 0
    for cluster_id in sorted(cluster_candidates_map.keys()):
        indices = cluster_candidates_map[cluster_id]
        cluster_candidate_list = [candidate_list[i] for i in indices]
        cluster_embeddings = embeddings[indices]
        
        # Skip very small clusters if needed (but keep for MVP)
        if len(cluster_candidate_list) < MIN_ALIASES_PER_CONCEPT:
            continue
        
        # Select label
        label = select_cluster_label(
            cluster_candidate_list, 
            cluster_embeddings,
            freq_map
        )
        
        # Collect aliases (all candidates in cluster except label)
        aliases = [c for c in cluster_candidate_list if c != label]

        # ── Post-clustering pruning ──────────────────────────────
        # Eject single-word aliases that ended up here only because
        # the multilingual embedding model maps short words to
        # nearly-identical vectors (e.g. végan ↔ taupe = 0.065).
        label, aliases, ejected = prune_cluster(label, aliases, {})
        if ejected:
            ejected_total += len(ejected)
            # Turn each ejected alias into its own singleton concept
            for ej in ejected:
                concepts.append({
                    "label": ej,
                    "aliases": [],
                    "languages": set(lang_map.get(ej, "").split("|")),
                    "freq_notes": freq_map.get(ej, 0),
                    "examples": list(
                        set(example_map.get(ej, "").split("|"))
                    )[:5],
                })
        # ─────────────────────────────────────────────────────────
        
        # Aggregate stats (only label + kept aliases)
        kept_members = [label] + aliases
        total_freq = sum(freq_map.get(c, 0) for c in kept_members)
        all_langs = set()
        all_examples = set()
        for c in kept_members:
            if c in lang_map:
                all_langs.update(lang_map[c].split("|"))
            if c in example_map:
                all_examples.update(example_map[c].split("|"))
        
        concepts.append({
            "label": label,
            "aliases": aliases,
            "languages": all_langs,
            "freq_notes": total_freq,
            "examples": list(all_examples)[:5]
        })
    
    # Sort by frequency (cluster size proxy) for stable IDs
    if ejected_total:
        log_stage("lexicon", f"Post-clustering pruning: ejected {ejected_total} misplaced aliases → new singletons")
    concepts.sort(key=lambda c: (-c["freq_notes"], c["label"]))
    
    # Assign concept IDs and taxonomy buckets
    lexicon_rows = []
    taxonomy: Dict[str, List[str]] = {
        "intent": [], "occasion": [], "preferences": [],
        "constraints": [], "lifestyle": [], "next_action": [], "other": []
    }
    
    for idx, concept in enumerate(concepts):
        concept_id = generate_concept_id(concept["label"], idx)
        bucket = assign_taxonomy_bucket(concept["label"], concept["aliases"])
        
        taxonomy[bucket].append(concept_id)
        
        lexicon_rows.append({
            "concept_id": concept_id,
            "label": concept["label"],
            "aliases": "|".join(concept["aliases"]) if concept["aliases"] else "",
            "languages": "|".join(sorted(concept["languages"])),
            "freq_notes": concept["freq_notes"],
            "examples": "|".join(concept["examples"]),
            "rule": f"bucket={bucket}"
        })
    
    lexicon_df = pd.DataFrame(lexicon_rows)
    
    # ── Merge with existing enriched vocabulary (preserve enriched concepts) ──
    lexicon_json_path = TAXONOMY_DIR / "lexicon_v1.json"
    existing_vocab = {}
    if lexicon_json_path.exists():
        try:
            with open(lexicon_json_path, "r", encoding="utf-8") as f:
                existing_vocab = json.load(f)
        except Exception:
            existing_vocab = {}
    
    # Build lookup from pipeline-discovered concepts (by label)
    pipeline_by_label = {}
    for row in lexicon_rows:
        pipeline_by_label[row["label"].lower()] = row
    
    # Start with existing enriched vocabulary, update freq_notes/examples from pipeline
    merged_json = {}
    for cid, entry in existing_vocab.items():
        label = entry.get("label", "").lower()
        merged_json[cid] = dict(entry)
        # Update freq_notes and examples if pipeline found this concept
        if label in pipeline_by_label:
            prow = pipeline_by_label[label]
            merged_json[cid]["freq_notes"] = prow["freq_notes"]
            merged_json[cid]["examples"] = prow["examples"].split("|") if prow["examples"] else []
    
    # Add any NEW concepts discovered by pipeline that aren't in enriched vocabulary
    existing_labels = {e.get("label", "").lower() for e in existing_vocab.values()}
    new_from_pipeline = 0
    for row in lexicon_rows:
        if row["label"].lower() not in existing_labels:
            merged_json[row["concept_id"]] = {
                "label": row["label"],
                "aliases": row["aliases"].split("|") if row["aliases"] else [],
                "languages": row["languages"],
                "freq_notes": row["freq_notes"],
                "examples": row["examples"].split("|") if row["examples"] else [],
                "rule": row["rule"]
            }
            new_from_pipeline += 1
    
    if existing_vocab:
        log_stage("lexicon", f"Merged: {len(existing_vocab)} enriched + {new_from_pipeline} new from pipeline = {len(merged_json)} total")
    
    # Write merged JSON
    with open(lexicon_json_path, "w", encoding="utf-8") as f:
        json.dump(merged_json, f, indent=2, ensure_ascii=False)
    log_stage("lexicon", f"Wrote {len(merged_json)} concepts to {lexicon_json_path}")
    
    # Write merged CSV
    merged_rows = []
    for cid, entry in merged_json.items():
        aliases = entry.get("aliases", [])
        examples = entry.get("examples", [])
        merged_rows.append({
            "concept_id": cid,
            "label": entry.get("label", ""),
            "aliases": "|".join(aliases) if isinstance(aliases, list) else str(aliases),
            "languages": entry.get("languages", "ALL"),
            "freq_notes": entry.get("freq_notes", 0),
            "examples": "|".join(examples) if isinstance(examples, list) else str(examples),
            "rule": entry.get("rule", "bucket=other")
        })
    merged_df = pd.DataFrame(merged_rows)
    lexicon_path = TAXONOMY_DIR / "lexicon_v1.csv"
    merged_df.to_csv(lexicon_path, index=False)
    log_stage("lexicon", f"Wrote {len(merged_df)} concepts to {lexicon_path}")
    
    # Rebuild taxonomy from merged vocabulary
    taxonomy = {
        "intent": [], "occasion": [], "preferences": [],
        "constraints": [], "lifestyle": [], "next_action": [], "other": []
    }
    for cid, entry in merged_json.items():
        bucket = entry.get("rule", "bucket=other").replace("bucket=", "")
        if bucket not in taxonomy:
            bucket = "other"
        taxonomy[bucket].append(cid)
    
    taxonomy_path = TAXONOMY_DIR / "taxonomy_v1.json"
    with open(taxonomy_path, "w", encoding="utf-8") as f:
        json.dump(taxonomy, f, indent=2, ensure_ascii=False)
    log_stage("lexicon", f"Wrote taxonomy to {taxonomy_path}")
    
    # Summary
    log_stage("lexicon", "Taxonomy bucket distribution:")
    for bucket, concept_ids in taxonomy.items():
        log_stage("lexicon", f"  {bucket}: {len(concept_ids)} concepts")
    
    log_stage("lexicon", "Lexicon building complete!")
    
    return lexicon_df, taxonomy


def main():
    """CLI entry point."""
    try:
        build_lexicon()
    except Exception as e:
        log_stage("lexicon", f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
