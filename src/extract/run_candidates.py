"""
Candidate extraction stage: Extract candidate keyphrases/terms from notes.

This module uses:
- YAKE (Yet Another Keyword Extractor)
- RAKE-NLTK (Rapid Automatic Keyword Extraction)
- TF-IDF n-grams as fallback

Output: data/processed/candidates.csv
"""
import sys
import re
from collections import defaultdict
from typing import List, Dict, Tuple, Set, Optional
import pandas as pd
import numpy as np

# NLP libraries
import yake
from rake_nltk import Rake
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import (
    DATA_PROCESSED, MIN_TOKEN_LENGTH, MAX_CANDIDATES_PER_NOTE, 
    MIN_CANDIDATE_FREQ, SUPPORTED_LANGUAGES, RANDOM_SEED
)
from src.utils import log_stage, set_all_seeds, normalize_text


# Language code mapping for YAKE
YAKE_LANG_MAP = {
    "FR": "fr", "EN": "en", "IT": "it", "ES": "es", "DE": "de"
}

# Stopwords for basic filtering (common across languages)
COMMON_STOPWORDS = {
    "le", "la", "les", "un", "une", "des", "de", "du", "et", "en", "a", "à",
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "il", "lo", "la", "i", "gli", "le", "un", "una", "e", "di", "da", "che",
    "el", "la", "los", "las", "un", "una", "y", "de", "en", "que", "con",
    "der", "die", "das", "ein", "eine", "und", "in", "zu", "für", "mit",
    "est", "sont", "avec", "pour", "dans", "sur", "par", "plus", "aussi",
    "is", "are", "was", "were", "be", "been", "have", "has", "had",
    "this", "that", "these", "those", "it", "its", "he", "she", "they",
    "ce", "cette", "ces", "il", "elle", "ils", "elles", "nous", "vous",
    "très", "bien", "comme", "très", "tout", "tous", "toutes", "peut",
    "more", "very", "also", "just", "about", "would", "could", "should",
}


def clean_candidate(text: str) -> str:
    """
    Clean and normalize a candidate phrase.
    - lowercase
    - strip punctuation from edges
    - collapse whitespace
    - remove very short tokens
    """
    text = text.lower().strip()
    # Remove leading/trailing punctuation
    text = re.sub(r'^[^\w]+|[^\w]+$', '', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text)
    return text


def is_valid_candidate(candidate: str) -> bool:
    """Check if candidate is valid after cleaning."""
    if len(candidate) < MIN_TOKEN_LENGTH:
        return False
    # Filter out pure numbers
    if candidate.replace(' ', '').isdigit():
        return False
    # Filter out single very common stopwords
    if candidate in COMMON_STOPWORDS:
        return False
    # Filter out candidates with only stopwords
    words = candidate.split()
    if all(w in COMMON_STOPWORDS for w in words):
        return False
    return True


def extract_yake_candidates(text: str, language: str, max_keywords: int = 20) -> List[Tuple[str, float]]:
    """
    Extract keywords using YAKE.
    Returns list of (keyword, score) tuples. Lower score = better in YAKE.
    """
    yake_lang = YAKE_LANG_MAP.get(language, "en")
    
    try:
        kw_extractor = yake.KeywordExtractor(
            lan=yake_lang,
            n=3,  # max n-gram size
            dedupLim=0.7,
            dedupFunc='seqm',
            windowsSize=1,
            top=max_keywords,
            features=None
        )
        keywords = kw_extractor.extract_keywords(text)
        # Convert YAKE score to a 0-1 score where higher is better
        # YAKE scores are lower for better keywords, typically 0-1
        results = []
        for kw, score in keywords:
            # Invert score: 1 / (1 + score) to get higher = better
            normalized_score = 1.0 / (1.0 + score)
            results.append((kw, normalized_score))
        return results
    except Exception as e:
        return []


def extract_rake_candidates(text: str, max_keywords: int = 20) -> List[Tuple[str, float]]:
    """
    Extract keywords using RAKE-NLTK.
    Returns list of (keyword, score) tuples.
    """
    try:
        rake = Rake(
            min_length=1,
            max_length=4,
            include_repeated_phrases=False
        )
        rake.extract_keywords_from_text(text)
        ranked = rake.get_ranked_phrases_with_scores()[:max_keywords]
        
        if not ranked:
            return []
        
        # Normalize scores to 0-1 range
        max_score = max(s for s, _ in ranked) if ranked else 1.0
        results = []
        for score, phrase in ranked:
            normalized_score = score / max_score if max_score > 0 else 0.0
            results.append((phrase, normalized_score))
        return results
    except Exception as e:
        return []


def extract_tfidf_candidates(
    texts: List[str], 
    max_features: int = 500,
    ngram_range: Tuple[int, int] = (1, 3)
) -> Dict[int, List[Tuple[str, float]]]:
    """
    Extract TF-IDF based candidates from all texts.
    Returns dict mapping doc index to list of (term, score) tuples.
    """
    try:
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=1,
            max_df=0.9,
            lowercase=True,
            token_pattern=r'(?u)\b[a-zA-ZàâäéèêëïîôùûüçÀÂÄÉÈÊËÏÎÔÙÛÜÇáéíóúñüÁÉÍÓÚÑÜàèìòùÀÈÌÒÙäöüßÄÖÜ]{2,}\b'
        )
        
        tfidf_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()
        
        results = {}
        for doc_idx in range(tfidf_matrix.shape[0]):
            doc_vector = tfidf_matrix[doc_idx].toarray().flatten()
            # Get top terms for this document
            top_indices = doc_vector.argsort()[-MAX_CANDIDATES_PER_NOTE:][::-1]
            doc_candidates = []
            for idx in top_indices:
                if doc_vector[idx] > 0:
                    doc_candidates.append((feature_names[idx], float(doc_vector[idx])))
            results[doc_idx] = doc_candidates
        
        return results
    except Exception as e:
        return {}


def run_candidates() -> pd.DataFrame:
    """
    Main candidate extraction function.
    
    Returns:
        DataFrame with candidate information
        
    Side effects:
        Writes data/processed/candidates.csv
    """
    set_all_seeds()
    
    log_stage("candidates", "Starting candidate extraction...")
    
    # Load cleaned notes
    notes_path = DATA_PROCESSED / "notes_clean.parquet"
    if not notes_path.exists():
        raise FileNotFoundError(f"Notes file not found: {notes_path}. Run ingest first.")
    
    notes_df = pd.read_parquet(notes_path)
    log_stage("candidates", f"Loaded {len(notes_df)} notes")
    
    # Dictionary to track candidates across all notes
    # candidate -> {languages: set, note_ids: list, scores: list}
    candidate_stats: Dict[str, Dict] = defaultdict(lambda: {
        "languages": set(),
        "note_ids": [],
        "scores": []
    })
    
    # Extract from each note using multiple methods
    texts = notes_df["text"].tolist()
    note_ids = notes_df["note_id"].tolist()
    languages = notes_df["language"].tolist()
    
    # First pass: YAKE + RAKE per document
    log_stage("candidates", "Extracting with YAKE and RAKE...")
    for idx, (text, note_id, lang) in enumerate(zip(texts, note_ids, languages)):
        if not text or len(text.strip()) < 10:
            continue
        
        # YAKE extraction
        yake_results = extract_yake_candidates(text, lang)
        for kw, score in yake_results:
            cleaned = clean_candidate(kw)
            if is_valid_candidate(cleaned):
                candidate_stats[cleaned]["languages"].add(lang)
                candidate_stats[cleaned]["note_ids"].append(note_id)
                candidate_stats[cleaned]["scores"].append(score)
        
        # RAKE extraction
        rake_results = extract_rake_candidates(text)
        for phrase, score in rake_results:
            cleaned = clean_candidate(phrase)
            if is_valid_candidate(cleaned):
                candidate_stats[cleaned]["languages"].add(lang)
                candidate_stats[cleaned]["note_ids"].append(note_id)
                candidate_stats[cleaned]["scores"].append(score)
    
    # Second pass: TF-IDF (global)
    log_stage("candidates", "Extracting with TF-IDF...")
    tfidf_results = extract_tfidf_candidates(texts)
    for doc_idx, candidates in tfidf_results.items():
        note_id = note_ids[doc_idx]
        lang = languages[doc_idx]
        for term, score in candidates:
            cleaned = clean_candidate(term)
            if is_valid_candidate(cleaned):
                candidate_stats[cleaned]["languages"].add(lang)
                candidate_stats[cleaned]["note_ids"].append(note_id)
                candidate_stats[cleaned]["scores"].append(score)
    
    # Filter candidates by minimum frequency
    log_stage("candidates", f"Total raw candidates: {len(candidate_stats)}")
    
    filtered_candidates = {}
    for candidate, stats in candidate_stats.items():
        unique_notes = set(stats["note_ids"])
        if len(unique_notes) >= MIN_CANDIDATE_FREQ:
            filtered_candidates[candidate] = {
                "languages": stats["languages"],
                "note_ids": list(unique_notes),
                "scores": stats["scores"]
            }
    
    log_stage("candidates", f"Candidates after frequency filter (>={MIN_CANDIDATE_FREQ}): {len(filtered_candidates)}")
    
    # Build output DataFrame
    rows = []
    for candidate, stats in sorted(filtered_candidates.items()):
        note_ids_list = sorted(stats["note_ids"])
        avg_score = np.mean(stats["scores"]) if stats["scores"] else 0.0
        
        rows.append({
            "candidate": candidate,
            "language": "|".join(sorted(stats["languages"])),
            "freq_notes": len(note_ids_list),
            "avg_score": round(avg_score, 4),
            "example_note_ids": "|".join(note_ids_list[:5])  # Store up to 5 examples
        })
    
    candidates_df = pd.DataFrame(rows)
    
    # Sort by frequency descending, then alphabetically
    candidates_df = candidates_df.sort_values(
        ["freq_notes", "candidate"], 
        ascending=[False, True]
    ).reset_index(drop=True)
    
    # Write output
    output_path = DATA_PROCESSED / "candidates.csv"
    candidates_df.to_csv(output_path, index=False)
    log_stage("candidates", f"Wrote {len(candidates_df)} candidates to {output_path}")
    
    # Summary
    log_stage("candidates", f"Top 10 candidates by frequency:")
    for _, row in candidates_df.head(10).iterrows():
        log_stage("candidates", f"  {row['candidate']} (freq={row['freq_notes']}, lang={row['language']})")
    
    log_stage("candidates", "Candidate extraction complete!")
    
    return candidates_df


def main():
    """CLI entry point."""
    try:
        run_candidates()
    except Exception as e:
        log_stage("candidates", f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
