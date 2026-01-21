"""
Candidate extraction stage: Extract candidate keyphrases/terms from notes.

This module uses:
- YAKE (Yet Another Keyword Extractor)
- RAKE-NLTK (Rapid Automatic Keyword Extraction)
- TF-IDF n-grams as fallback
- Entity extraction for budget, dates, etc.

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

# Expanded stopwords - words that are NOT useful on their own
COMMON_STOPWORDS = {
    # French
    "le", "la", "les", "un", "une", "des", "de", "du", "et", "en", "a", "à",
    "est", "sont", "avec", "pour", "dans", "sur", "par", "plus", "aussi",
    "ce", "cette", "ces", "il", "elle", "ils", "elles", "nous", "vous",
    "très", "bien", "comme", "tout", "tous", "toutes", "peut", "être",
    "qui", "que", "quoi", "dont", "où", "si", "mais", "ou", "donc",
    "ans", "année", "années", "mois", "jour", "jours", "fois", "peu",
    "fait", "faire", "dit", "bon", "bonne", "chez", "après", "avant",
    # English
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "is", "are", "was", "were", "be", "been", "have", "has", "had",
    "this", "that", "these", "those", "it", "its", "he", "she", "they",
    "more", "very", "also", "just", "about", "would", "could", "should",
    "years", "year", "old", "time", "times", "around", "some", "will",
    "been", "being", "which", "their", "there", "then", "than", "when",
    # Italian
    "il", "lo", "la", "i", "gli", "le", "un", "una", "e", "di", "da", "che",
    "anni", "anno", "circa", "sono", "con", "per", "non", "come", "anche",
    "suo", "sua", "suoi", "sue", "molto", "più", "già", "dopo", "prima",
    # Spanish
    "el", "la", "los", "las", "un", "una", "y", "de", "en", "que", "con",
    "años", "año", "es", "son", "para", "por", "como", "más", "muy",
    "su", "sus", "todo", "todos", "esta", "este", "esto", "sin", "sobre",
    # German
    "der", "die", "das", "ein", "eine", "und", "in", "zu", "für", "mit",
    "jahre", "jahr", "ist", "sind", "hat", "haben", "auch", "auf", "bei",
    "von", "nach", "sich", "als", "noch", "oder", "wenn", "nur", "sein",
    # Generic useless words
    "etc", "good", "great", "nice", "bien", "bueno", "gut", "bene",
    "new", "nouveau", "nueva", "nuovo", "neu", "neuen",
    "first", "premier", "primera", "primo", "erste",
    "next", "prochain", "próximo", "prossimo", "nächste",
    "both", "deux", "beide", "entrambi", "ambos",
}

# Words that are useless WITHOUT additional context (like a number or qualifier)
CONTEXT_REQUIRED_WORDS = {
    "budget", "prix", "price", "preis", "prezzo", "precio",  # Need amount
    "age", "âge", "edad", "età", "alter",  # Need number
    "client", "cliente", "kunde", "kundin",  # Too generic alone
    "potential", "potentiel", "potenziale", "potencial",  # Need context
    "excellent", "excelente", "eccellente", "ausgezeichnet",  # Need what's excellent
}

# Translation map to standardize to French
FRENCH_STANDARDIZATION = {
    # Colors
    "black": "noir", "negro": "noir", "nero": "noir", "schwarz": "noir",
    "brown": "marron", "marrón": "marron", "marrone": "marron", "braun": "marron",
    "white": "blanc", "blanco": "blanc", "bianco": "blanc", "weiß": "blanc", "weiss": "blanc",
    "blue": "bleu", "azul": "bleu", "blu": "bleu", "blau": "bleu",
    "red": "rouge", "rojo": "rouge", "rosso": "rouge", "rot": "rouge",
    "green": "vert", "verde": "vert", "grün": "vert",
    "navy": "bleu marine", "marino": "bleu marine",
    "beige": "beige", "cream": "crème", "crema": "crème",
    
    # Materials
    "leather": "cuir", "piel": "cuir", "cuoio": "cuir", "leder": "cuir",
    "canvas": "toile", "tela": "toile", "leinwand": "toile",
    "silk": "soie", "seda": "soie", "seta": "soie", "seide": "soie",
    
    # Products
    "handbag": "sac à main", "bag": "sac", "bolso": "sac", "borsa": "sac", "tasche": "sac",
    "wallet": "portefeuille", "cartera": "portefeuille", "portafoglio": "portefeuille", "brieftasche": "portefeuille",
    "briefcase": "attaché-case", "maletín": "attaché-case", "valigetta": "attaché-case", "aktentasche": "attaché-case",
    "belt": "ceinture", "cinturón": "ceinture", "cintura": "ceinture", "gürtel": "ceinture",
    "scarf": "foulard", "bufanda": "foulard", "sciarpa": "foulard", "schal": "foulard",
    "watch": "montre", "reloj": "montre", "orologio": "montre", "uhr": "montre",
    "jewelry": "bijoux", "jewellery": "bijoux", "joyería": "bijoux", "gioielli": "bijoux", "schmuck": "bijoux",
    
    # Occasions
    "birthday": "anniversaire", "cumpleaños": "anniversaire", "compleanno": "anniversaire", "geburtstag": "anniversaire",
    "wedding": "mariage", "boda": "mariage", "matrimonio": "mariage", "hochzeit": "mariage",
    "christmas": "noël", "navidad": "noël", "natale": "noël", "weihnachten": "noël",
    "gift": "cadeau", "regalo": "cadeau", "geschenk": "cadeau",
    "graduation": "diplôme", "graduación": "diplôme", "laurea": "diplôme", "abschluss": "diplôme",
    
    # Lifestyle
    "travel": "voyage", "viaje": "voyage", "viaggio": "voyage", "reise": "voyage",
    "golf": "golf", "tennis": "tennis", "yoga": "yoga",
    "sailing": "voile", "vela": "voile", "segeln": "voile",
    "art": "art", "arte": "art", "kunst": "art",
    "wine": "vin", "vino": "vin", "wein": "vin",
    "collection": "collection", "colección": "collection", "collezione": "collection", "sammlung": "collection",
    
    # Family
    "husband": "mari", "marido": "mari", "marito": "mari", "ehemann": "mari",
    "wife": "épouse", "esposa": "épouse", "moglie": "épouse", "ehefrau": "épouse",
    "daughter": "fille", "hija": "fille", "figlia": "fille", "tochter": "fille",
    "son": "fils", "hijo": "fils", "figlio": "fils", "sohn": "fils",
    "mother": "mère", "madre": "mère", "mutter": "mère",
    "father": "père", "padre": "père", "vater": "père",
    "family": "famille", "familia": "famille", "famiglia": "famille", "familie": "famille",
    
    # Actions
    "follow up": "suivi", "follow-up": "suivi", "seguimiento": "suivi", "nachverfolgung": "suivi",
    "appointment": "rendez-vous", "cita": "rendez-vous", "appuntamento": "rendez-vous", "termin": "rendez-vous",
    "invitation": "invitation", "invitación": "invitation", "invito": "invitation", "einladung": "invitation",
    
    # Dietary
    "vegan": "végan", "vegano": "végan",
    "vegetarian": "végétarien", "vegetariano": "végétarien", "vegetarisch": "végétarien",
    "allergy": "allergie", "alergia": "allergie", "allergia": "allergie",
    
    # Style
    "classic": "classique", "clásico": "classique", "classico": "classique", "klassisch": "classique",
    "modern": "moderne", "moderno": "moderne",
    "elegant": "élégant", "elegante": "élégant",
    "sophisticated": "raffiné", "sofisticado": "raffiné", "sofisticato": "raffiné", "raffiniert": "raffiné",
    
    # Events
    "event": "événement", "evento": "événement", "veranstaltung": "événement",
    "vip": "vip", "vip event": "événement vip",
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


def standardize_to_french(candidate: str) -> str:
    """Standardize candidate to French if translation exists."""
    candidate_lower = candidate.lower().strip()
    
    # Check direct translation
    if candidate_lower in FRENCH_STANDARDIZATION:
        return FRENCH_STANDARDIZATION[candidate_lower]
    
    # Check multi-word phrases
    words = candidate_lower.split()
    translated_words = []
    for word in words:
        if word in FRENCH_STANDARDIZATION:
            translated_words.append(FRENCH_STANDARDIZATION[word])
        else:
            translated_words.append(word)
    
    return " ".join(translated_words)


def is_valid_candidate(candidate: str) -> bool:
    """Check if candidate is valid after cleaning."""
    if len(candidate) < MIN_TOKEN_LENGTH:
        return False
    
    # Filter out pure numbers
    if candidate.replace(' ', '').replace('.', '').replace(',', '').isdigit():
        return False
    
    # Filter out single very common stopwords
    if candidate in COMMON_STOPWORDS:
        return False
    
    # Filter out candidates with only stopwords
    words = candidate.split()
    if all(w in COMMON_STOPWORDS for w in words):
        return False
    
    # Filter out context-required words when alone
    if candidate in CONTEXT_REQUIRED_WORDS:
        return False
    
    # Filter out patterns like "XX ans", "XX years" (age without context)
    if re.match(r'^\d+\s*(ans|years|años|anni|jahre)$', candidate):
        return False
    
    # Filter out just numbers with K/k (like "5k", "10K")
    if re.match(r'^\d+[kK]$', candidate):
        return False
    
    return True


def extract_entities(text: str) -> List[Tuple[str, float]]:
    """
    Extract meaningful entities from text:
    - Budget amounts (e.g., "budget 5000€", "budget 3-4K")
    - Professions with context
    - Specific preferences
    Returns list of (entity, score) tuples.
    """
    entities = []
    text_lower = text.lower()
    
    # Budget patterns - extract budget WITH amount
    budget_patterns = [
        r'budget\s*(?:de\s*)?(?:environ\s*)?(\d+[\s\-à]+\d+\s*[kK€$]?|\d+\s*[kK€$])',
        r'budget\s*(?:around|about|circa|um|intorno)?\s*(\d+[\s\-to]+\d+\s*[kK€$]?|\d+\s*[kK€$])',
        r'presupuesto\s*(?:de\s*)?(\d+[\s\-a]+\d+\s*[kK€$]?|\d+\s*[kK€$])',
        r'(\d+[\s\-]+\d+\s*[kK€$])\s*(?:budget|flexible)',
        r'(\d+\s*[kK€$])\s*(?:budget|flexible)',
    ]
    
    for pattern in budget_patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            amount = match.strip()
            # Normalize: convert K to €
            amount = re.sub(r'(\d+)\s*[kK]', r'\1000€', amount)
            entities.append((f"budget {amount}", 0.9))
    
    # Age patterns - extract meaningful age context (person + age)
    age_patterns = [
        r'(\w+)\s*,?\s*(\d{2})\s*ans',  # French: "mari, 50 ans"
        r'(\w+)\s*,?\s*(\d{2})\s*years?\s*old',  # English
        r'(\w+)\s*,?\s*(\d{2})\s*años',  # Spanish
        r'(\w+)\s*,?\s*(\d{2})\s*anni',  # Italian
        r'(\w+)\s*,?\s*(\d{2})\s*jahre',  # German
    ]
    
    for pattern in age_patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            if len(match) == 2:
                person, age = match
                # Skip if person is a stopword or number
                if person not in COMMON_STOPWORDS and not person.isdigit():
                    entities.append((f"{person} {age} ans", 0.7))
    
    # Allergy patterns - extract specific allergies
    allergy_patterns = [
        r'allerg(?:ie|y|ia)\s+(?:au?x?\s+)?(\w+)',
        r'intoléran(?:t|ce)\s+(?:au?x?\s+)?(\w+)',
        r'(\w+)\s+allergy',
    ]
    
    for pattern in allergy_patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            if match not in COMMON_STOPWORDS:
                entities.append((f"allergie {match}", 0.85))
    
    # Diet patterns
    diet_patterns = [
        (r'\b(végan|vegan|vegano)\b', "végan"),
        (r'\b(végétarien|vegetarian|vegetariano|vegetarisch)\b', "végétarien"),
        (r'\b(pescetari\w+|pescétari\w+)\b', "pescétarien"),
        (r'\b(sans gluten|gluten.?free|glutenfrei)\b', "sans gluten"),
    ]
    
    for pattern, label in diet_patterns:
        if re.search(pattern, text_lower):
            entities.append((label, 0.9))
    
    # VIP/loyalty patterns
    if re.search(r'\b(vip|fidèle|loyal|excellent\s+client|high\s+potential|alto\s+potenziale)\b', text_lower):
        entities.append(("client vip", 0.8))
    
    return entities


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
    
    # First pass: Entity extraction (budgets, allergies, etc.)
    log_stage("candidates", "Extracting entities (budgets, allergies, etc.)...")
    for idx, (text, note_id, lang) in enumerate(zip(texts, note_ids, languages)):
        if not text or len(text.strip()) < 10:
            continue
        
        # Extract entities
        entity_results = extract_entities(text)
        for entity, score in entity_results:
            cleaned = clean_candidate(entity)
            if cleaned:
                # Standardize to French
                standardized = standardize_to_french(cleaned)
                candidate_stats[standardized]["languages"].add(lang)
                candidate_stats[standardized]["note_ids"].append(note_id)
                candidate_stats[standardized]["scores"].append(score)
    
    # Second pass: YAKE + RAKE per document
    log_stage("candidates", "Extracting with YAKE and RAKE...")
    for idx, (text, note_id, lang) in enumerate(zip(texts, note_ids, languages)):
        if not text or len(text.strip()) < 10:
            continue
        
        # YAKE extraction
        yake_results = extract_yake_candidates(text, lang)
        for kw, score in yake_results:
            cleaned = clean_candidate(kw)
            if is_valid_candidate(cleaned):
                # Standardize to French
                standardized = standardize_to_french(cleaned)
                candidate_stats[standardized]["languages"].add(lang)
                candidate_stats[standardized]["note_ids"].append(note_id)
                candidate_stats[standardized]["scores"].append(score)
        
        # RAKE extraction
        rake_results = extract_rake_candidates(text)
        for phrase, score in rake_results:
            cleaned = clean_candidate(phrase)
            if is_valid_candidate(cleaned):
                # Standardize to French
                standardized = standardize_to_french(cleaned)
                candidate_stats[standardized]["languages"].add(lang)
                candidate_stats[standardized]["note_ids"].append(note_id)
                candidate_stats[standardized]["scores"].append(score)
    
    # Third pass: TF-IDF (global)
    log_stage("candidates", "Extracting with TF-IDF...")
    tfidf_results = extract_tfidf_candidates(texts)
    for doc_idx, candidates in tfidf_results.items():
        note_id = note_ids[doc_idx]
        lang = languages[doc_idx]
        for term, score in candidates:
            cleaned = clean_candidate(term)
            if is_valid_candidate(cleaned):
                # Standardize to French
                standardized = standardize_to_french(cleaned)
                candidate_stats[standardized]["languages"].add(lang)
                candidate_stats[standardized]["note_ids"].append(note_id)
                candidate_stats[standardized]["scores"].append(score)
    
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
