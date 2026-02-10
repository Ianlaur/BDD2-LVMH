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

from server.shared.config import (
    DATA_PROCESSED, MIN_TOKEN_LENGTH, MAX_CANDIDATES_PER_NOTE, 
    MIN_CANDIDATE_FREQ, SUPPORTED_LANGUAGES, RANDOM_SEED
)
from server.shared.utils import log_stage, set_all_seeds, normalize_text


# Language code mapping for YAKE
YAKE_LANG_MAP = {
    "FR": "fr", "EN": "en", "IT": "it", "ES": "es", "DE": "de",
    "PT": "pt", "NL": "nl"
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
    "se", "ne", "pas", "sa", "ses", "mon", "ma", "mes", "ton", "ta", "tes",
    "car", "ni", "au", "aux", "vers", "entre", "sous",
    # English
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "is", "are", "was", "were", "be", "been", "have", "has", "had",
    "this", "that", "these", "those", "it", "its", "he", "she", "they",
    "more", "very", "also", "just", "about", "would", "could", "should",
    "years", "year", "old", "time", "times", "around", "some", "will",
    "been", "being", "which", "their", "there", "then", "than", "when",
    "from", "with", "not", "who", "what", "how", "where", "into",
    "him", "her", "them", "his", "our", "your", "we", "you",
    "up", "out", "so", "no", "do", "did", "does", "done", "can",
    "may", "if", "each", "all", "any", "own", "other", "both",
    "here", "now", "get", "got", "back", "only", "still", "too",
    "mrs", "mr", "ms", "dr", "sr", "sra", "frau", "herr", "signor",
    "signora", "señor", "señora",
    # Italian
    "il", "lo", "la", "i", "gli", "le", "un", "una", "e", "di", "da", "che",
    "anni", "anno", "circa", "sono", "con", "per", "non", "come", "anche",
    "suo", "sua", "suoi", "sue", "molto", "più", "già", "dopo", "prima",
    "del", "della", "dei", "degli", "delle", "nel", "nella", "nei",
    # Spanish
    "el", "la", "los", "las", "un", "una", "y", "de", "en", "que", "con",
    "años", "año", "es", "son", "para", "por", "como", "más", "muy",
    "su", "sus", "todo", "todos", "esta", "este", "esto", "sin", "sobre",
    "del", "al", "lo", "nos", "les",
    # German
    "der", "die", "das", "ein", "eine", "und", "in", "zu", "für", "mit",
    "jahre", "jahr", "ist", "sind", "hat", "haben", "auch", "auf", "bei",
    "von", "nach", "sich", "als", "noch", "oder", "wenn", "nur", "sein",
    "dem", "den", "des", "einer", "einem", "einen",
    # Portuguese
    "anos", "sra", "senhor", "senhora", "procura", "gosta",
    "orçamento", "em", "ou", "do", "da", "dos", "das", "no", "na",
    # Dutch
    "jaar", "zoekt", "passie", "voor", "het", "een", "van", "op",
    # Generic useless words / filler
    "etc", "good", "great", "nice", "bien", "bueno", "gut", "bene",
    "new", "nouveau", "nueva", "nuovo", "neu", "neuen",
    "first", "premier", "primera", "primo", "erste",
    "next", "prochain", "próximo", "prossimo", "nächste",
    "both", "deux", "beide", "entrambi", "ambos",
    # Pipeline-level fillers that are NOT concepts
    "noted", "mentioned", "discussed", "looking", "looking for",
    "coming up", "coming up in", "follow up", "end of month",
    "prefers", "wants", "likes", "needs", "interested",
    "important", "importante", "regular", "frequent", "potential",
    "client", "cliente", "kunde", "kundin",
    "petit", "petite", "small", "large", "grand", "grande",
    "who", "per", "with", "from",
}

# Words that are useless WITHOUT additional context (like a number or qualifier)
CONTEXT_REQUIRED_WORDS = {
    "budget", "prix", "price", "preis", "prezzo", "precio",  # Need amount
    "age", "âge", "edad", "età", "alter",  # Need number
    "potential", "potentiel", "potenziale", "potencial",  # Need context
    "excellent", "excelente", "eccellente", "ausgezeichnet",  # Need what's excellent
}

# Latin-script regex (covers FR/EN/IT/ES/DE/PT/NL + diacritics)
_LATIN_RE = re.compile(
    r'^[a-zA-ZàâäéèêëïîôùûüçÀÂÄÉÈÊËÏÎÔÙÛÜÇ'
    r'áéíóúñüÁÉÍÓÚÑÜàèìòùÀÈÌÒÙäöüßÄÖÜ'
    r'ãõÃÕêôÊÔ'  # Portuguese
    r"@\s\-\'\.\d]+$"
)


def _is_latin_only(text: str) -> bool:
    """Return True if text contains only Latin-script characters (plus digits, spaces, punctuation)."""
    return bool(_LATIN_RE.match(text))

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
    
    # Reject mixed-script / non-Latin garbage
    # (Non-Latin notes are template-based; useful tokens are already Latin)
    if not _is_latin_only(candidate):
        return False
    
    # Filter out pure numbers
    if candidate.replace(' ', '').replace('.', '').replace(',', '').isdigit():
        return False
    
    # Filter out single very common stopwords
    if candidate in COMMON_STOPWORDS:
        return False
    
    # Filter out candidates where EVERY word is a stopword
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
    
    # Filter out single-character tokens left over
    if len(candidate.replace(' ', '')) < 2:
        return False
    
    # Filter out likely person names (single capitalized word, not a known concept word)
    # These leak from "rendez-vous [PERSON]" patterns and multi-language templates
    if len(words) == 1 and not _is_known_concept_word(candidate):
        return False
    
    return True


# Words that we KNOW are valid single-word concepts for LVMH
_KNOWN_CONCEPT_WORDS = {
    # Products
    "sac", "montre", "foulard", "portefeuille", "ceinture", "bijoux",
    "bracelet", "collier", "bague", "pendentif", "broche", "brooch",
    "cufflinks", "sunglasses", "fragrance", "parfum", "trunk",
    "clutch", "briefcase", "wallet", "belt", "scarf", "watch",
    "pen", "handbag", "boots", "shoes", "ring", "necklace",
    "earrings", "charm", "keychain", "luggage", "backpack",
    # Materials
    "cuir", "soie", "cachemire", "toile", "canvas", "silk", "leather",
    "cashmere", "linen", "velvet", "suede", "denim", "cotton", "wool",
    "crocodile", "python", "ostrich", "alligator",
    # Colors
    "noir", "blanc", "bleu", "rouge", "vert", "marron", "cognac",
    "bordeaux", "beige", "crème", "taupe", "ivory", "champagne",
    "midnight", "navy", "slate", "emerald", "burgundy", "matte",
    "gold", "rose", "silver", "platinum", "black", "white", "blue",
    "red", "green", "brown", "grey", "gray", "pink", "purple",
    "orange", "yellow", "coral", "turquoise", "teal",
    # Lifestyle/hobbies
    "golf", "tennis", "yoga", "pilates", "voile", "sailing",
    "equitation", "ski", "skiing", "surfing", "kitesurf", "polo",
    "opera", "ballet", "photography", "chess", "marathon", "cycling",
    "running", "hiking", "diving", "fishing", "equestrian", "rowing",
    "fencing", "boxing", "sailing", "cricket", "rugby", "football",
    "swimming", "climbing", "bonsai", "gardening", "painting",
    "sculpture", "calligraphy", "cooking", "wine", "art", "vin",
    # Diet/constraints
    "végan", "végétarien", "pescétarien", "allergie", "halal", "kosher",
    "organic", "gluten", "lactose", "keto", "diabetic", "intolerant",
    # Occasions
    "anniversaire", "mariage", "cadeau", "birthday", "wedding",
    "graduation", "retirement", "christmas", "valentine",
    "engagement", "baptism", "communion",
    # Actions/intent
    "collection", "réparation", "échange", "retour", "rendez-vous",
    "bespoke", "personalization", "monogram", "engraving",
    # Luxury concepts
    "luxury", "exclusive", "limited", "rare", "vintage", "designer",
    "couture", "artisan", "sustainable", "heritage", "classic",
    "classique", "elegant", "élégant", "raffiné", "sophisticated",
    "modern", "moderne", "contemporary", "minimalist", "bohemian",
    # Travel / places
    "voyage", "safari", "cruise", "travel",
    # Misc
    "flexible", "renovating", "autumn", "spring", "summer", "winter",
    "treffen", "meeting", "reunión", "incontro",
    "vip", "fidèle", "loyal",
    # Standardized French terms (from FRENCH_STANDARDIZATION values)
    "sac à main", "attaché-case",
}


def _is_known_concept_word(word: str) -> bool:
    """Check if a single word is a known LVMH-relevant concept."""
    return word.lower() in _KNOWN_CONCEPT_WORDS


def _build_domain_scan_patterns():
    """
    Build a list of (compiled_regex, french_form) for domain-specific keyword scanning.

    Covers every key in FRENCH_STANDARDIZATION (the multilingual → French map)
    plus every term in _KNOWN_CONCEPT_WORDS. Each is compiled into a word-boundary
    regex so we get fast, reliable matching across all note languages.
    """
    seen: set = set()
    patterns = []

    # 1. From FRENCH_STANDARDIZATION: foreign term → French concept
    for foreign_term, french_form in FRENCH_STANDARDIZATION.items():
        key = foreign_term.lower()
        if key in seen or key in COMMON_STOPWORDS or len(key) < 3:
            continue
        seen.add(key)
        try:
            pat = re.compile(r'\b' + re.escape(key) + r'\b', re.IGNORECASE)
            patterns.append((pat, french_form))
        except re.error:
            continue

    # 2. From _KNOWN_CONCEPT_WORDS: words that are valid concepts on their own
    for word in _KNOWN_CONCEPT_WORDS:
        key = word.lower()
        if key in seen or key in COMMON_STOPWORDS or len(key) < 3:
            continue
        seen.add(key)
        # The standardized form is the French version if it exists, else the word itself
        french_form = FRENCH_STANDARDIZATION.get(key, key)
        try:
            pat = re.compile(r'\b' + re.escape(key) + r'\b', re.IGNORECASE)
            patterns.append((pat, french_form))
        except re.error:
            continue

    return patterns


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


# Regex to grab runs of Latin-script words from mixed-script text
_LATIN_PHRASE_RE = re.compile(
    r'[A-Za-zàâäéèêëïîôùûüçáéíóúñüàèìòùäöüßãõêô]'
    r'[A-Za-zàâäéèêëïîôùûüçáéíóúñüàèìòùäöüßãõêô\s\-]{2,}',
    re.UNICODE
)


def _extract_latin_phrases(text: str) -> List[str]:
    """
    Pull out contiguous Latin-script phrases from a mixed-script note.

    These template-based notes (JA, ZH, AR, RU, KO, PT, NL) embed the
    meaningful keywords in Latin script, e.g.
      ``Girard様、39歳、Corporate Lawyer。NavyのFragrance Collection``

    We grab each Latin run, split on obvious delimiters, and return
    phrases of 2+ meaningful words.
    """
    raw_spans = _LATIN_PHRASE_RE.findall(text)
    phrases: List[str] = []
    for span in raw_spans:
        # Split on delimiters that separate distinct concepts
        for part in re.split(r'[\.。,،;:\|/]', span):
            part = part.strip()
            if len(part) < 3:
                continue
            # Keep multi-word or known single words
            words = part.split()
            meaningful = [w for w in words if w.lower() not in COMMON_STOPWORDS]
            if meaningful:
                phrases.append(' '.join(meaningful))
    return phrases


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
        
        # For non-Latin-script languages, extract only the Latin-script
        # phrases that appear (these notes are template-based with meaningful
        # keywords already in Latin).
        if lang not in SUPPORTED_LANGUAGES:
            latin_phrases = _extract_latin_phrases(text)
            for phrase in latin_phrases:
                cleaned = clean_candidate(phrase)
                if is_valid_candidate(cleaned):
                    standardized = standardize_to_french(cleaned)
                    candidate_stats[standardized]["languages"].add(lang)
                    candidate_stats[standardized]["note_ids"].append(note_id)
                    candidate_stats[standardized]["scores"].append(0.7)
            continue
        
        # YAKE extraction (supported languages only)
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
    
    # Fourth pass: Domain-specific keyword scan
    # YAKE/RAKE miss common domain words (golf, tennis, black, classic, etc.)
    # because they optimize for "specificity".  We do a direct scan for known
    # LVMH-relevant terms using the standardization dictionary and known words.
    log_stage("candidates", "Extracting domain-specific keywords...")
    _domain_scan_terms = _build_domain_scan_patterns()
    for idx, (text, note_id, lang) in enumerate(zip(texts, note_ids, languages)):
        if not text or len(text.strip()) < 10:
            continue
        text_lower = text.lower()
        for pattern, french_form in _domain_scan_terms:
            if pattern.search(text_lower):
                candidate_stats[french_form]["languages"].add(lang)
                candidate_stats[french_form]["note_ids"].append(note_id)
                candidate_stats[french_form]["scores"].append(0.8)
    
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
