"""
Global configuration for the LVMH Voice-to-Tag pipeline.
All random seeds and constants are centralized here for reproducibility.
"""
import os
from pathlib import Path

# ============================================================
# RANDOM SEEDS (determinism)
# ============================================================
RANDOM_SEED = 42
NUMPY_SEED = 42
SKLEARN_RANDOM_STATE = 42
UMAP_RANDOM_STATE = 42

# ============================================================
# PATHS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # src/shared/config.py -> src/shared -> src -> project root
DATA_DIR = BASE_DIR / "data"
DATA_INPUT = DATA_DIR / "input"  # Renamed from raw to input
DATA_RAW = DATA_INPUT  # Alias for backward compatibility
DATA_PROCESSED = DATA_DIR / "processed"
DATA_OUTPUTS = DATA_DIR / "outputs"
TAXONOMY_DIR = BASE_DIR / "taxonomy"
ACTIVATIONS_DIR = BASE_DIR / "activations"
DASHBOARD_DIR = BASE_DIR / "dashboard"  # React dashboard location
MODELS_DIR = BASE_DIR / "models"
SENTENCE_TRANSFORMERS_CACHE = MODELS_DIR / "sentence_transformers"

# ============================================================
# MODEL CONFIGURATION
# ============================================================
SENTENCE_TRANSFORMER_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# ============================================================
# INPUT SCHEMA
# ============================================================
REQUIRED_COLUMNS = ["ID", "Date", "Duration", "Language", "Length", "Transcription"]

# ============================================================
# CLUSTERING
# ============================================================
# Number of clusters: env var or heuristic min(8, max(3, sqrt(n/2)))
CLUSTERS_K = int(os.environ.get("CLUSTERS_K", 0))  # 0 = use heuristic

# ============================================================
# CANDIDATE EXTRACTION
# ============================================================
MIN_TOKEN_LENGTH = 3
MAX_CANDIDATES_PER_NOTE = 30
MIN_CANDIDATE_FREQ = 2  # minimum notes a candidate must appear in

# ============================================================
# CONCEPT GROUPING
# ============================================================
# Higher threshold = more merging
CONCEPT_CLUSTER_DISTANCE_THRESHOLD = 0.20  # cosine distance for agglomerative clustering
MIN_ALIASES_PER_CONCEPT = 1
TOP_CONCEPTS_PER_CLUSTER = 5  # number of top concepts to show in profiles

# ============================================================
# CONCEPT DETECTION
# ============================================================
MAX_ALIAS_MATCHES_PER_NOTE = 3  # max times same alias can be recorded per note

# ============================================================
# SUPPORTED LANGUAGES
# ============================================================
SUPPORTED_LANGUAGES = {"FR", "EN", "IT", "ES", "DE", "PT", "NL"}

# ============================================================
# PRIVACY & RGPD/GDPR COMPLIANCE
# ============================================================
# Enable anonymization of sensitive data in transcriptions
ENABLE_ANONYMIZATION = os.environ.get("ENABLE_ANONYMIZATION", "true").lower() == "true"

# Anonymization aggressiveness (false = conservative, true = aggressive)
ANONYMIZATION_AGGRESSIVE = os.environ.get("ANONYMIZATION_AGGRESSIVE", "false").lower() == "true"

# ============================================================
# TAXONOMY BUCKET KEYWORDS (multilingual)
# ============================================================
TAXONOMY_RULES = {
    "intent": [
        # Purchase/buying intent
        "buy", "purchase", "achat", "acheter", "acquisto", "compra", "comprar", "kauf", "kaufen",
        "gift", "cadeau", "regalo", "geschenk",
        "looking for", "cherche", "cerca", "busca", "sucht",
        "interested", "intéressé", "interessato", "interesado", "interessiert",
        "wants", "want", "veut", "vuole", "quiere", "will",
        "appointment", "rendez-vous", "appuntamento", "cita", "termin",
        "visit", "visite", "visita", "besuch",
        "repair", "réparation", "riparazione", "reparación", "reparatur",
        "exchange", "échange", "cambio", "tausch",
        "return", "retour", "reso", "devolución", "rückgabe",
    ],
    "occasion": [
        "birthday", "anniversaire", "compleanno", "cumpleaños", "geburtstag",
        "anniversary", "anniversaire mariage", "anniversario", "aniversario", "jubiläum",
        "wedding", "mariage", "matrimonio", "boda", "hochzeit",
        "graduation", "diplôme", "diplomation", "laurea", "graduación", "abschluss",
        "christmas", "noël", "natale", "navidad", "weihnachten",
        "holiday", "vacances", "vacanza", "vacaciones", "urlaub",
        "trip", "voyage", "viaggio", "viaje", "reise",
        "event", "événement", "evento", "veranstaltung",
        "celebration", "fête", "festa", "celebración", "feier",
        "new job", "nouveau travail", "nuovo lavoro", "nuevo trabajo", "neue arbeit",
    ],
    "preferences": [
        # Colors
        "black", "noir", "nero", "negro", "schwarz",
        "brown", "marron", "marrone", "marrón", "braun",
        "cognac", "bordeaux", "burgundy",
        "navy", "marine", "blu", "azul", "blau",
        "beige", "cream", "crème",
        "rose gold", "or rose", "oro rosa",
        # Materials
        "leather", "cuir", "cuoio", "piel", "leder",
        "canvas", "toile", "tela", "leinwand",
        # Styles
        "classic", "classique", "classico", "clásico", "klassisch",
        "modern", "moderne", "moderno",
        "elegant", "élégant", "elegante",
        "sophisticated", "raffiné", "sofisticato", "sofisticado", "raffiniert",
        "young", "jeune", "giovane", "joven", "jung",
        # Categories
        "handbag", "sac", "borsa", "bolso", "handtasche",
        "wallet", "portefeuille", "portafoglio", "cartera", "brieftasche",
        "briefcase", "attaché-case", "valigetta", "maletín", "aktentasche",
        "travel bag", "sac voyage", "borsa viaggio", "bolsa viaje", "reisetasche",
        "accessories", "accessoires", "accessori", "accesorios", "zubehör",
    ],
    "constraints": [
        "budget", "prix", "prezzo", "precio", "preis",
        "allergy", "allergie", "allergia", "alergia",
        "intolerance", "intolérance", "intolleranza", "intolerancia", "unverträglichkeit",
        "vegetarian", "végétarien", "vegetariano", "vegetarisch",
        "vegan", "végan",
        "pescatarian", "pescétarien", "pescetariano", "pescetariano",
        "dislike", "n'aime pas", "non piace", "no gusta", "mag nicht",
        "urgent", "urgent", "urgente", "dringend",
        "size", "taille", "taglia", "tamaño", "größe",
        "nickel", "nichel", "níquel",
        "gluten",
        "latex",
        "nut", "noix", "noce", "nuez", "nuss",
        "shellfish", "fruits de mer", "frutti di mare", "mariscos", "meeresfrüchte",
    ],
    "lifestyle": [
        # Sports
        "golf", "tennis", "yoga", "pilates", "running", "course", "marathon",
        "equestrian", "dressage", "sailing", "voile", "vela", "segeln",
        "kitesurf", "ski", "skiing",
        # Hobbies
        "art", "collection", "collectionne", "colleziona", "colecciona", "sammelt",
        "photography", "photographie", "fotografia", "fotografía",
        "wine", "vin", "vino", "wein",
        # Travel
        "travel", "voyage", "viaggio", "viaje", "reise",
        "asia", "asie", "europe", "europa", "america", "amérique",
        "maldives", "kenya", "safari", "bali", "dubai",
        # Professional
        "banker", "banquier", "banchiere", "banquero",
        "lawyer", "avocat", "avvocato", "abogado", "anwalt",
        "doctor", "médecin", "medico", "médico", "arzt",
        "surgeon", "chirurgien", "chirurgo", "cirujano",
        "architect", "architecte", "architetto", "arquitecto",
        "entrepreneur", "designer",
        # Family
        "family", "famille", "famiglia", "familia",
        "husband", "mari", "marito", "marido", "ehemann",
        "wife", "femme", "moglie", "esposa", "ehefrau",
        "daughter", "fille", "figlia", "hija", "tochter",
        "son", "fils", "figlio", "hijo", "sohn",
    ],
    "next_action": [
        "follow up", "rappeler", "richiamare", "llamar", "nachfassen",
        "call", "appeler", "telefonare", "llamar", "anrufen",
        "send", "envoyer", "inviare", "enviar", "senden",
        "invite", "inviter", "invitare", "invitar", "einladen",
        "schedule", "planifier", "programmare", "programar", "planen",
        "contact", "contacter", "contattare", "contactar", "kontaktieren",
        "show", "montrer", "mostrare", "mostrar", "zeigen",
        "present", "présenter", "presentare", "presentar", "präsentieren",
        "preview", "avant-première", "anteprima", "avance",
        "vip event", "événement vip", "evento vip",
    ],
}

# ============================================================
# PLAYBOOKS FILE
# ============================================================
PLAYBOOKS_FILE = ACTIVATIONS_DIR / "playbooks.yml"
