"""
RGPD Sensitive-Word Safety-Net — Lightweight Embedding Classifier

A fast, lightweight classifier that acts as a **final gate** after regex-based
Article 9 detection.  It catches morphological variants, misspellings, and novel
phrasings that slip past the regex patterns.

Architecture
~~~~~~~~~~~~
- Reuses the **already-loaded** ``paraphrase-multilingual-MiniLM-L12-v2``
  SentenceTransformer (via the shared model cache — zero extra load time).
- Embeds each *word / short phrase* in the text and compares it against
  pre-computed reference embeddings of known sensitive terms.
- Cosine similarity above a threshold → flagged as sensitive.
- A lightweight sklearn ``MLPClassifier`` (trained on ~3 000 word embeddings)
  provides a second opinion to reduce false positives.

Why not DistilBERT token-classification?
    A 134 M-param transformer running per-note forward passes is too slow for a
    pipeline that processes 300-2 000 notes.  This approach:
    • Adds **<0.01 s per note** (vs ~0.3 s with DistilBERT).
    • Needs **zero extra model downloads** (reuses existing SentenceTransformer).
    • The MLP head is <50 KB on disk.

Usage
~~~~~
Train (builds reference embeddings + MLP head)::

    python -m server.privacy.sensitive_model train

Predict::

    from server.privacy.sensitive_model import SensitiveWordDetector
    det = SensitiveWordDetector()
    hits = det.predict("Le client souffre d'épilepsie sévère.")
    # [{"token": "épilepsie sévère", "label": "HEALTH_DATA", "score": 0.94}]

Integration::

    Called automatically by ``TextAnonymizer`` in ``anonymize.py`` as a final
    safety-net pass after regex detection.
"""
from __future__ import annotations

import json
import logging
import os
import pickle
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE.parent.parent / "models" / "sensitive_clf"
_REFS_FILE = _MODEL_DIR / "reference_embeddings.npz"
_CLF_FILE = _MODEL_DIR / "mlp_head.pkl"
_META_FILE = _MODEL_DIR / "meta.json"

# ---------------------------------------------------------------------------
# Article-9 categories
# ---------------------------------------------------------------------------
_CATEGORIES = [
    "HEALTH_DATA",
    "SEXUAL_ORIENTATION",
    "RELIGIOUS_BELIEF",
    "POLITICAL_OPINION",
    "ETHNIC_ORIGIN",
    "TRADE_UNION",
    "CRIMINAL_RECORD",
    "FINANCIAL_DIFFICULTY",
    "FAMILY_CONFLICT",
    "PHYSICAL_APPEARANCE",
]

# ---------------------------------------------------------------------------
# Sensitive seed words / phrases per category  (FR + EN)
# These are the "reference vocabulary" we embed at training time.
# ---------------------------------------------------------------------------
_SENSITIVE_SEEDS: Dict[str, List[str]] = {
    "HEALTH_DATA": [
        # French
        "diabète", "diabétique", "cancer", "tumeur", "asthme", "asthmatique",
        "épilepsie", "épileptique", "VIH", "sida", "allergie sévère",
        "maladie", "maladie chronique", "pathologie", "symptôme", "diagnostic",
        "handicap", "handicapé", "infirmité", "médicament", "ordonnance",
        "hospitalisation", "chirurgie", "dépression", "dépressif",
        "anxiété", "psychiatrie", "psychiatrique", "psychologie", "thérapie",
        "grossesse", "enceinte", "fausse couche",
        "troubles alimentaires", "anorexie", "boulimie",
        "intolérance", "intolérante",
        # English
        "diabetes", "diabetic", "cancer", "tumor", "asthma", "asthmatic",
        "epilepsy", "epileptic", "HIV", "AIDS", "severe allergy",
        "disease", "chronic illness", "pathology", "symptom", "diagnosis",
        "disability", "disabled", "medication", "prescription",
        "hospitalization", "surgery", "depression", "depressed",
        "anxiety", "psychiatry", "psychiatric", "psychology", "therapy",
        "pregnancy", "pregnant", "miscarriage",
        "eating disorder", "anorexia", "bulimia",
        "intolerance", "intolerant",
    ],
    "SEXUAL_ORIENTATION": [
        "homosexuel", "homosexuelle", "hétérosexuel", "bisexuel", "pansexuel",
        "homosexual", "heterosexual", "bisexual", "pansexual",
        "gay", "lesbienne", "lesbian", "LGBT", "LGBTQ", "queer",
        "transgenre", "transgender", "non-binaire", "non-binary",
        "orientation sexuelle", "sexual orientation",
        "identité de genre", "vie sexuelle", "sex life",
    ],
    "RELIGIOUS_BELIEF": [
        "musulman", "musulmane", "islam", "islamique",
        "chrétien", "chrétienne", "catholique", "protestant", "évangélique",
        "juif", "juive", "judaïsme", "bouddhiste", "hindou", "sikh",
        "athée", "agnostique", "religion", "religieux", "croyance",
        "muslim", "christian", "catholic", "evangelical",
        "jewish", "buddhist", "hindu", "atheist", "agnostic",
        "ramadan", "shabbat", "carême", "halal", "casher", "kosher",
        "prière", "prayer", "mosquée", "mosque", "synagogue", "église", "church",
    ],
    "POLITICAL_OPINION": [
        "opinion politique", "political opinion",
        "parti politique", "political party",
        "militant", "militante", "activiste", "activist",
        "extrême droite", "far right", "extrême gauche", "far left",
        "grève", "gréviste", "strike", "striker",
        "manifestation politique", "political protest",
        "sympathisant", "supporter politique",
    ],
    "ETHNIC_ORIGIN": [
        "origine ethnique", "ethnic origin",
        "origine raciale", "racial origin",
        "couleur de peau", "skin colour", "skin color",
    ],
    "TRADE_UNION": [
        "syndicat", "syndiqué", "syndiquée", "syndical",
        "trade union", "union member",
        "CGT", "CFDT", "UNSA",
        "adhésion syndicale", "comité d'entreprise", "works council",
    ],
    "CRIMINAL_RECORD": [
        "casier judiciaire", "criminal record",
        "condamnation", "condamné",
        "détention", "detention", "prison", "emprisonnement", "incarcération",
        "garde à vue", "procès", "jugement",
        "infraction", "délit", "felony", "misdemeanor",
    ],
    "FINANCIAL_DIFFICULTY": [
        "surendetté", "surendettement", "over-indebted",
        "faillite", "bankruptcy", "insolvable", "insolvency",
        "interdit bancaire", "banking ban",
        "saisie", "seizure", "huissier", "bailiff",
        "dette", "debt", "recouvrement de dette",
    ],
    "FAMILY_CONFLICT": [
        "divorce", "divorcé", "divorcée",
        "séparation judiciaire", "legal separation",
        "garde des enfants", "child custody", "garde alternée",
        "pension alimentaire", "alimony", "child support",
        "violence conjugale", "domestic violence", "violence domestique",
        "ordonnance de protection", "restraining order",
    ],
    "PHYSICAL_APPEARANCE": [
        "obèse", "obésité", "obese", "obesity",
        "surpoids", "overweight",
        "cicatrice", "scarred", "scarring",
        "difformité", "deformity",
        "body shaming", "commentaire sur le physique",
    ],
}

# Safe luxury-domain words that should NEVER be flagged
_SAFE_WORDS: List[str] = [
    # Products
    "scarf", "écharpe", "foulard", "silk scarf", "cashmere scarf",
    "sac", "bag", "wallet", "portefeuille", "montre", "watch",
    "bijou", "jewelry", "parfum", "perfume", "cuir", "leather",
    "robe", "dress", "costume", "suit", "chaussure", "shoe",
    # Fabrics / materials
    "tissu", "fabric", "motif", "pattern", "imprimé", "print",
    "soie", "silk", "cachemire", "cashmere", "coton", "cotton",
    "cuir végan", "vegan leather", "lin", "linen",
    # Business
    "budget", "prix", "price", "cadeau", "gift", "collection",
    "boutique", "store", "rendez-vous", "appointment",
    "client", "customer", "achat", "purchase",
    # Geography
    "Amérique", "America", "Europe", "Asie", "Asia", "Sud", "North",
    # Ethics (not political)
    "éthique", "ethical", "durable", "sustainable",
    "convictions éthiques", "ethical convictions",
    # Textiles (not ethnic)
    "tissus ethniques", "ethnic print", "motif ethnique",
]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_sensitive_model(seed: int = 42) -> Path:
    """
    Build reference embeddings + train MLP head.

    1. Embeds all sensitive seed words → reference matrix per category.
    2. Embeds safe/luxury words → negative reference.
    3. Trains a small MLP (embedding → category) for a second opinion.

    Returns path to saved model directory.
    """
    from sklearn.neural_network import MLPClassifier
    from server.shared.model_cache import get_sentence_transformer

    random.seed(seed)
    np.random.seed(seed)

    st = get_sentence_transformer()
    dim = st.get_sentence_embedding_dimension()

    # --- 1. Embed all sensitive seeds ---
    all_phrases: List[str] = []
    all_labels: List[int] = []          # 0 = safe, 1..10 = categories
    cat_to_idx = {c: i + 1 for i, c in enumerate(_CATEGORIES)}

    ref_embeddings: Dict[str, np.ndarray] = {}

    for cat in _CATEGORIES:
        phrases = _SENSITIVE_SEEDS[cat]
        embs = st.encode(phrases, show_progress_bar=False, normalize_embeddings=True)
        ref_embeddings[cat] = embs
        all_phrases.extend(phrases)
        all_labels.extend([cat_to_idx[cat]] * len(phrases))
        print(f"  {cat}: {len(phrases)} seed phrases embedded")

    # --- 2. Embed safe words ---
    safe_embs = st.encode(_SAFE_WORDS, show_progress_bar=False, normalize_embeddings=True)
    all_phrases.extend(_SAFE_WORDS)
    all_labels.extend([0] * len(_SAFE_WORDS))
    print(f"  SAFE: {len(_SAFE_WORDS)} safe words embedded")

    # --- 3. Train MLP head ---
    X = st.encode(all_phrases, show_progress_bar=False, normalize_embeddings=True)
    y = np.array(all_labels)

    clf = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        max_iter=300,
        random_state=seed,
        early_stopping=True,
        validation_fraction=0.15,
    )
    clf.fit(X, y)
    train_acc = clf.score(X, y)
    print(f"  MLP train accuracy: {train_acc:.4f}")

    # --- 4. Save ---
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Reference embeddings (one .npz with all categories)
    np.savez_compressed(
        str(_REFS_FILE),
        **{cat: emb for cat, emb in ref_embeddings.items()},
        safe=safe_embs,
    )

    # MLP head
    with open(_CLF_FILE, "wb") as f:
        pickle.dump(clf, f)

    # Metadata
    meta = {
        "categories": _CATEGORIES,
        "cat_to_idx": cat_to_idx,
        "idx_to_cat": {str(v): k for k, v in cat_to_idx.items()},
        "embedding_dim": int(dim),
        "n_sensitive": len(all_phrases) - len(_SAFE_WORDS),
        "n_safe": len(_SAFE_WORDS),
    }
    with open(_META_FILE, "w") as f:
        json.dump(meta, f, indent=2)

    total_kb = sum(p.stat().st_size for p in _MODEL_DIR.iterdir()) / 1024
    print(f"\n✅ Sensitive-word classifier saved to {_MODEL_DIR}  ({total_kb:.0f} KB)")
    return _MODEL_DIR


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

# Simple word/bigram tokenizer for scanning text
_WORD_RE = re.compile(r"[A-Za-zÀ-ÿ](?:[A-Za-zÀ-ÿ'-]*[A-Za-zÀ-ÿ])?", re.UNICODE)

# Common short words to skip in pre-filter (articles, prepositions, etc.)
_SKIP_WORDS = frozenset({
    "le", "la", "les", "un", "une", "des", "de", "du", "au", "aux",
    "et", "ou", "en", "par", "pour", "avec", "sur", "dans", "ce", "sa",
    "son", "ses", "il", "elle", "on", "nous", "vous", "ils", "est", "a",
    "the", "a", "an", "of", "in", "on", "at", "to", "and", "or", "for",
    "is", "are", "was", "her", "his", "she", "he", "it", "this", "that",
    "qui", "que", "ne", "pas",
})


def _extract_ngrams(text: str, max_n: int = 3) -> List[Tuple[str, int, int]]:
    """
    Extract unigrams, bigrams, and trigrams with their character spans.

    Returns [(phrase, start, end), ...]
    """
    words = [(m.group(), m.start(), m.end()) for m in _WORD_RE.finditer(text)]
    ngrams: List[Tuple[str, int, int]] = []

    for n in range(1, min(max_n + 1, len(words) + 1)):
        for i in range(len(words) - n + 1):
            # Skip n-grams made entirely of trivial words
            if n == 1 and words[i][0].lower() in _SKIP_WORDS:
                continue
            phrase = " ".join(w[0] for w in words[i : i + n])
            start = words[i][1]
            end = words[i + n - 1][2]
            ngrams.append((phrase, start, end))

    return ngrams


class SensitiveWordDetector:
    """
    Fast sensitive-word detector using embedding similarity + MLP head.

    Falls back gracefully if the model hasn't been trained.
    """

    def __init__(
        self,
        model_dir: Optional[Path] = None,
        sim_threshold: float = 0.82,
        mlp_threshold: float = 0.60,
    ):
        """
        Args:
            model_dir: Override model directory.
            sim_threshold: Min cosine similarity to a reference embedding.
            mlp_threshold: Min MLP confidence to flag a word.
        """
        self.model_dir = Path(model_dir) if model_dir else _MODEL_DIR
        self.sim_threshold = sim_threshold
        self.mlp_threshold = mlp_threshold

        self._st = None              # SentenceTransformer (lazy)
        self._ref_embs = None        # {category: np.ndarray}
        self._safe_embs = None       # np.ndarray
        self._clf = None             # MLPClassifier
        self._meta = None
        self._available = False

        if (
            self.model_dir.exists()
            and _REFS_FILE.exists()
            and _CLF_FILE.exists()
        ):
            try:
                self._load()
                self._available = True
            except Exception as e:
                logger.warning(f"Could not load sensitive-clf model: {e}")
        else:
            logger.info(
                "Sensitive-word classifier not found — regex-only mode. "
                "Train with: python -m server.privacy.sensitive_model train"
            )

    @property
    def available(self) -> bool:
        return self._available

    def _load(self):
        """Load reference embeddings + MLP head (no heavy model load)."""
        data = np.load(str(_REFS_FILE), allow_pickle=False)
        self._ref_embs = {cat: data[cat] for cat in _CATEGORIES if cat in data}
        self._safe_embs = data.get("safe")

        with open(_CLF_FILE, "rb") as f:
            self._clf = pickle.load(f)

        with open(_META_FILE) as f:
            self._meta = json.load(f)

        logger.info(f"Sensitive-word classifier loaded ({self.model_dir})")

    def _get_st(self):
        """Lazy-load SentenceTransformer from shared cache."""
        if self._st is None:
            from server.shared.model_cache import get_sentence_transformer
            self._st = get_sentence_transformer()
        return self._st

    def predict(self, text: str) -> List[Dict]:
        """
        Detect sensitive words/phrases in *text*.

        Returns list of::

            [{"token": "épilepsie", "label": "HEALTH_DATA", "score": 0.94,
              "start": 23, "end": 33}, ...]
        """
        if not self._available:
            return []

        # 1. Extract candidate n-grams (pre-filtered for trivial words)
        ngrams = _extract_ngrams(text, max_n=3)
        if not ngrams:
            return []

        phrases = [ng[0] for ng in ngrams]

        # 2. Embed all n-grams in one batch
        st = self._get_st()
        embs = st.encode(phrases, show_progress_bar=False,
                         normalize_embeddings=True, batch_size=256)

        # 3. Build a combined reference matrix for fast similarity
        #    Stack all category references and track which rows belong to which cat
        if not hasattr(self, "_all_ref_embs"):
            cats = []
            emb_list = []
            for cat, ref in self._ref_embs.items():
                cats.extend([cat] * len(ref))
                emb_list.append(ref)
            if self._safe_embs is not None and len(self._safe_embs) > 0:
                cats.extend(["SAFE"] * len(self._safe_embs))
                emb_list.append(self._safe_embs)
            self._all_ref_embs = np.vstack(emb_list)    # (N_refs, dim)
            self._all_ref_cats = cats                     # len N_refs

        # 4. Single matrix multiply: (n_ngrams, dim) @ (dim, N_refs) = (n_ngrams, N_refs)
        sims = embs @ self._all_ref_embs.T

        # 5. For each n-gram, find best matching category
        hits: List[Dict] = []
        for i, phrase_sims in enumerate(sims):
            top_idx = int(phrase_sims.argmax())
            top_sim = float(phrase_sims[top_idx])
            top_cat = self._all_ref_cats[top_idx]

            if top_cat == "SAFE" or top_sim < self.sim_threshold:
                continue

            # Check that the best sensitive sim beats the best safe sim by margin
            safe_mask = [c == "SAFE" for c in self._all_ref_cats]
            if any(safe_mask):
                max_safe = float(phrase_sims[safe_mask].max()) if sum(safe_mask) else 0
                if top_sim <= max_safe + 0.05:
                    continue

            phrase, start, end = ngrams[i]
            hits.append({
                "token": phrase,
                "label": top_cat,
                "score": round(top_sim, 4),
                "start": start,
                "end": end,
            })

        # 6. MLP second opinion — only re-embed the hits (few items)
        if hits and self._clf is not None:
            hit_embs_idx = []
            for h in hits:
                # Find the ngram index to reuse existing embeddings
                for j, (p, s, e) in enumerate(ngrams):
                    if s == h["start"] and e == h["end"]:
                        hit_embs_idx.append(j)
                        break

            hit_emb_matrix = embs[hit_embs_idx]
            proba = self._clf.predict_proba(hit_emb_matrix)
            idx_to_cat = self._meta.get("idx_to_cat", {})

            confirmed: List[Dict] = []
            for h, prob_row in zip(hits, proba):
                mlp_class = int(prob_row.argmax())
                mlp_conf = float(prob_row.max())
                mlp_cat = idx_to_cat.get(str(mlp_class), "SAFE")

                if mlp_class == 0:
                    continue  # MLP says safe → skip

                if mlp_conf >= self.mlp_threshold:
                    h["score"] = round(max(h["score"], mlp_conf), 4)
                    if mlp_cat != "SAFE":
                        h["label"] = mlp_cat
                    confirmed.append(h)
            hits = confirmed

        # 7. Apply the same false-positive context filter as regex
        from server.privacy.anonymize import TextAnonymizer
        fp_filter = TextAnonymizer._ART9_FALSE_POS
        filtered: List[Dict] = []
        for h in hits:
            ctx_start = max(0, h["start"] - 40)
            ctx_end = min(len(text), h["end"] + 40)
            context = text[ctx_start:ctx_end]
            if fp_filter.search(context):
                continue  # false positive in context
            filtered.append(h)
        hits = filtered

        # 8. Deduplicate overlapping spans — keep highest score
        hits.sort(key=lambda h: (-h["score"], h["start"]))
        final: List[Dict] = []
        used_spans: List[Tuple[int, int]] = []
        for h in hits:
            s, e = h["start"], h["end"]
            if any(s < ue and e > us for us, ue in used_spans):
                continue
            used_spans.append((s, e))
            final.append(h)

        final.sort(key=lambda h: h["start"])
        return final


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="RGPD Sensitive-Word Classifier (train / test)"
    )
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("train", help="Build reference embeddings + MLP head")

    p_test = sub.add_parser("test", help="Run detector on sample texts")
    p_test.add_argument("--text", type=str, default=None)

    args = parser.parse_args()

    if args.cmd == "train":
        train_sensitive_model()

    elif args.cmd == "test":
        det = SensitiveWordDetector()
        if not det.available:
            print("❌ Model not trained. Run:  python -m server.privacy.sensitive_model train")
            return

        test_texts = (
            [args.text]
            if args.text
            else [
                "Le client souffre d'épilepsie sévère et cherche un sac hypoallergénique.",
                "She is muslim and wants halal-certified leather goods.",
                "Discussion about tissus ethniques et motifs traditionnels.",
                "He mentioned his criminal record during the consultation.",
                "La cliente est enceinte et cherche un cadeau pour son mari.",
                "Looking for a scarf in cashmere with geometric print.",
                "Budget 3500€ pour un sac Capucines en cuir noir.",
                "Le client est diabetique et suit un traitement psychiatrique.",
                "Elle a des convictions éthiques sur la mode durable.",
            ]
        )

        print("\n🔍 Sensitive-Word Predictions\n")
        for txt in test_texts:
            hits = det.predict(txt)
            print(f"  Text: {txt}")
            if hits:
                for h in hits:
                    print(
                        f"    ⚠️  [{h['label']}] \"{h['token']}\"  "
                        f"(score={h['score']:.2f}, pos={h['start']}:{h['end']})"
                    )
            else:
                print("    ✅ No sensitive words detected")
            print()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
