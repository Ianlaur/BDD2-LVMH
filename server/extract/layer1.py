"""Layer 1: Deterministic extraction combining Aho-Corasick v2, budget regex, and negation detection.

This is the first extraction layer. All outputs are tier="stated" (high confidence).
Concepts found within negation spans are moved to rejections.
"""
from __future__ import annotations
import json
from pathlib import Path

from server.ontology.schema import Ontology
from server.ontology.loader import load_ontology
from server.extract.aho_v2 import build_automaton_v2, extract_concepts_v2, AutomatonV2
from server.extract.budget_regex import extract_budgets
from server.extract.negation import detect_negation_spans
from server.shared.config import TAXONOMY_DIR


def _load_stopwords() -> dict[str, list[str]]:
    path = TAXONOMY_DIR / "stopwords.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


# Module-level cache
_cached_automaton: AutomatonV2 | None = None
_cached_ontology: Ontology | None = None


def get_automaton(ontology: Ontology | None = None) -> AutomatonV2:
    """Get or build the cached automaton."""
    global _cached_automaton, _cached_ontology
    if _cached_automaton is not None and ontology is _cached_ontology:
        return _cached_automaton
    if ontology is None:
        ontology = load_ontology(TAXONOMY_DIR / "ontology_v2.json")
    stopwords = _load_stopwords()
    _cached_automaton = build_automaton_v2(ontology, stopwords=stopwords)
    _cached_ontology = ontology
    return _cached_automaton


def extract_layer1(
    text: str,
    ontology: Ontology | None = None,
    lang: str = "fr",
) -> dict:
    """Run Layer 1 deterministic extraction.

    Returns:
        {
            "concepts": [{"concept_id", "matched_alias", "start", "end", "weight", "dimension", "tier"}],
            "budgets": [{"amount_text", "start", "end", "concept_id", "tier"}],
            "negations": [{"start", "end", "trigger"}],
            "rejections": [{"concept_id", "matched_alias", "start", "end", "weight", "dimension", "tier"}],
        }
    """
    auto = get_automaton(ontology)

    # 1. Detect negation spans
    negation_spans = detect_negation_spans(text, lang=lang)

    # 2. Extract concepts via Aho-Corasick
    raw_concepts = extract_concepts_v2(auto, text, lang=lang)

    # 3. Extract budgets
    budgets = extract_budgets(text)
    for b in budgets:
        b["tier"] = "stated"

    # 4. Separate positive concepts from negated ones
    concepts = []
    rejections = []
    for c in raw_concepts:
        c["tier"] = "stated"
        is_negated = False
        for neg in negation_spans:
            # Check if concept's matched span overlaps with negation span
            if c["start"] >= neg["start"] and c["start"] < neg["end"]:
                is_negated = True
                break
        if is_negated:
            rejections.append(c)
        else:
            concepts.append(c)

    return {
        "concepts": concepts,
        "budgets": budgets,
        "negations": negation_spans,
        "rejections": rejections,
    }
