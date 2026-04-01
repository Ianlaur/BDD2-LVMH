"""Aho-Corasick v2 extraction using hierarchical ontology.

Builds an automaton from the ontology's concept aliases, filtered by stopwords.
Returns matches with concept IDs, evidence spans, and weights.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field

from ahocorasick_rs import AhoCorasick, MatchKind

from server.ontology.schema import Ontology

# Suffixes that may follow a matched alias and still constitute a valid match.
# E.g. "vintage watch" should match within "vintage watches" (suffix: "es").
_VALID_SUFFIXES = re.compile(r"^(s|es|ed|ing|er|eur|euse|aux|iste|e|é|ée|ées|és)?(\W|$)")


@dataclass
class AutomatonV2:
    """Compiled Aho-Corasick automaton with pattern metadata."""
    automaton: AhoCorasick
    patterns: list[str]                     # index → alias string
    pattern_concept_id: list[str]           # index → concept ID
    pattern_weight: list[float]             # index → concept weight
    word_boundary_re: re.Pattern = field(default_factory=lambda: re.compile(r"\b"))


def build_automaton_v2(
    ontology: Ontology,
    stopwords: dict[str, list[str]] | None = None,
    min_alias_length: int = 3,
) -> AutomatonV2:
    """Build Aho-Corasick automaton from ontology concepts.

    Args:
        ontology: The hierarchical ontology
        stopwords: {"lang_code": ["word1", "word2"]} — aliases to exclude
        min_alias_length: minimum character length for an alias to be included
    """
    stopwords = stopwords or {}
    # Flatten all stopwords into a single lowercase set
    stop_set: set[str] = set()
    for words in stopwords.values():
        for w in words:
            stop_set.add(w.lower().strip())

    patterns: list[str] = []
    pattern_concept_id: list[str] = []
    pattern_weight: list[float] = []

    for concept in ontology.all_concepts():
        for alias in concept.all_aliases():
            alias_lower = alias.lower().strip()
            if len(alias_lower) < min_alias_length:
                continue
            if alias_lower in stop_set:
                continue
            patterns.append(alias_lower)
            pattern_concept_id.append(concept.id)
            pattern_weight.append(concept.weight)

    automaton = AhoCorasick(patterns, matchkind=MatchKind.LeftmostLongest)

    return AutomatonV2(
        automaton=automaton,
        patterns=patterns,
        pattern_concept_id=pattern_concept_id,
        pattern_weight=pattern_weight,
    )


def extract_concepts_v2(
    auto: AutomatonV2,
    text: str,
    lang: str = "fr",
) -> list[dict]:
    """Extract concepts from text using the v2 automaton.

    Returns list of dicts:
        - concept_id: str (e.g., "product_affinity.watches.vintage_watch")
        - matched_alias: str
        - start: int (character offset)
        - end: int (character offset)
        - weight: float
        - dimension: str (first part of concept_id)
    """
    text_lower = text.lower()
    text_len = len(text_lower)

    # Pre-compute word boundary positions for start-of-match checking
    boundaries: set[int] = set()
    for m in auto.word_boundary_re.finditer(text_lower):
        boundaries.add(m.start())

    raw_matches = auto.automaton.find_matches_as_indexes(text_lower)
    seen_concepts: set[str] = set()
    results: list[dict] = []

    for pat_idx, start, end in raw_matches:
        # Word boundary check at START of match
        if start not in boundaries:
            continue

        # Word boundary or valid suffix check at END of match.
        # The automaton may match a prefix of a longer word (e.g. "vintage watch"
        # inside "vintage watches").  We allow the match if the character(s)
        # immediately after the match form a known morphological suffix or a
        # non-word character / end-of-string.
        if end not in boundaries:
            tail = text_lower[end:]
            if not _VALID_SUFFIXES.match(tail):
                continue

        concept_id = auto.pattern_concept_id[pat_idx]

        # Deduplicate: keep first occurrence of each concept
        if concept_id in seen_concepts:
            continue
        seen_concepts.add(concept_id)

        results.append({
            "concept_id": concept_id,
            "matched_alias": auto.patterns[pat_idx],
            "start": start,
            "end": end,
            "weight": auto.pattern_weight[pat_idx],
            "dimension": concept_id.split(".")[0],
        })

    results.sort(key=lambda r: r["start"])
    return results
