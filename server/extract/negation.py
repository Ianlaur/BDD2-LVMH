"""Multilingual negation detection with scope limiting.

Detects negation patterns and returns character spans of negated content.
Negation scope is limited to the next 5 tokens after the negation word.
"""
from __future__ import annotations
import re

# Negation triggers per language (lowercase)
NEGATION_PATTERNS: dict[str, list[str]] = {
    "fr": [
        r"\bpas\s+de\b", r"\bsans\b", r"\bne\s+\w+\s+pas\b",
        r"\bne\s+\w+\s+jamais\b", r"\baucun[e]?\b", r"\bjamais\s+de\b",
        r"\bn['']aime\s+pas\b", r"\bne\s+veut\s+pas\b",
    ],
    "en": [
        r"\bno\b", r"\bnot\b", r"\bnever\b", r"\bwithout\b",
        r"\bdon['']t\b", r"\bdoesn['']t\b", r"\bdidn['']t\b",
        r"\bnone\b",
    ],
    "it": [
        r"\bnon\b", r"\bsenza\b", r"\bnessun[oa]?\b", r"\bmai\b",
    ],
    "es": [
        r"\bno\b", r"\bsin\b", r"\bningún[oa]?\b", r"\bnunca\b", r"\bjamás\b",
    ],
    "de": [
        r"\bnicht\b", r"\bkein[e]?\b", r"\bohne\b", r"\bnie\b", r"\bniemals\b",
    ],
    "pt": [
        r"\bnão\b", r"\bsem\b", r"\bnenhum[a]?\b", r"\bnunca\b",
    ],
}

# Scope: how many tokens after negation trigger to include
NEGATION_SCOPE_TOKENS = 5

# Conjunction/clause-break words that stop negation scope
SCOPE_BREAKERS = {
    "mais", "but", "however", "ma", "pero", "aber", "porém",
    "et", "and", "e", "y", "und",
}

# Punctuation characters that break scope
SCOPE_BREAK_PUNCT = set(",.;!?")


def _tokenize_with_positions(text: str, start: int) -> list[tuple[str, int, int]]:
    """Tokenize the text from `start` offset, returning (token, char_start, char_end) tuples.

    Punctuation characters are returned as individual tokens so they can act
    as scope breakers.
    """
    tokens: list[tuple[str, int, int]] = []
    i = start
    n = len(text)
    while i < n:
        # Skip whitespace
        if text[i].isspace():
            i += 1
            continue
        # Punctuation token
        if text[i] in SCOPE_BREAK_PUNCT:
            tokens.append((text[i], i, i + 1))
            i += 1
            continue
        # Word token
        j = i
        while j < n and not text[j].isspace() and text[j] not in SCOPE_BREAK_PUNCT:
            j += 1
        tokens.append((text[i:j], i, j))
        i = j
    return tokens


def detect_negation_spans(text: str, lang: str = "fr") -> list[dict]:
    """Detect negation spans in text.

    Returns list of {"start": int, "end": int, "trigger": str} dicts,
    where start/end are character offsets of the negated content (excluding
    the trigger itself).
    """
    lang = lang.lower()[:2]
    patterns = NEGATION_PATTERNS.get(lang, NEGATION_PATTERNS.get("en", []))
    text_lower = text.lower()
    results = []

    for pattern in patterns:
        for match in re.finditer(pattern, text_lower):
            trigger_end = match.end()

            # Tokenize the remainder of the text from trigger_end
            tokens = _tokenize_with_positions(text, trigger_end)

            scope_end = trigger_end
            token_count = 0

            for token_str, tok_start, tok_end in tokens:
                # Check for scope-breaking punctuation
                if token_str in SCOPE_BREAK_PUNCT:
                    break

                # Check for scope-breaking conjunction
                clean = token_str.lower()
                if clean in SCOPE_BREAKERS:
                    break

                scope_end = tok_end
                token_count += 1
                if token_count >= NEGATION_SCOPE_TOKENS:
                    break

            if scope_end > trigger_end:
                # Trim leading whitespace from negated span
                span_start = trigger_end
                while span_start < scope_end and text[span_start] == " ":
                    span_start += 1
                results.append({
                    "start": span_start,
                    "end": scope_end,
                    "trigger": text[match.start():match.end()],
                })

    # Sort by start position and remove overlaps
    results.sort(key=lambda r: r["start"])
    merged: list[dict] = []
    for r in results:
        if merged and r["start"] < merged[-1]["end"]:
            merged[-1]["end"] = max(merged[-1]["end"], r["end"])
        else:
            merged.append(r)

    return merged
