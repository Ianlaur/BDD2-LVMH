"""Regex-based budget and amount extraction from text.

Extracts budget mentions, price ranges, and monetary amounts in multiple currencies.
Returns structured results with evidence spans for traceability.
"""
from __future__ import annotations
import re

# Budget keyword pattern — matches amounts, ranges, currencies in FR/EN/ES/IT/DE
# Uses a lazy filler to absorb qualifiers like "autour de", "around", "environ", "circa"
_BUDGET_RE = re.compile(
    r"(?:budget|prix|price|prezzo|precio|preis|presupuesto|orçamento)"
    r"[\s:]*"
    r"(?:(?:autour|around|environ|circa|about|ungefähr|alrededor)\s+(?:de|of|di)?\s*)?"
    r"((?:€|EUR|USD|\$|£|¥)?\s*\d[\d\s.,]*\d*\s*(?:k|K|€|EUR|USD|\$|£|¥|euros?|dollars?)?"
    r"(?:\s*[-–àa]\s*(?:€|EUR|USD|\$|£|¥)?\s*\d[\d\s.,]*\d*\s*(?:k|K|€|EUR|USD|\$|£|¥|euros?|dollars?)?)?"
    r"(?:\s*\+)?)",
    re.IGNORECASE,
)

# Standalone amount pattern (€5000, $3000, 25K€, 5000 euros)
_AMOUNT_RE = re.compile(
    r"(?:€|EUR |USD |\$|£)\s*\d[\d\s.,]*\d*\s*(?:k|K)?\s*(?:\+)?"
    r"|"
    r"\d[\d\s.,]*\d*\s*(?:€|EUR|USD|\$|£|euros?|dollars?)\s*(?:\+)?",
    re.IGNORECASE,
)


def extract_budgets(text: str) -> list[dict]:
    """Extract budget mentions from text.

    Returns list of dicts with keys:
        - amount_text: the matched amount string
        - start: character offset start (of the full match in text)
        - end: character offset end (of the full match in text)
        - concept_id: "budget_intelligence.amount.detected"
    """
    results: list[dict] = []
    # Track all matched intervals (start, end) to suppress sub-span duplicates
    matched_intervals: list[tuple[int, int]] = []

    def _is_subsumed(s: int, e: int) -> bool:
        """Return True if [s, e) is fully contained within an already-recorded interval."""
        return any(ms <= s and e <= me for ms, me in matched_intervals)

    for pattern in [_BUDGET_RE, _AMOUNT_RE]:
        for match in pattern.finditer(text):
            start, end = match.start(), match.end()

            # For _BUDGET_RE, the captured group is the amount substring only
            if match.lastindex and match.group(1):
                amount_text = match.group(1).strip()
            else:
                amount_text = match.group(0).strip()

            # Skip if this span is fully covered by a prior (longer) match
            if _is_subsumed(start, end):
                continue

            matched_intervals.append((start, end))

            if amount_text:
                results.append({
                    "amount_text": amount_text,
                    "start": start,
                    "end": end,
                    "concept_id": "budget_intelligence.amount.detected",
                })

    results.sort(key=lambda r: r["start"])
    return results
